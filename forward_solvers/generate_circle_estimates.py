import os
import time
import torch
import click

import scipy.optimize as op
from scipy.optimize import Bounds
import scipy.io as sio

from eit import EIT
from fem import Mesh, V_h, dtn_map
from utils import generate_GCOORD, assemble_EL_connectivity, interpolate_pts

# Generate a Four Circles sigma instance. 
def generate_circles_sigma(p, selected):

    center_1 = torch.tensor([torch.rand(1).item() * 0.3 + 0.1, torch.rand(1).item() * 0.3 + 0.1], device=p.device, dtype=p.dtype)
    center_2 = torch.tensor([torch.rand(1).item() * 0.3 + 0.1, torch.rand(1).item() * 0.3 - 0.4], device=p.device, dtype=p.dtype)
    center_3 = torch.tensor([torch.rand(1).item() * 0.3 - 0.4, torch.rand(1).item() * 0.3 + 0.1], device=p.device, dtype=p.dtype)
    center_4 = torch.tensor([torch.rand(1).item() * 0.3 - 0.4, torch.rand(1).item() * 0.3 - 0.4], device=p.device, dtype=p.dtype)

    r1 = torch.rand(1).item() * 0.3 + 0.1
    r2 = torch.rand(1).item() * 0.3 + 0.1
    r3 = torch.rand(1).item() * 0.3 + 0.1
    r4 = torch.rand(1).item() * 0.3 + 0.1

    total = torch.ones(p.shape[0], dtype=p.dtype, device=p.device)
    if 1 in selected:
        cond1 = torch.sqrt(torch.sum((p - center_1)**2, dim=1)) < r1
        total += 2 * cond1
    if 2 in selected:
        cond2 = torch.sqrt(torch.sum((p - center_2)**2, dim=1)) < r2
        total += 4*cond2
    if 3 in selected:
        cond3 = torch.sqrt(torch.sum((p - center_3)**2, dim=1)) < r3
        total += 6*cond3
    if 4 in selected:
        cond4 = torch.sqrt(torch.sum((p - center_4)**2, dim=1)) < r4
        total += 8*cond4
    return total


def generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise):

    dtn_data, sol = dtn_map(v_h, sigma_vec_true)

    # add desired noise - match dtype and device
    noise_data = (torch.rand(dtn_data.shape, device=dtn_data.device, dtype=dtn_data.dtype) * 2 - 1) * noise * dtn_data
    dtn_data = dtn_data + noise_data

    # initial guess: 1 is value of background medium
    sigma_vec_0 = 1. + torch.zeros(mesh.t.shape[0], dtype=torch.float64, device=mesh.device)

    eit = EIT(v_h)
    eit.update_matrices(sigma_vec_0)

    def J(x):
        return eit.misfit(dtn_data, x)
    
    opt_tol = 1e-30

    bounds_l = [1. for _ in range(len(sigma_vec_0))]
    bounds_r = [float('inf') for _ in range(len(sigma_vec_0))]
    bounds = Bounds(bounds_l, bounds_r)

    res = op.minimize(J, sigma_vec_0, method='L-BFGS-B',
                      jac = True,
                      tol = opt_tol,
                      bounds=bounds, 
                      options={'maxiter': num_iters,
                                'disp': False, 'ftol':opt_tol, 'gtol':opt_tol}, 
                     )

    return torch.tensor(res.x, dtype=torch.float64, device=mesh.device)

@click.command()
@click.option('--img-size', type=int, required=True, help='size of output image')
@click.option('--num-samples', type=int, required=True, help='number of samples')
@click.option('--noise', type=float, required=True, help='noise level')
@click.option('--num-iters', type=int, required=True, help='max number of BFGS iterations')
@click.option('--data-root', type=str, required=True, help='root directory for the dataset')
@click.option('--mesh-file', type=str, required=True, help='name of the mesh file')
def main(
    img_size: int,
    num_samples: int,
    noise: float,
    num_iters: int,
    data_root: str,
    mesh_file: str,
):
    #geometry
    nx          = img_size + 1
    ny          = img_size + 1
    lx          = 2
    ly          = 2
    nnodel      = 4  #number of nodes per element
    
    # model parameters
    nex         = nx-1
    ney         = ny-1
    nnod        = nx*ny #number of nodes
    nel         = nex*ney #number of finite elements

    #generate square mesh and element connectivity
    GCOORD = generate_GCOORD(lx, ly, nx, ny)
    EL2NOD = assemble_EL_connectivity(nnod, nnodel, nex, nx)

    # Load the mesh. 
    mat_fname  = os.path.join(data_root, mesh_file)
    mat_contents = sio.loadmat(mat_fname)
    
    p = torch.tensor(mat_contents['p'], dtype=torch.float64)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.int64)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.int64)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.int64)

    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)

    centroids = torch.mean(p[t], dim=1)

    sigma_true = torch.zeros((num_samples, len(centroids)), dtype=torch.float64)
    sigma_pred = torch.zeros((num_samples, len(centroids)), dtype=torch.float64)
    imgs_true = torch.zeros((num_samples, img_size, img_size), dtype=torch.float64)
    imgs_pred = torch.zeros((num_samples, img_size, img_size), dtype=torch.float64)


    save_name = f"circles_bfgs_{str(num_iters)}_res_{str(img_size)}_noise_{str(noise)}"
    save_path = os.path.join(data_root, save_name)
    
    for i in range(num_samples):
        k = torch.randint(1, 5, (1,)).item()
        selected = torch.randperm(5)[:k]
        sigma_vec_true = generate_circles_sigma(centroids, selected)
 
        t_i = time.time()
        sigma_vec_pred = generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise)

        sq_img_true = 1. + torch.zeros((nx-1) * (ny-1), dtype=torch.float64, device=centroids.device)
        sq_img_pred = 1. + torch.zeros((nx-1) * (ny-1), dtype=torch.float64, device=centroids.device)

        interp_vals_true = torch.tensor(interpolate_pts(centroids, sigma_vec_true, GCOORD), dtype=torch.float64, device=centroids.device)
        interp_vals_pred = torch.tensor(interpolate_pts(centroids, sigma_vec_pred, GCOORD), dtype=torch.float64, device=centroids.device)
        for iel in range(0,nel):
            ECOORD_true = interp_vals_true[EL2NOD[iel, :]]
            ECOORD_pred = interp_vals_pred[EL2NOD[iel, :]]
            
            #based on ECOORD pts, average them out to find pixel value 
            sq_img_true[iel] = torch.mean(ECOORD_true)
            sq_img_pred[iel] = torch.mean(ECOORD_pred)
            
        t_f = time.time()
    
        sq_img_true = torch.flip(sq_img_true.reshape((nx-1, ny-1)), dims=[0])
        sq_img_pred = torch.flip(sq_img_pred.reshape((nx-1, ny-1)), dims=[0])
        
        sigma_true[i, ...] = sigma_vec_true
        sigma_pred[i, ...] = sigma_vec_pred
        imgs_true[i, ...] = sq_img_true
        imgs_pred[i, ...] = sq_img_pred
        
        if i % 100 == 0:
            print(f'Time elapsed is {(t_f - t_i):.4f}', flush=True)
            print(i, flush=True)
            checkpoint_name = save_path + ".pt"
            torch.save({
                'imgs_true': imgs_true,
                'imgs_pred': imgs_pred, 
                'sigma_true': sigma_true,
                'sigma_pred': sigma_pred
            }, checkpoint_name)
        
    # Final save
    final_save_name = save_path + ".pt"
    torch.save({
        'imgs_true': imgs_true,
        'imgs_pred': imgs_pred,
        'sigma_true': sigma_true, 
        'sigma_pred': sigma_pred
    }, final_save_name)
        

if __name__ == "__main__":
    main()
    