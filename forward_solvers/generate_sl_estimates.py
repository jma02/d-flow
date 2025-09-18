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
from utils_shepp_logan import randomSheppLogan

# Generate a Shepp-Logan sigma instance. 
def generate_shepp_logan_sigma(n, pad, GCOORD_down, centroids):
    sl = randomSheppLogan(n=n, pad=pad, M=1).reshape((n + 2* pad, n + 2* pad))
    sl_sigma = interpolate_pts(GCOORD_down.cpu().numpy(), sl.flatten(), centroids.cpu().numpy())
    return torch.tensor(sl_sigma, dtype=centroids.dtype, device=centroids.device)


def generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise):

    dtn_data, sol = dtn_map(v_h, sigma_vec_true)

    # add desired noise
    noise_data = torch.rand(dtn_data.shape, device=dtn_data.device, dtype=dtn_data.dtype) * 2 * noise - noise
    noise_data = noise_data * dtn_data
    dtn_data = dtn_data + noise_data

    # initial guess: 1 is value of background medium
    sigma_vec_0 = 1. + torch.zeros(mesh.t.shape[0], dtype=torch.float64, device=sigma_vec_true.device)

    eit = EIT(v_h)
    eit.update_matrices(sigma_vec_0)

    def J(x):
        return eit.misfit(dtn_data, x)
    
    opt_tol = 1e-30

    bounds_l = [1. for _ in range(len(sigma_vec_0))]
    bounds_r = [2 for _ in range(len(sigma_vec_0))]
    bounds = Bounds(bounds_l, bounds_r)

    # t_i = time.time()
    res = op.minimize(J, sigma_vec_0, method='L-BFGS-B',
                      jac = True,
                      tol = opt_tol,
                      bounds=bounds, 
                      options={'maxiter': num_iters,
                                'disp': False, 'ftol':opt_tol, 'gtol':opt_tol}, 
                     )
                       # callback=callback)

    # t_f = time.time()

    return torch.tensor(res.x, dtype=torch.float64, device=sigma_vec_true.device)

@click.command()
@click.option('--img-size', type=int, required=True, help='size of output image')
@click.option('--num-samples', type=int, required=True, help='number of samples')
@click.option('--noise', type=float, required=True, help='noise level')
@click.option('--num-iters', type=int, required=True, help='max number of BFGS iterations')
@click.option('--original-size', type=int, required=True, help='size of original image')
@click.option('--pad-size', type=int, required=True, help='size of padding')
@click.option('--data-root', type=str, required=True, help='root directory for the dataset')
@click.option('--mesh-file', type=str, required=True, help='name of the mesh file')
def main(
    img_size: int,
    num_samples: int,
    noise: float,
    num_iters: int,
    original_size: int,
    pad_size: int,
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

    img_size_down = original_size + 2 * pad_size
    x = torch.linspace(-1, 1, img_size_down, dtype=torch.float64)
    y = torch.linspace(-1, 1, img_size_down, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    img_points = torch.stack([xx.ravel(), yy.ravel()]).T

    # Generate a mesh for downsampling the original image. 
    GCOORD_down = img_points.reshape((img_size_down, img_size_down, 2))
    GCOORD_down = torch.flip(GCOORD_down, dims=[0])
    GCOORD_down = GCOORD_down.reshape((-1, 2))

    #generate square mesh and element connectivity
    GCOORD = torch.tensor(generate_GCOORD(lx, ly, nx, ny), dtype=torch.float64)
    EL2NOD = torch.tensor(assemble_EL_connectivity(nnod, nnodel, nex, nx), dtype=torch.long)

    mat_fname  = os.path.join(data_root, mesh_file)
    mat_contents = sio.loadmat(mat_fname)
    
    p = torch.tensor(mat_contents['p'], dtype=torch.float64)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.long) 
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.long)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.long)
    
    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)
    
    centroids = torch.mean(p[t], dim=1)  
    
    sigma_true = torch.zeros((num_samples, len(centroids)), dtype=torch.float64)
    sigma_pred = torch.zeros((num_samples, len(centroids)), dtype=torch.float64)
    imgs_true = torch.zeros((num_samples, img_size, img_size), dtype=torch.float64)
    imgs_pred = torch.zeros((num_samples, img_size, img_size), dtype=torch.float64)

    save_name = f"sl_bfgs_{str(num_iters)}_res_{str(img_size)}_noise_{str(noise)}"
    save_path = os.path.join(data_root, save_name)
    
    for i in range(num_samples):
        sigma_vec_true = generate_shepp_logan_sigma(original_size, pad_size, GCOORD_down, centroids) + 1
 
        t_i = time.time()
        sigma_vec_pred = generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise)
        
        sq_img_true = 1. + torch.zeros((nx-1) * (ny-1), dtype=torch.float64)
        sq_img_pred = 1. + torch.zeros((nx-1) * (ny-1), dtype=torch.float64)

        interp_vals_true = torch.tensor(interpolate_pts(centroids, sigma_vec_true, GCOORD), dtype=torch.float64, device=centroids.device)
        interp_vals_pred = torch.tensor(interpolate_pts(centroids, sigma_vec_pred, GCOORD), dtype=torch.float64, device=centroids.device)
        for iel in range(0,nel):
            ECOORD_true = torch.take(interp_vals_true, EL2NOD[iel, :])
            ECOORD_pred = torch.take(interp_vals_pred, EL2NOD[iel, :])
            
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
            npy_name = save_path
            torch.save({'imgs_true': imgs_true, 'imgs_pred': imgs_pred, 'sigma_true': sigma_true, 'sigma_pred': sigma_pred}, npy_name + '.pt')
        
    torch.save({'imgs_true': imgs_true, 'imgs_pred': imgs_pred, 'sigma_true': sigma_true, 'sigma_pred': sigma_pred}, save_path + '.pt')
    

if __name__ == "__main__":
    main()
