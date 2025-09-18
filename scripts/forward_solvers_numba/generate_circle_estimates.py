import os
import time
import numpy as np
import scipy
import math
import click

import scipy.optimize as op
from scipy.optimize import Bounds
import scipy.io as sio

from eit import EIT
from fem import Mesh, V_h, dtn_map
from utils import generate_GCOORD, assemble_EL_connectivity, interpolate_pts

# Generate a Four Circles sigma instance. 
def generate_circles_sigma(p, selected):

    center_1 = np.array([np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)])
    center_2 = np.array([np.random.uniform(0.1, 0.4), np.random.uniform(-0.4, -0.1)])
    center_3 = np.array([np.random.uniform(-0.4, -0.1), np.random.uniform(0.1, 0.4)])
    center_4 = np.array([np.random.uniform(-0.4, -0.1), np.random.uniform(-0.4, -0.1)])

    r1 = np.random.uniform(0.1, 0.4)
    r2 = np.random.uniform(0.1, 0.4)
    r3 = np.random.uniform(0.1, 0.4)
    r4 = np.random.uniform(0.1, 0.4)

    total = np.ones(p.shape[0], dtype=np.float64)
    if 1 in selected:
        cond1 = np.sqrt(np.sum((p - center_1)**2, axis=1)) < r1
        total += 2 * cond1
    if 2 in selected:
        cond2 = np.sqrt(np.sum((p - center_2)**2, axis=1)) < r2
        total += 4*cond2
    if 3 in selected:
        cond3 = np.sqrt(np.sum((p - center_3)**2, axis=1)) < r3
        total += 6*cond3
    if 4 in selected:
        cond4 = np.sqrt(np.sum((p - center_4)**2, axis=1)) < r4
        total += 8*cond4
    return total


def generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise):

    dtn_data, sol = dtn_map(v_h, sigma_vec_true)

    # add desired noise
    noise_data = np.random.uniform(-noise, noise, dtn_data.shape) * dtn_data
    dtn_data = dtn_data + noise_data

    # initial guess: 1 is value of background medium
    sigma_vec_0 = 1. + np.zeros(mesh.t.shape[0], dtype=np.float64)

    eit = EIT(v_h)
    eit.update_matrices(sigma_vec_0)

    def J(x):
        return eit.misfit(dtn_data, x)
    
    opt_tol = 1e-30

    bounds_l = [1. for _ in range(len(sigma_vec_0))]
    bounds_r = [np.inf for _ in range(len(sigma_vec_0))]
    bounds = Bounds(bounds_l, bounds_r)

    t_i = time.time()
    res = op.minimize(J, sigma_vec_0, method='L-BFGS-B',
                      jac = True,
                      tol = opt_tol,
                      bounds=bounds, 
                      options={'maxiter': num_iters,
                                'disp': False, 'ftol':opt_tol, 'gtol':opt_tol}, 
                     )
                       # callback=callback)

    # t_f = time.time()

    return res.x

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
    
    p = np.array(mat_contents['p'])
    t = np.array(mat_contents['t']-1) 
    vol_idx = mat_contents['vol_idx'].reshape((-1,))-1 
    bdy_idx = mat_contents['bdy_idx'].reshape((-1,))-1 
    
    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)
    
    centroids = np.mean(p[t], axis=1)  
    
    sigma_true = np.zeros((num_samples, len(centroids)))
    sigma_pred = np.zeros((num_samples, len(centroids)))
    imgs_true = np.zeros((num_samples, img_size, img_size))
    imgs_pred = np.zeros((num_samples, img_size, img_size))


    save_name = f"circles_bfgs_{str(num_iters)}_res_{str(img_size)}_noise_{str(noise)}"
    save_path = os.path.join(data_root, save_name)
    
    for i in range(num_samples):
        k = np.random.randint(1, 5)
        selected = np.random.choice(np.arange(1, 6), size=k, replace=False)
        sigma_vec_true = generate_circles_sigma(centroids, selected)
 
        t_i = time.time()
        sigma_vec_pred = generate_EIT_sol(num_iters, mesh, v_h, sigma_vec_true, noise)
        
        sq_img_true = 1. + np.zeros((nx-1) * (ny-1))
        sq_img_pred = 1. + np.zeros((nx-1) * (ny-1))

        interp_vals_true = interpolate_pts(centroids, sigma_vec_true, GCOORD)
        interp_vals_pred = interpolate_pts(centroids, sigma_vec_pred, GCOORD)
        for iel in range(0,nel):
            ECOORD_true = np.take(interp_vals_true, EL2NOD[iel, :], axis=0)
            ECOORD_pred = np.take(interp_vals_pred, EL2NOD[iel, :], axis=0)
            
            #based on ECOORD pts, average them out to find pixel value 
            sq_img_true[iel] = np.mean(ECOORD_true)
            sq_img_pred[iel] = np.mean(ECOORD_pred)
            
        t_f = time.time()
    
        sq_img_true = np.flip(sq_img_true.reshape((nx-1, ny-1)), axis=0)
        sq_img_pred = np.flip(sq_img_pred.reshape((nx-1, ny-1)), axis=0)
        
        sigma_true[i, ...] = sigma_vec_true
        sigma_pred[i, ...] = sigma_vec_pred
        imgs_true[i, ...] = sq_img_true
        imgs_pred[i, ...] = sq_img_pred
        
        if i % 100 == 0:
            print(f'Time elapsed is {(t_f - t_i):.4f}', flush=True)
            print(i, flush=True)
            npy_name = save_path
            np.savez(npy_name, imgs_true=imgs_true, imgs_pred=imgs_pred, sigma_true=sigma_true, sigma_pred=sigma_pred)
        
    np.savez(save_path, imgs_true=imgs_true, imgs_pred=imgs_pred, sigma_true=sigma_true, sigma_pred=sigma_pred)
        

if __name__ == "__main__":
    main()
    