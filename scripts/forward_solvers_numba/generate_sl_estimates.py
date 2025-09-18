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
from utils_shepp_logan import randomSheppLogan

# Generate a Shepp-Logan sigma instance. 
def generate_shepp_logan_sigma(n, pad, GCOORD_down, centroids):
    sl = randomSheppLogan(n=n, pad = pad, M = 1).reshape((n + 2* pad, n + 2* pad))
    sl_sigma = interpolate_pts(GCOORD_down, sl.flatten(), centroids)

    return sl_sigma


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

    return res.x

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
    x = np.linspace(-1, 1, img_size_down)
    y = np.linspace(-1, 1, img_size_down)
    xx, yy = np.meshgrid(x, y)
    img_points = np.stack([xx.ravel(), yy.ravel()]).T

    # Generate a mesh for downsampling the original image. 
    GCOORD_down = img_points.reshape((img_size_down, img_size_down, 2))
    GCOORD_down = np.flip(GCOORD_down, axis=0)
    GCOORD_down = GCOORD_down.reshape((-1, 2))

    #generate square mesh and element connectivity
    GCOORD = generate_GCOORD(lx, ly, nx, ny)
    EL2NOD = assemble_EL_connectivity(nnod, nnodel, nex, nx)

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

    save_name = f"sl_bfgs_{str(num_iters)}_res_{str(img_size)}_noise_{str(noise)}"
    save_path = os.path.join(data_root, save_name)
    
    for i in range(num_samples):
        sigma_vec_true = generate_shepp_logan_sigma(original_size, pad_size, GCOORD_down, centroids) + 1
 
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
    

# Check if the script is being run directly (not imported)
if __name__ == "__main__":
    main()