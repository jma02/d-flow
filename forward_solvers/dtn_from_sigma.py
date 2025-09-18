import torch
from torch import Tensor
from forward_solvers.fem import Mesh, V_h, dtn_map
from forward_solvers.utils import generate_GCOORD, assemble_EL_connectivity, interpolate_pts, interpolate_pts_torch

def dtn_from_sigma(
        sigma_vec: Tensor, # the image in question
        v_h: V_h,
        mesh: Mesh,
        img_size: int,
        device: str
) -> Tensor:
    centroids = torch.mean(mesh.p[mesh.t], dim=1)

    sigma_vec_grid = interpolate_pts_torch(sigma_vec, centroids, img_size, device=device).to(device)
    dtn_data, sol = dtn_map(v_h, sigma_vec_grid)
    return dtn_data


def dtn_from_sigma_scipy(
        sigma_vec: Tensor, # the image in question
        v_h: V_h,
        mesh: Mesh,
        original_size: int,
        pad_size: int,
) -> Tensor:
     img_size_down = original_size + 2 * pad_size
     x = torch.linspace(-1, 1, img_size_down, dtype=torch.float64)
     y = torch.linspace(-1, 1, img_size_down, dtype=torch.float64)
     xx, yy = torch.meshgrid(x, y, indexing='ij')
     img_points = torch.stack([xx.ravel(), yy.ravel()]).T
     
     # Generate a mesh for downsampling the original image. 
     GCOORD_down = img_points.reshape((img_size_down, img_size_down, 2))
     GCOORD_down = torch.flip(GCOORD_down, dims=[0])
     GCOORD_down = GCOORD_down.reshape((-1, 2))
     centroids = torch.mean(mesh.p[mesh.t], dim=1)

     centroids = centroids.to("cpu")
     sigma_vec = sigma_vec.to("cpu")
     GCOORD_down = GCOORD_down.to("cpu")
     
     sigma_vec_true = torch.from_numpy(interpolate_pts(GCOORD_down, sigma_vec.flatten(), centroids))
     dtn_data, sol = dtn_map(v_h, sigma_vec_true)
     return dtn_data

    