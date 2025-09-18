import time
import torch
from torchmetrics import CatMetric
from torchdiffeq import odeint
import torch.optim as optim
from torch.autograd.functional import jacobian
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import cmocean
import numpy as np

import torch.nn as nn

from transforms import radon


def loss_fn(x, y):
    loss = torch.mean((x - y) ** 2)
    return loss


def ode_integrate(ode_func: nn.Module,
                  init_x: torch.Tensor,
                  ode_opts: Dict = {},
                  t_eps: float = 0,
                  init_t: float = 0.,
                  final_t: float = 1.,
                  t_arr: Optional[List[float]] = None,
                  num_steps: int = 5,
                  intermediate_points: bool = False
      ) -> torch.Tensor:

    if t_arr is None:
        t = torch.linspace(init_t - t_eps, final_t, num_steps).to(init_x.device)
    else:
        t = torch.tensor(t_arr, dtype=torch.float32, device=init_x.device)

    ode_opts = {
        "atol": 1e-5, 
        "rtol": 1e-5, 
        "method": "euler",
        **ode_opts
    }

    z = odeint(
        func = lambda t, x: ode_func(x, t.expand(x.shape[0])),
        y0 = init_x,
        t = t,
        **ode_opts
    )

    if not intermediate_points:
        return z[-1]
    return z

def dflow(
        model: nn.Module,
        train_min: float,
        train_max: float,
        init_x: Optional[torch.Tensor] = None,
        N: int = 50,
        max_iter: int = 20,
        optim_steps: int = 1000,
        patience: int = 100,
        optimizer: str = 'LBFGS',
        target_cost: float = .05,
        lr: float = 0.1,
        y: torch.Tensor = None,
        im_size: int = 128,
        device: str = 'cuda',
        track_singular_value_spectrum: bool = False,
        verbose: bool = False,
        history_size: int = 10
    ):
    x1_trajectory = []
    if init_x is None:
        init_x = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)
    init_x.requires_grad = True
    # history_size depends on your VRAM.
    if optimizer == 'LBFGS':
        optimizer = optim.LBFGS([init_x], max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe', history_size=history_size)
    else:
        optimizer = torch.optim.SGD([init_x], lr=lr, nesterov=True, momentum=0.9)

    # optimizer = optim.Adam([init_x], lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)

    svd_traj = []

    metrics = {'loss': CatMetric(), 
            'cost': CatMetric(), 
            'reg': CatMetric(),
            'norm_x0': CatMetric(), 
            'std_x0': CatMetric(), 
            'mean_x0': CatMetric(), 
            'time': CatMetric()
            }
    start_time = time.time()
    patience_ctr = 0
    best_loss = float('inf')
    x1 = ode_integrate(model, init_x, num_steps=5)
    x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
    x1_trajectory.append(x1.detach().cpu().numpy())
    loss = 0

    for step in range(optim_steps):
        def closure():
            nonlocal x1
            optimizer.zero_grad()

            reg_loss = torch.tensor(0.).to(init_x)

            ### solve for x1
            x1 = ode_integrate(ode_func=model, init_x =init_x, num_steps=5)
            # unnormalize x1
            x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
            
            # degraded_x1 = H(x1)
            degraded_x1 = radon.radon_transform(x1, N=N)
            cost = loss_fn(degraded_x1, y)
            loss = cost 

            norm_x0 = init_x.norm()
            std_x0 = init_x.std()
            mean_x0 = init_x.mean()

            metrics['norm_x0'].update(norm_x0.item())
            metrics['std_x0'].update(std_x0.item())
            metrics['mean_x0'].update(mean_x0.item())
            metrics['cost'].update(cost.item())
            metrics['reg'].update(reg_loss.item())
            metrics['loss'].update(loss.item())

            loss.backward()

            if (step % 10 == 0 or step == optim_steps - 1) and track_singular_value_spectrum:
                jac_dim = im_size * im_size
                with torch.no_grad():
                    # compute singular value decay of jacobian at x1
                    jacobian_x1 = jacobian(lambda x: ode_integrate(model, x, num_steps=5, intermediate_points=False).view(-1), inputs=x1, vectorize=False)
                    jacobian_x1 = jacobian_x1.view(jac_dim, jac_dim)

                    _, svd_x1, _ = torch.linalg.svd(jacobian_x1, full_matrices=False)
                    svd_traj.append(svd_x1)

            return loss


        optimizer.step(closure)
        
        x1_trajectory.append(x1.detach().cpu().numpy())


        elapsed = time.time() - start_time
        metrics['time'].update(elapsed)

        elapsed = elapsed/60
        if verbose:
            print(f"[Step {step}] Loss {loss.item()}"
                + f"| time: {elapsed} mins")

        if target_cost is not None:
            mets_cost = metrics['cost'].compute()
            if mets_cost.dim() > 0:
                mets_cost = mets_cost[-1]
            last_cost = mets_cost.item()
            if last_cost <= target_cost:
                print(f'reached cost of {last_cost}')
                break
        if loss < best_loss:
            best_loss = loss
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            print(f'Early stopping at step {step} with best loss {best_loss}')
            break

    return x1, metrics, svd_traj, x1_trajectory


# include full gif name if saving
def animate_iterates(x1_trajectory, 
                     gt, 
                     sparse_meas=None, # please input this as a padded image with the same shape as the gt measurement
                     interp_meas=None,
                     save_path="", 
                     save=True, 
                     forward_operator="radon", 
                     N=24
                     ):
    frame_indices = list(range(len(x1_trajectory))) 

    if sparse_meas is not None and interp_meas is not None:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 10))
    else:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    gt_img = gt.squeeze().cpu().numpy()
    if sparse_meas is not None and interp_meas is not None:
        sparse_meas = sparse_meas.squeeze()
        interp_meas = interp_meas.squeeze() 
        # check if meas is a tensor
        if isinstance(sparse_meas, torch.Tensor):
            sparse_meas = sparse_meas.cpu().numpy()
        if isinstance(interp_meas, torch.Tensor):
            interp_meas = interp_meas.cpu().numpy()

    gt_measurement = radon.radon_transform(gt_img, N=N)

    def animate_comparison(frame):
        try:
            # Clear all axes
            if sparse_meas is not None:
                for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
                    ax.clear()
            else:
                for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                    ax.clear()

            current_img = x1_trajectory[frame].squeeze()

            current_measurement = radon.radon_transform(current_img, N=N).squeeze()

            # Plot ground truth
            im1 = ax1.imshow(gt_img, cmap=cmocean.cm.dense, vmin=gt_img.min(), vmax=gt_img.max())
            ax1.set_title('Ground Truth', fontsize=12)
            ax1.axis('off')
            

            # Plot reconstruction
            im2 = ax2.imshow(current_img, cmap=cmocean.cm.dense, vmin=gt_img.min(), vmax=gt_img.max())
            ax2.set_title(f'Reconstruction - Step {frame}', fontsize=12)
            ax2.text(0.5, -0.1, f'L2 Rel. Error: {np.linalg.norm(current_img - gt_img) / np.linalg.norm(gt_img):.4f}', fontsize=12,
                ha='center', va='top', transform=ax2.transAxes)
            ax2.axis('off')

            # Plot difference
            diff = current_img - gt_img
            im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            ax3.set_title(f'Difference', fontsize=12)
            ax3.text(0.5, -0.1, f'Max error: {np.abs(diff).max():.4f}', fontsize=12,
                ha='center', va='top', transform=ax3.transAxes)
            ax3.axis('off')

            # Find common measurement scale
            meas_min = min(gt_measurement.min(), current_measurement.min())
            meas_max = max(gt_measurement.max(), current_measurement.max())

            im4 = ax4.imshow(gt_measurement, cmap=cmocean.cm.dense, vmin=meas_min, vmax=meas_max)
            ax4.set_title('Ground Truth Measurement', fontsize=12)
            ax4.axis('off')

            im5 = ax5.imshow(current_measurement, cmap=cmocean.cm.dense, vmin=meas_min, vmax=meas_max)
            ax5.set_title(f'Reconstruction Measurement - Step {frame}', fontsize=12)
            ax5.text(0.5, -0.1, f'MSE: {np.mean((current_measurement - gt_measurement)**2)}', fontsize=12,
                ha='center', va='top', transform=ax5.transAxes)
            ax5.axis('off')

            diff_measurement = current_measurement - gt_measurement
            im6 = ax6.imshow(diff_measurement, cmap='RdBu_r', vmin=-np.abs(diff_measurement).max(), vmax=np.abs(diff_measurement).max())
            ax6.set_title(f'Difference Measurement - Step {frame}', fontsize=12)
            ax6.text(0.5, -0.1, f'Max error: {np.abs(diff_measurement).max()}', fontsize=12,
                ha='center', va='top', transform=ax6.transAxes)
            ax6.axis('off')

            if sparse_meas is not None and interp_meas is not None:
                im7 = ax7.imshow(sparse_meas, cmap=cmocean.cm.dense, vmin=meas_min, vmax=meas_max)
                ax7.set_title('Sparse Measurement', fontsize=12)
                ax7.axis('off')

                im8 = ax8.imshow(interp_meas, cmap=cmocean.cm.dense, vmin=meas_min, vmax=meas_max)
                ax8.set_title(f'Interpolated Measurement', fontsize=12)
                ax8.text(0.5, -0.1, f'MSE: {np.mean((interp_meas - gt_measurement)**2)}', fontsize=12,
                    ha='center', va='top', transform=ax8.transAxes)
                ax8.axis('off')

                sparse_diff = gt_measurement - interp_meas 
                im9 = ax9.imshow(sparse_diff, cmap='RdBu_r', vmin=-np.abs(sparse_diff).max(), vmax=np.abs(sparse_diff).max())
                ax9.set_title(f'Difference Interp. Meas. Vs. Gt', fontsize=12)
                ax9.text(0.5, -0.1, f'Max error: {np.abs(sparse_diff).max()}', fontsize=12,
                    ha='center', va='top', transform=ax9.transAxes)
                ax9.axis('off')
                

            if sparse_meas is not None and interp_meas is not None:
                return [im1, im2, im3, im4, im5, im6, im7, im8, im9]
            else:
                return [im1, im2, im3, im4, im5, im6]
        except Exception as e:
            print(f"Error in frame {frame}: {e}")
            return []


    # Use same frame sampling as before
    anim_comparison = animation.FuncAnimation(fig, animate_comparison, frames=frame_indices, 
                                            interval=300, blit=False, repeat=True)

    # Save as GIF if requested
    if save:
        gif_filename = f'{save_path}'
        anim_comparison.save(gif_filename, writer='pillow', fps=4)
        # print(f"Animation saved as {gif_filename}")

    plt.close(fig)  # Close the figure to prevent blank canvas display

    # Convert to HTML5 video for inline playback
    html_comparison = HTML(anim_comparison.to_jshtml())
    return html_comparison