import torch
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from train_interpolant import get_indices
from transforms import radon
from dflow_func import dflow, animate_iterates
from unet import Unet
from utils import load_checkpoint
from torchdiffeq import odeint
import os
import argparse
import polars as pl

parser = argparse.ArgumentParser(description="Run the suite of tests for sparse to full data completion using our `naive` idea.")
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--problem', type=str, default='shepp-logan', help='Dataset to use')
parser.add_argument('--noise', type=float, default=0.0, help='amount of multiplicative noise to use')

args = parser.parse_args()

torch.manual_seed(77)
np.random.seed(77)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


problem = args.problem
device = args.device
noise = args.noise

# i am too lazy to change the variable names
im_size = 128
# load the media
shepps = torch.load(f"data/{problem}-dataset-128.pt")["test"]

# for our prior model
checkpoint_path = f'problems/{problem}/checkpoints/{im_size}x{im_size}/ckp_63000.tar'

train_min, train_max = torch.min(torch.load(f'data/{problem}-dataset-{im_size}.pt')["train"]), torch.max(torch.load(f'data/{problem}-dataset-{im_size}.pt')["train"])
model = Unet(ch=32).to(device)

_, _, model, _, _, _= load_checkpoint(model=model, path=checkpoint_path)
model.eval()
print("Loaded prior model")

# is this a machine learning sin?
models_dict = {}
subsample_rates = range(1,5)
n_full = 24
for n_sub in subsample_rates:
    interpolant_ckpt_path = f'problems/interpolant/{problem}/{n_sub}-{n_full}-{im_size}x{im_size}/checkpoints/ckp_9513.tar'
    interp_model = Unet(ch=32, ch_mul=[1, 2], att_channels=[0, 1]).to(device)
    _, _, interp_model, _, _, _= load_checkpoint(model=interp_model, path=interpolant_ckpt_path)
    interp_model.eval()
    models_dict[n_sub] = interp_model
print("Loaded interpolant models")

run = "sparse-to-full-metrics"
os.makedirs(f'dflow-radon-metrics/{run}-{problem}/noise{noise}', exist_ok=True)
os.chdir(f'dflow-radon-metrics/{run}-{problem}/noise{noise}')


init_x = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)

errors = {}
for n_sub in subsample_rates:
    errors[n_sub] = {"l2_rel_errors_interp": [], "l1_rel_errors_interp": [],
    "l2_rel_errors_no_interp": [], "l1_rel_errors_no_interp": []}

# animate every 500
animate_idx = 500

for i, shepp in enumerate(shepps):
    if i % animate_idx == 0:
        os.makedirs(f'idx{i}', exist_ok=True)
    shepp = shepp.to(device)
    shepp = shepp.squeeze(0)
    for n_sub in range(1,5):
        radon_shepp = radon.radon_transform(shepp, N=n_sub)
        radon_shepp = (1 + noise * torch.randn_like(radon_shepp)) * radon_shepp

        # just used in plotting
        full_meas = radon.radon_transform(shepp, N=n_full)
        full_meas = (1 + noise * torch.randn_like(full_meas)) * full_meas

        # if i % animate_idx == 0:
        #     plt.imsave(f'idx{i}/idx{i}-shepp-{n_sub}-meas-radon.png', radon_shepp.cpu().numpy(), cmap=cmocean.cm.dense)
        #     plt.imsave(f'idx{i}/idx{i}-shepp-{n_full}-full-meas-radon.png', full_meas.cpu().numpy(), cmap=cmocean.cm.dense)
        radon_shepp = radon_shepp.to(device)
        t = torch.linspace(0, 1, 5).to(device)
        padded_meas = torch.zeros((1, 1, n_full, radon_shepp.shape[-1]), device=device)
        indices = get_indices(n_sub, n_full)
        padded_meas[:, :, indices, :] = radon_shepp.unsqueeze(0).unsqueeze(0).to(device)
        
        interp_model = models_dict[n_sub]
        with torch.no_grad():
            interp_meas = odeint(
                    func = lambda t, x: interp_model(x, t.expand(x.shape[0])),
                    t = t,
                    y0 = padded_meas,
                    method = 'dopri5',
                    atol = 1e-5,
                    rtol = 1e-5,
                )[-1].squeeze() 
        # if i % animate_idx == 0:
        #     plt.imsave(f'idx{i}/idx{i}-shepp-{n_sub}-{n_full}-interp-radon.png', interp_meas.squeeze().cpu().numpy(), cmap=cmocean.cm.dense)

        reconstruct_with_interp, metrics_interp, _, x1_traj_interp = dflow(
            max_iter=20,
            optim_steps=100,
            target_cost=0.05,
            init_x=init_x.clone(),
            patience=30,
            model=model,
            lr=0.2,
            y=interp_meas.unsqueeze(0),
            train_min=train_min,
            train_max=train_max,
            N=n_full,
            optimizer="LBFGS",
            device=device
        )
        if i % animate_idx == 0:
            save_path = f'idx{i}/idx{i}-reconstruction_gif_{n_sub}-{n_full}-interp.gif'
            animate_iterates(x1_trajectory=x1_traj_interp, 
                             sparse_meas=padded_meas,
                             interp_meas=interp_meas,
                             gt=shepp, 
                             save_path=save_path,
                             save=True
                             )


        reconstruct_no_interp, metrics_no_interp, _, x1_traj_no_interp = dflow(
            max_iter=20,
            optim_steps=100,
            target_cost=0.05,
            init_x=init_x.clone(),
            patience=30,
            model=model,
            lr=0.2,
            y=radon_shepp.unsqueeze(0).unsqueeze(0),
            train_min=train_min,
            train_max=train_max,
            N=n_sub,
            optimizer="LBFGS",
            device=device
        )
        if i % animate_idx == 0:
            save_path = f'idx{i}/idx{i}-reconstruction_gif_{n_sub}-no-interp.gif'
            animate_iterates(x1_trajectory=x1_traj_no_interp, 
                             gt=shepp, 
                             save_path=save_path, 
                             save=True
                             )



        reconstruct_with_interp = reconstruct_with_interp.squeeze().detach()
        reconstruct_no_interp = reconstruct_no_interp.squeeze().detach()
        l2_error_interp = torch.norm(reconstruct_with_interp - shepp).cpu() / torch.norm(shepp).cpu()
        l1_error_interp = torch.norm(reconstruct_with_interp - shepp, p=1).cpu() / torch.norm(shepp, p=1).cpu()
        l2_error_no_interp = torch.norm(reconstruct_no_interp - shepp).cpu() / torch.norm(shepp).cpu()
        l1_error_no_interp = torch.norm(reconstruct_no_interp - shepp, p=1).cpu() / torch.norm(shepp, p=1).cpu()
        errors[n_sub]["l2_rel_errors_interp"].append(l2_error_interp)
        errors[n_sub]["l1_rel_errors_interp"].append(l1_error_interp)
        errors[n_sub]["l2_rel_errors_no_interp"].append(l2_error_no_interp)
        errors[n_sub]["l1_rel_errors_no_interp"].append(l1_error_no_interp)

        if i % animate_idx == 0:
            # create a plot of final reconstructions, with and without interpolation, their corresponding radon transforms, the ground truth, and the error maps
            fig, axes = plt.subplots(3, 5, figsize=(15, 10))
            shepp_clone = shepp.clone().cpu().numpy()
            reconstruct_with_interp = reconstruct_with_interp.cpu().numpy()
            reconstruct_no_interp = reconstruct_no_interp.cpu().numpy()
            radon_shepp = full_meas.clone().cpu().numpy()
            vmin = min(shepp_clone.min(), reconstruct_with_interp.min(), reconstruct_no_interp.min())
            vmax = max(shepp_clone.max(), reconstruct_with_interp.max(), reconstruct_no_interp.max())
            axes[0, 0].imshow(shepp_clone, cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axes[0, 0].set_title('Gt', fontsize=12)
            axes[0, 0].axis('off')
            axes[0, 1].imshow(reconstruct_with_interp, cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'Recons. (w/ Interp)', fontsize=12)
            axes[0, 1].text(0.5, -0.1, f'L2 Relative Error: {l2_error_interp:.4f}', 
                ha='center', va='top', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].axis('off')
            axes[0, 2].imshow(reconstruct_no_interp, cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axes[0, 2].set_title(f'Recons. (wo/ Interp)', fontsize=12)
            axes[0, 2].text(0.5, -0.1, f'L2 Relative Error: {l2_error_no_interp:.4f}', 
                ha='center', va='top', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].axis('off')
            axes[0, 3].imshow((reconstruct_with_interp - shepp_clone), cmap='RdBu_r', vmin=-np.abs((reconstruct_with_interp - shepp_clone).max()), vmax=np.abs((reconstruct_with_interp - shepp_clone).max()))
            axes[0, 3].set_title('Error Map (w/ Interp)', fontsize=12)
            axes[0, 3].text(0.5, -0.1, f'Max error: {np.abs((reconstruct_with_interp - shepp_clone)).max():.4f}',
                ha='center', va='top', transform=axes[0, 3].transAxes, fontsize=12)
            axes[0, 3].axis('off')
            axes[0, 4].imshow((reconstruct_no_interp - shepp_clone), cmap='RdBu_r', vmin=-np.abs((reconstruct_no_interp - shepp_clone).max()), vmax=np.abs((reconstruct_no_interp - shepp_clone).max()))
            axes[0, 4].set_title('Error Map (wo/ Interp)', fontsize=12)
            axes[0, 4].text(0.5, -0.1, f'Max error: {np.abs((reconstruct_no_interp - shepp_clone)).max():.4f}',
                ha='center', va='top', transform=axes[0, 4].transAxes, fontsize=12)
            axes[0, 4].axis('off')

            # create a plot of the radon transforms
            axes[1, 0].imshow(radon_shepp, cmap=cmocean.cm.dense)
            axes[1, 0].set_title('Radon (Gt)', fontsize=12)
            axes[1, 0].axis('off')
            radon_reconstruct_interp = radon.radon_transform(reconstruct_with_interp, N=n_full)
            axes[1, 1].imshow(radon_reconstruct_interp, cmap=cmocean.cm.dense)
            axes[1, 1].set_title(f'Radon Recon. (w/ Interp)', fontsize=12)
            axes[1, 1].text(0.5, -0.1, f'MSE: {metrics_interp["loss"][-1].compute().item():.4f}', 
                ha='center', va='top', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].axis('off')
            radon_reconstruct_no_interp = radon.radon_transform(reconstruct_no_interp, N=n_full)
            axes[1, 2].imshow(radon_reconstruct_no_interp, cmap=cmocean.cm.dense)
            axes[1, 2].set_title(f'Radon Recon. (wo/ Interp)', fontsize=12)
            axes[1, 2].text(0.5, -0.1, f'MSE: {metrics_no_interp["loss"][-1].compute().item():.4f}', 
                ha='center', va='top', transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].axis('off')
            # plot error
            axes[1, 3].imshow((radon_reconstruct_interp - radon_shepp), cmap='RdBu_r', vmin=-np.abs((radon_reconstruct_interp - radon_shepp).max()), vmax=np.abs((radon_reconstruct_interp - radon_shepp).max()))
            axes[1, 3].set_title('Error Map (w/ Interp)', fontsize=12)
            axes[1, 3].text(0.5, -0.1, f'Max error: {np.abs((radon_reconstruct_interp - radon_shepp)).max():.4f}', fontsize=12,
                            ha='center', va='top', transform=axes[1, 3].transAxes)
            axes[1, 3].axis('off')

            axes[1, 4].imshow((radon_reconstruct_no_interp - radon_shepp), cmap='RdBu_r', vmin=-np.abs((radon_reconstruct_no_interp - radon_shepp).max()), vmax=np.abs((radon_reconstruct_no_interp - radon_shepp).max()))
            axes[1, 4].set_title('Error Map (wo/ Interp)', fontsize=12)
            axes[1, 4].text(0.5, -0.1, f'Max error: {np.abs((radon_reconstruct_no_interp - radon_shepp)).max():.4f}', fontsize=12,
                            ha='center', va='top', transform=axes[1, 4].transAxes)
            axes[1, 4].axis('off')

            axes[2, 0].imshow(padded_meas.squeeze().cpu().numpy(), cmap=cmocean.cm.dense)
            axes[2, 0].set_title(f'Padded Meas.', fontsize=12)
            axes[2, 0].axis('off')

            axes[2, 1].imshow(interp_meas.squeeze().cpu().numpy(), cmap=cmocean.cm.dense)
            axes[2, 1].set_title(f'Interp. Meas.', fontsize=12)
            axes[2, 1].text(0.5, -0.1, f'MSE: {np.mean((interp_meas.squeeze().cpu().numpy() - full_meas.cpu().numpy())**2):.4f}', fontsize=12,
                            ha='center', va='top', transform=axes[2, 1].transAxes)
            axes[2, 1].axis('off')

            # error between interp and gt
            axes[2, 2].imshow((interp_meas.squeeze().cpu().numpy() - full_meas.cpu().numpy()), cmap='RdBu_r', vmin=-np.abs((interp_meas.squeeze().cpu().numpy() - full_meas.cpu().numpy()).max()), vmax=np.abs((interp_meas.squeeze().cpu().numpy() - full_meas.cpu().numpy()).max()))
            axes[2, 2].set_title('Error Map (Interp. Meas. Vs. Gt)', fontsize=12)
            axes[2, 2].text(0.5, -0.1, f'Max error: {np.abs((interp_meas.squeeze().cpu().numpy() - full_meas.cpu().numpy())).max():.4f}', fontsize=12,
                            ha='center', va='top', transform=axes[2, 2].transAxes)
            axes[2, 2].axis('off')

            axes[2, 3].axis('off')
            axes[2, 4].axis('off')

            plt.tight_layout()
            plt.savefig(f'idx{i}/idx{i}-final-reconstructions-{n_sub}-to-{n_full}.png', dpi=200)


# save 
for n_sub in subsample_rates:
    mean_l2_interp = np.mean(errors[n_sub]["l2_rel_errors_interp"])
    std_l2_interp = np.std(errors[n_sub]["l2_rel_errors_interp"])
    mean_l1_interp = np.mean(errors[n_sub]["l1_rel_errors_interp"])
    std_l1_interp = np.std(errors[n_sub]["l1_rel_errors_interp"])
    mean_l2_no_interp = np.mean(errors[n_sub]["l2_rel_errors_no_interp"])
    std_l2_no_interp = np.std(errors[n_sub]["l2_rel_errors_no_interp"])
    mean_l1_no_interp = np.mean(errors[n_sub]["l1_rel_errors_no_interp"])
    std_l1_no_interp = np.std(errors[n_sub]["l1_rel_errors_no_interp"])
    df = pl.DataFrame({
        "l2_rel_errors_interp": errors[n_sub]["l2_rel_errors_interp"],
        "l1_rel_errors_interp": errors[n_sub]["l1_rel_errors_interp"],
        "l2_rel_errors_no_interp": errors[n_sub]["l2_rel_errors_no_interp"],
        "l1_rel_errors_no_interp": errors[n_sub]["l1_rel_errors_no_interp"],
        "mean_l2_interp": mean_l2_interp,
        "std_l2_interp": std_l2_interp,
        "mean_l1_interp": mean_l1_interp,
        "std_l1_interp": std_l1_interp,
        "mean_l2_no_interp": mean_l2_no_interp,
        "std_l2_no_interp": std_l2_no_interp,
        "mean_l1_no_interp": mean_l1_no_interp,
        "std_l1_no_interp": std_l1_no_interp,
    })
    df.write_csv(f"errors_{n_sub}_to_{n_full}.csv")