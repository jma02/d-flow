import torch
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from transforms import radon
from dflow_func import dflow, animate_iterates
from unet import Unet
from utils import load_checkpoint
import os
import argparse

parser = argparse.ArgumentParser(description="Run some sparsity tests.")
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
shepps = torch.load(f"data/{problem}-dataset-128.pt")["test"]
shepp = shepps[10].squeeze()
print(shepp.shape)

checkpoint_path = f'problems/{problem}/checkpoints/{im_size}x{im_size}/ckp_63000.tar'
# getting normalization constants
train_min, train_max = torch.min(torch.load(f'data/{problem}-dataset-{im_size}.pt')["train"]), torch.max(torch.load(f'data/{problem}-dataset-{im_size}.pt')["train"])
# match the number of channels to your model
model = Unet(ch=32).to(device)

_, _, model, _, _, _= load_checkpoint(model=model, path=checkpoint_path)
model.eval()

run = "run1"
os.makedirs(f'dflow-radon-metrics/{run}-{problem}/noise{noise}', exist_ok=True)
os.chdir(f'dflow-radon-metrics/{run}-{problem}/noise{noise}')
plt.imsave("shepp.png", shepp, cmap=cmocean.cm.dense)


init_x = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)
for N in [1,2,3,4,5,10,25]:
    radon_shepp = radon.radon_transform(shepp, N=N)
    radon_shepp = (1 + noise * torch.randn_like(radon_shepp)) * radon_shepp
    print(radon_shepp.shape)
    plt.imsave(f'shepp-N{N}-radon.png', radon_shepp, cmap=cmocean.cm.dense)
    radon_shepp = radon_shepp.to(device)
    reconstructed, metrics, svd_traj, x1_trajectory = dflow(
        max_iter=20,
        optim_steps=100,
        target_cost=0.05,
        init_x=init_x.clone(),
        patience=30,
        model=model,
        lr=0.1,
        y=radon_shepp.unsqueeze(0).unsqueeze(0),
        train_min=train_min,
        train_max=train_max,
        N=N,
        optimizer="LBFGS",
        device=device
    )
    save_path = f'reconstruction_gif_N{N}.gif'
    animate_iterates(x1_trajectory=x1_trajectory, gt=shepp, save_path=save_path, save=True)
    torch.save(metrics, f'N{N}-metrics.pt')