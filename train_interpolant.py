import os
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import MSELoss

from get_loaders import get_loaders_interpolant
from unet import Unet
from flow import OptimalTransportFlow
from utils import make_checkpoint, load_checkpoint
import matplotlib.pyplot as plt
import cmocean
import argparse
from torchdiffeq import odeint

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def get_indices(n_sub, n_full):
    assert n_full % n_sub == 0, "n_full must be a multiple of n_sub"
    step = n_full // n_sub
    indices = [i * step for i in range(n_sub)]
    return indices

def get_loss_fn(model: Unet, flow: OptimalTransportFlow):
    def loss_fn(source: Tensor, target: Tensor) -> Tensor:
        t = torch.rand(source.shape[0], device=source.device)
        x0 = torch.zeros(target.shape, device=target.device)
        indices = get_indices(config["n_sub"], config["n_full"])
        x0[:, :, indices, :] = source
        x1 = target

        xt = flow.step(t, x0, x1)
        pred_vel = model(xt, t)
        true_vel = flow.target(t, x0, x1)

        loss = MSELoss()(pred_vel, true_vel)
        return loss
    return loss_fn


def get_lr(config, step):
    if step < config['warmup_steps']:
        lr = config['min_lr'] + (config['max_lr'] - config['min_lr']) * (step / config['warmup_steps'])
        return lr

    if step > config['max_steps']:
        return config['min_lr']

    decay_ratio = (step - config['warmup_steps']) / (config['max_steps'] - config['warmup_steps'])
    lr = config['max_lr'] - (config['max_lr'] - config['min_lr']) * decay_ratio
    return lr


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--ckpt-path', type=str, default="checkpoints", help='Path to save checkpoints')
    parser.add_argument('--samples-path', type=str, default="samples", help='Path to save generated samples')

    parser.add_argument('--image-size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='circles', help='Dataset to use')
    parser.add_argument('--n-sub', type=int, default=1, help='number of sparse angles')
    parser.add_argument('--n-full', type=int, default=24, help='number of full angles')

    args = parser.parse_args()
    config = {
        'sigma_min': 1e-2,
        'min_lr': 1e-8,
        'max_lr': 5e-4,
        'warmup_steps': 45000,
        'epochs': args.num_epochs,
        'max_steps': 400000,
        'batch_size': args.batch_size,
        'log_freq': 1000, # sparsely log since we are training offline
        'num_workers': 32,
        'image_size': args.image_size,
        'problem': args.problem,
        'n_sub': args.n_sub,
        'n_full': args.n_full,
        'sample_path': args.samples_path,
        'ckpt_path': args.ckpt_path,
    }

    device = args.device

    model = Unet(ch=32, ch_mul=[1, 2], att_channels=[0, 1]).to(device)
    model = torch.compile(model)

    flow = OptimalTransportFlow(config['sigma_min'])
    loss_fn = get_loss_fn(model, flow)
    
    optim = torch.optim.Adam(model.parameters(), lr=config['min_lr'])
    # after loading the data we change working directory
    train_loader, test_loader = get_loaders_interpolant(config)
    os.makedirs(f"problems/interpolant/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}", exist_ok=True)
    os.chdir(f"problems/interpolant/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}")
    os.makedirs(args.samples_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    scaler = torch.amp.GradScaler()

    ckpt = args.ckpt
    if ckpt is not None:
        step, curr_epoch, model, optim, scaler, _ = load_checkpoint(ckpt, model, optim, scaler, None)
        print(f'Loaded checkpoint [step {step} ({curr_epoch})]')
    else:
        step = 0
        curr_epoch = 0

    accumulation_steps = 2

    for epoch in tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs"):
        model.train()
        
        epoch_loss = 0
        num_batches = 0

        for i, (x, y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            if i % accumulation_steps == 0:
                optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x, y) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optim)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()

                for g in optim.param_groups:
                    lr = get_lr(config, step)
                    g['lr'] = lr

                true_loss = loss.item() * accumulation_steps
                if (step + 1) % config['log_freq'] == 0:
                    print(f'Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f} | Lr: {lr:.3e}')
                    
                epoch_loss += true_loss
                num_batches += 1
                step += 1
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
        
        model.eval()
        with torch.no_grad():
            print(f'Generating samples at epoch {epoch}')
            x, y = next(iter(test_loader))
            x = x[0].unsqueeze(0).to(device)
            y = y[0].unsqueeze(0).to(device)
            
            x_pad = torch.zeros(y.shape, device=device)
            indices = get_indices(config["n_sub"], config["n_full"])
            x_pad[:, :, indices, :] = x
            x = x_pad



            num_steps = 5
            timesteps = torch.linspace(0.0, 1.0, num_steps, device=device)
            output = odeint(
                func = lambda t, x: model(x, t.expand(x.shape[0])),
                t = timesteps,
                y0 = x,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze()
            
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            plt.suptitle(f'Epoch {epoch}', fontsize=16)

            x = x.squeeze()
            output = output.squeeze()
            print(f"Output shape: {output.shape}")
            y = y.squeeze()

            im = axs[0, 0].imshow(x.cpu().numpy(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[0, 0], shrink=0.3)
            axs[0, 0].set_title('Input')

            im = axs[0, 1].imshow(y.cpu().numpy(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[0, 1], shrink=0.3)
            axs[0, 1].set_title('Ground Truth')

            im = axs[1, 0].imshow(output.cpu().numpy(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[1, 0], shrink=0.3)
            axs[1, 0].set_title('Final Output')

            im = axs[1, 1].imshow((output - y).abs().cpu().numpy(), cmap='hot')
            fig.colorbar(im, ax=axs[1, 1], shrink=0.3)
            axs[1, 1].set_title('Error (Final Output - Ground Truth)')


            plt.tight_layout()

            # log the plot locally
            plt.savefig(f'{config["sample_path"]}/sample_epoch_{epoch}.png')

        if epoch % 10 == 0 or epoch == config['epochs']:
            make_checkpoint(f'{config["ckpt_path"]}/ckp_{step}.tar', step, epoch, model, optim, scaler, ema_model=None)
            print(f"Checkpoint saved at step {step}, epoch {epoch}")