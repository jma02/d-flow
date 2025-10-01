import os
import numpy as np
from tqdm import tqdm
import logging

import torch
from torch import Tensor
from torch.nn import MSELoss

from get_loaders import get_loaders_multiflow_v2
from unet_v2 import UnetV2 
from flow import OptimalTransportFlow
from utils import make_checkpoint
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

def get_loss_fn(model: UnetV2, flow: OptimalTransportFlow):
    def loss_fn(target: Tensor) -> Tensor:
        t = torch.rand(target.shape[0], device=target.device)
        x0 = torch.randn_like(target, device=target.device)
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
    # again assuming only one gpu
    # implement checkpoints if needed later.
    # parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--ckpt-path', type=str, default="checkpoints", help='Path to save checkpoints')
    parser.add_argument('--samples-path', type=str, default="samples", help='Path to save generated samples')

    parser.add_argument('--image-size', type=int, default=128, help='Size of the input images')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='ct-shepp-logan', help='Dataset to use')
    parser.add_argument('--n-sub', type=int, default=2, help='number of sparse angles')
    parser.add_argument('--n-full', type=int, default=24, help='number of full angles')
    # this approach doesn't require n_full to be divisible by n_sub! 

    args = parser.parse_args()
    config = {
        'sigma_min': 1e-2,
        'min_lr': 1e-7,
        'max_lr': 5e-3,
        'warmup_steps': 22500,
        'epochs': args.num_epochs,
        'max_steps': 400000,
        'batch_size': args.batch_size,
        'log_freq': 50, # more frequent logging
        'num_workers': 32,
        'image_size': args.image_size,
        'problem': args.problem,
        'n_sub': args.n_sub,
        'n_full': args.n_full,
        'sample_path': args.samples_path,
        'ckpt_path': args.ckpt_path,
        'modal': False
    }

    n_sub = args.n_sub
    n_full = args.n_full
    # device = args.device

    # miniature models for smaller image size
    sub_meas_model = UnetV2(ch=64, ch_mul=[1, 2]).to("cuda:1") # 2 x 182
    full_meas_model = UnetV2(ch=64, ch_mul=[1, 2]).to("cuda:1") # 24 x 182

    media_model = UnetV2(ch=32).to("cuda:0") # 128 x 128

    
    # torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, inductor=logging.DEBUG)

    media_model = torch.compile(media_model)
    sub_meas_model = torch.compile(sub_meas_model)
    full_meas_model = torch.compile(full_meas_model)

    flow = OptimalTransportFlow(config['sigma_min'])
    media_loss_fn = get_loss_fn(media_model, flow)
    sub_meas_loss_fn = get_loss_fn(sub_meas_model, flow)
    full_meas_loss_fn = get_loss_fn(full_meas_model, flow)

    media_optim = torch.optim.Adam(media_model.parameters(), lr=config['min_lr'])
    sub_meas_optim = torch.optim.Adam(sub_meas_model.parameters(), lr=config['min_lr'])
    full_meas_optim = torch.optim.Adam(full_meas_model.parameters(), lr=config['min_lr'])

    # after loading the data we change working directory
    train_loader, test_loader = get_loaders_multiflow_v2(config)
    run_suffix = "operator-learning"
    os.makedirs(f"problems/multiflow-v2-{run_suffix}/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}", exist_ok=True)
    os.chdir(f"problems/multiflow-v2-{run_suffix}/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}")
    os.makedirs(args.samples_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    media_scaler = torch.amp.GradScaler()
    sub_meas_scaler = torch.amp.GradScaler()
    full_meas_scaler = torch.amp.GradScaler()

    step = 0
    curr_epoch = 0

    pbar = tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs")
    for epoch in pbar:
        media_model.train()
        sub_meas_model.train()
        full_meas_model.train()

        for i, (sub, full, media) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False, total=len(train_loader)):
            sub = sub.to("cuda:1")
            full = full.to("cuda:1")
            media = media.to("cuda:0")

            media_optim.zero_grad(set_to_none=True)
            sub_meas_optim.zero_grad(set_to_none=True)
            full_meas_optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda:1"):
                media_loss = media_loss_fn(media) 
                sub_loss = sub_meas_loss_fn(sub)

            with torch.amp.autocast(device_type="cuda:0"):
                full_loss = full_meas_loss_fn(full)

            media_scaler.scale(media_loss).backward()
            full_meas_scaler.scale(full_loss).backward()
            sub_meas_scaler.scale(sub_loss).backward()

            media_scaler.unscale_(media_optim)
            sub_meas_scaler.unscale_(sub_meas_optim)
            full_meas_scaler.unscale_(full_meas_optim)

            media_grad = torch.nn.utils.clip_grad_norm_(media_model.parameters(), max_norm=1.0)
            sub_grad = torch.nn.utils.clip_grad_norm_(sub_meas_model.parameters(), max_norm=1.0)
            full_grad = torch.nn.utils.clip_grad_norm_(full_meas_model.parameters(), max_norm=1.0)

            media_scaler.step(media_optim)
            sub_meas_scaler.step(sub_meas_optim)
            full_meas_scaler.step(full_meas_optim)

            media_scaler.update()
            sub_meas_scaler.update()
            full_meas_scaler.update()

            for g in media_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr
            for g in sub_meas_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr
            
            for g in full_meas_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr

            true_media_loss = media_loss.item()
            true_sub_loss = sub_loss.item()
            true_full_loss = full_loss.item()
            if (step + 1) % config['log_freq'] == 0:
                pbar.set_postfix({
                    'Step': step,
                    'Sub': f'{true_sub_loss:.3f}',
                    'Full': f'{true_full_loss:.3f}', 
                    'Media': f'{true_media_loss:.3f}',
                    'LR': f'{lr:.1e}',
                    'SubGrad': f'{sub_grad:.2f}',
                    'FullGrad': f'{full_grad:.2f}',
                    'MediaGrad': f'{media_grad:.2f}'
                })

            step += 1
    
        media_model.eval()
        sub_meas_model.eval()
        full_meas_model.eval()
        with torch.no_grad():
            x0_media = torch.randn(1, 1, config['image_size'], config['image_size'], device="cuda:0")
            x0_sub = torch.randn(1, 1 , config['n_sub'], 182, device="cuda:1")
            x0_full = torch.randn(1, 1 , config['n_full'], 182, device="cuda:1")


            t = torch.linspace(0.0, 1.0, 5, device="cpu")

            sub_samples = odeint(
                func = lambda t, x: sub_meas_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t,
                y0 = x0_sub,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            full_samples = odeint(
                func = lambda t, x: full_meas_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t,
                y0 = x0_full,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            media_samples = odeint(
                func = lambda t, x: media_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t,
                y0 = x0_media,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            fig, axs = plt.subplots(3, 2, figsize=(12, 20))
            plt.suptitle(f'Epoch {epoch}', fontsize=16)

            im = axs[0, 0].imshow(x0_sub.squeeze().cpu(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[0, 0], shrink=0.3)
            axs[0, 0].set_title('Sub init')
            axs[0, 0].axis('off')

            im = axs[0, 1].imshow(sub_samples, cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[0, 1], shrink=0.3)
            axs[0, 1].set_title('Sub final')
            axs[0, 1].axis('off')

            im = axs[1, 0].imshow(x0_full.squeeze().cpu(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[1, 0], shrink=0.3)
            axs[1, 0].set_title('Full init')
            axs[1, 0].axis('off')

            im = axs[1, 1].imshow(full_samples, cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[1, 1], shrink=0.3)
            axs[1, 1].set_title('Full final')
            axs[1, 1].axis('off')

            im = axs[2, 0].imshow(x0_media.squeeze().cpu(), cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[2, 0])
            axs[2, 0].set_title('Media init')
            axs[2, 0].axis('off')

            im = axs[2, 1].imshow(media_samples, cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[2, 1])
            axs[2, 1].set_title('Media final')
            axs[2, 1].axis('off')

            # log the plot locally
            plt.savefig(f'{config["sample_path"]}/sample_epoch_{epoch}.png')
            plt.close()

    
    
        if epoch % 10 == 0 or epoch == config['epochs']:
            os.makedirs(f'{config["ckpt_path"]}/sub_{n_sub}', exist_ok=True)
            os.makedirs(f'{config["ckpt_path"]}/full_{n_full}', exist_ok=True)
            os.makedirs(f'{config["ckpt_path"]}/media', exist_ok=True)
            make_checkpoint(
                f'{config["ckpt_path"]}/media/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=media_model, 
                optim=media_optim, 
                scaler=media_scaler, 
            )
            make_checkpoint(
                f'{config["ckpt_path"]}/sub_{n_sub}/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=sub_meas_model, 
                optim=sub_meas_optim, 
                scaler=sub_meas_scaler, 
            )
            make_checkpoint(
                f'{config["ckpt_path"]}/full_{n_full}/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=full_meas_model, 
                optim=full_meas_optim, 
                scaler=full_meas_scaler, 
            )