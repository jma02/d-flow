import os
import numpy as np
from tqdm import tqdm

import torch
import wandb
from torch import Tensor
from torch.nn import MSELoss

from get_loaders import get_loaders_multiflow_v1
from unet import Unet
from flow import OptimalTransportFlow
from utils import make_checkpoint_multiflow_v1
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

def get_loss_fn(model: Unet, flow: OptimalTransportFlow):
    def loss_fn(source: Tensor, target: Tensor, t: Tensor) -> Tensor:
        # t = torch.rand(source.shape[0], device=source.device)
        x0 = source
        x1 = target

        xt = flow.step(t, x0, x1)
        print(f"xt shape: {xt.shape}")  
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
    """
    if you fix the random seed, you will get the same sequence of gaussian noise
    over and over again, so this is one way to train our multiflow.

    so as long as our training takes exactly the same number of iterations,
    or we know that we draw from gaussian exactly the same number of times,
    this is a correct way to train these multiflows.

    but i think that jointly training all models at the same time is probably
    better, since i can't guarantee that the above holds.

    the challenge with this is obviously storing the gradients and weights 
    concurrently. so let's try just loading it all on one gpu.

    if it doesn't fit, the correct way to solve this would be to split the weights
    onto our two gpus. this takes abit of work, and also, usually the second gpu
    is busy.

    so the option i will try if one gpu doesn't work is to just buy gpu time on modal.
    """

    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
    # again assuming only one gpu
    # implement checkpoints if needed later.
    # parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--ckpt-path', type=str, default="checkpoints", help='Path to save checkpoints')
    parser.add_argument('--samples-path', type=str, default="samples", help='Path to save generated samples')

    parser.add_argument('--image-size', type=int, default=128, help='Size of the input images')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='shepp-logan', help='Dataset to use')
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
    }

    n_sub = args.n_sub
    n_full = args.n_full
    wandb.init(project="dflow", config=config)
    device = args.device

    # miniature models for smaller image size
    sub_meas_model = Unet(ch=32, ch_mul=[1, 2], att_channels=[0, 1]).to(device)
    full_meas_model = Unet(ch=32, ch_mul=[1, 2], att_channels=[0, 1]).to(device)

    media_model = Unet(ch=32).to(device)

    media_model = torch.compile(media_model)
    sub_meas_model = torch.compile(sub_meas_model)
    full_meas_model = torch.compile(full_meas_model)

    flow = OptimalTransportFlow(config['sigma_min'])
    sub_meas_loss_fn = get_loss_fn(sub_meas_model, flow)
    full_meas_loss_fn = get_loss_fn(full_meas_model, flow)
    media_loss_fn = get_loss_fn(media_model, flow)

    sub_meas_optim = torch.optim.Adam(sub_meas_model.parameters(), lr=config['min_lr'])
    full_meas_optim = torch.optim.Adam(full_meas_model.parameters(), lr=config['min_lr'])
    media_optim = torch.optim.Adam(media_model.parameters(), lr=config['min_lr'])

    # after loading the data we change working directory
    train_loader, test_loader = get_loaders_multiflow_v1(config)
    os.makedirs(f"problems/multiflow-v1/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}", exist_ok=True)
    os.chdir(f"problems/multiflow-v1/{config['problem']}/{config['n_sub']}-{config['n_full']}-{config['image_size']}x{config['image_size']}")
    os.makedirs(args.samples_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    sub_meas_scaler = torch.amp.GradScaler()
    full_meas_scaler = torch.amp.GradScaler()
    media_scaler = torch.amp.GradScaler()

    # ckpt = args.ckpt
    # if ckpt is not None:
    #     step, curr_epoch, model, optim, scaler, ema_model = load_checkpoint(ckpt, model, optim, scaler, ema_model)
    #     print(f'Loaded checkpoint [step {step} ({curr_epoch})]')
    # else:
    #     step = 0
    #     curr_epoch = 0
    step = 0
    curr_epoch = 0
    accumulation_steps = 2

    # flattened dimensions
    # yes this is hard coded
    sub_flattened_dim = config['n_sub'] * 182
    full_flattened_dim = config['n_full'] * 182
    media_flattened_dim = config['image_size'] * config['image_size']

    # in principle having ambient dim > media_flattened_dim should work but let's just try this
    ambient_dim = media_flattened_dim

    # projecting matrices
    A_sub = torch.randn(sub_flattened_dim, ambient_dim, device=device)
    A_full = torch.randn(full_flattened_dim, ambient_dim, device=device)
    # let's try not using this
    A_media = torch.randn(media_flattened_dim, ambient_dim, device=device)

    for epoch in tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs"):
        sub_meas_model.train()
        full_meas_model.train()
        media_model.train()

        epoch_sub_loss = 0
        epoch_full_loss = 0
        epoch_media_loss = 0

        num_batches = 0

        for i, (sub, full, media) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            sub = sub.to(device)
            full = full.to(device)
            media = media.to(device)

            if i % accumulation_steps == 0:
                sub_meas_optim.zero_grad(set_to_none=True)
                full_meas_optim.zero_grad(set_to_none=True)
                media_optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device):
                t = torch.rand(sub.shape[0], device=device)
                x0_ambient = torch.randn(sub.shape[0], ambient_dim, device=device)
                
                # batchwise matmul 
                x0_sub = torch.matmul(x0_ambient, A_sub.T)
                x0_full = torch.matmul(x0_ambient, A_full.T)
                # x0_media = torch.matmul(x0_ambient, A_media.T)

                # reshape to image dimensions
                x0_sub = x0_sub.view(sub.shape[0], 1, config['n_sub'], 182)
                x0_full = x0_full.view(full.shape[0], 1, config['n_full'], 182)
                x0_media = x0_ambient.view(media.shape[0], 1, config['image_size'], config['image_size'])

                sub_loss = sub_meas_loss_fn(x0_sub, sub, t) / accumulation_steps
                full_loss = full_meas_loss_fn(x0_full, full, t) / accumulation_steps
                media_loss = media_loss_fn(x0_media, media, t) / accumulation_steps


            full_meas_scaler.scale(full_loss).backward()
            sub_meas_scaler.scale(sub_loss).backward()
            media_scaler.scale(media_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                sub_meas_scaler.unscale_(sub_meas_optim)
                full_meas_scaler.unscale_(full_meas_optim)
                media_scaler.unscale_(media_optim)

                sub_grad = torch.nn.utils.clip_grad_norm_(sub_meas_model.parameters(), max_norm=1.0)
                full_grad = torch.nn.utils.clip_grad_norm_(full_meas_model.parameters(), max_norm=1.0)
                media_grad = torch.nn.utils.clip_grad_norm_(media_model.parameters(), max_norm=1.0)

                sub_meas_scaler.step(sub_meas_optim)
                full_meas_scaler.step(full_meas_optim)
                media_scaler.step(media_optim)

                sub_meas_scaler.update()
                full_meas_scaler.update()
                media_scaler.update()

                for g in sub_meas_optim.param_groups:
                    lr = get_lr(config, step)
                    g['lr'] = lr
                
                for g in full_meas_optim.param_groups:
                    lr = get_lr(config, step)
                    g['lr'] = lr

                for g in media_optim.param_groups:
                    lr = get_lr(config, step)
                    g['lr'] = lr

                true_sub_loss = sub_loss.item() * accumulation_steps
                true_full_loss = full_loss.item() * accumulation_steps
                true_media_loss = media_loss.item() * accumulation_steps
                if (step + 1) % config['log_freq'] == 0:
                    print(f'Step {step}: '
                          f'Sub Loss: {true_sub_loss:.6f}, '
                          f'Full Loss: {true_full_loss:.6f}, '
                          f'Media Loss: {true_media_loss:.6f}, '
                          f'LR: {lr:.2e}, '
                          f'Sub Grad: {sub_grad:.3f}, '
                          f'Full Grad: {full_grad:.3f}, '
                          f'Media Grad: {media_grad:.3f}')
                    
                    # Log training metrics to wandb
                    wandb.log({
                        "true_sub_loss": true_sub_loss,
                        "sub_grad_norm": sub_grad.item(),
                        "true_full_loss": true_full_loss,
                        "full_grad_norm": full_grad.item(),
                        "true_media_loss": true_media_loss,
                        "media_grad_norm": media_grad.item(),
                        "learning_rate": lr,
                        "step": step,
                        "epoch": epoch
                    })

                epoch_sub_loss += true_sub_loss
                epoch_full_loss += true_full_loss
                epoch_media_loss += true_media_loss
                num_batches += 1
                step += 1
        
        # Log epoch metrics
        avg_epoch_sub_loss = epoch_sub_loss / num_batches
        avg_epoch_full_loss = epoch_full_loss / num_batches
        avg_epoch_media_loss = epoch_media_loss / num_batches
        wandb.log({
            "epoch_sub_loss": avg_epoch_sub_loss,
            "epoch_full_loss": avg_epoch_full_loss,
            "epoch_media_loss": avg_epoch_media_loss,
            "epoch": epoch
        })

        sub_meas_model.eval()
        full_meas_model.eval()
        media_model.eval()
        with torch.no_grad():
            print(f'Generating samples at epoch {epoch}')
            x0_amb = torch.randn(1, ambient_dim, device=device)
            x0_sub = torch.matmul(x0_amb, A_sub.T).view(1, 1, config['n_sub'], 182)
            x0_full = torch.matmul(x0_amb, A_full.T).view(1, 1, config['n_full'], 182)
            # x0_media = torch.matmul(x0_amb, A_media.T).view(1, 1, config['image_size'], config['image_size'])
            x0_media = x0_amb.view(1, 1, config['image_size'], config['image_size'])

            t = torch.linspace(0.0, 1.0, 5, device=device)

            sub_samples = odeint(
                func = lambda t, x: sub_meas_model(x, t.expand(x.shape[0])),
                t = t,
                y0 = x0_sub,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            full_samples = odeint(
                func = lambda t, x: full_meas_model(x, t.expand(x.shape[0])),
                t = t,
                y0 = x0_full,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            media_samples = odeint(
                func = lambda t, x: media_model(x, t.expand(x.shape[0])),
                t = t,
                y0 = x0_media,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            fig, axs = plt.subplots(5, 2, figsize=(12, 20))
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

            # take the media sample and apply radon transform with n_sub and n_full
            from transforms import radon
            sub_radon = radon.radon_transform(media_samples, N=config['n_sub']).cpu()
            full_radon = radon.radon_transform(media_samples, N=config['n_full']).cpu()
            im = axs[3, 0].imshow(sub_radon, cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[3, 0], shrink=0.3)
            axs[3, 0].set_title('Operator Applied To Media Sub')
            axs[3, 0].axis('off')

            im = axs[3, 1].imshow(full_radon, cmap=cmocean.cm.dense)
            fig.colorbar(im, ax=axs[3, 1], shrink=0.3)
            axs[3, 1].set_title('Operator Applied To Media Full')
            axs[3, 1].axis('off')

            # calcuate rel error
            sub_error = torch.norm(sub_radon - sub_samples) / torch.norm(sub_samples)
            full_error = torch.norm(full_radon - full_samples) / torch.norm(full_samples)

            im = axs[4, 0].imshow((sub_radon - sub_samples).abs(), cmap='hot')
            fig.colorbar(im, ax=axs[4, 0], shrink=0.3)
            axs[4, 0].set_title(f'Full Gen Radon vs. Op, L2 Rel Error {sub_error:.4f}', fontsize=8)
            axs[4, 0].axis('off')

            im = axs[4, 1].imshow((full_radon - full_samples).abs(), cmap='hot')
            fig.colorbar(im, ax=axs[4, 1], shrink=0.3)
            axs[4, 1].set_title(f'Full Gen Radon vs. Op, L2 Rel Error {full_error:.4f}', fontsize=8)
            axs[4, 1].axis('off')


            # log the plot locally
            plt.savefig(f'{config["sample_path"]}/sample_epoch_{epoch}.png')
            plt.close()

            # log the plot to wandb
            wandb.log({
                "samples": wandb.Image(plt.gcf()),
                "epoch": epoch,
            })
    
        if epoch % 10 == 0 or epoch == config['epochs']:
            make_checkpoint_multiflow_v1(
                f'{config["ckpt_path"]}/sub_{n_sub}/ckp_{step}.tar', 
                step, 
                epoch, 
                sub_meas_model, 
                sub_meas_optim, 
                sub_meas_scaler, 
                A_sub 
                )
            make_checkpoint_multiflow_v1(
                f'{config["ckpt_path"]}/full_{n_full}/ckp_{step}.tar', 
                step, 
                epoch, 
                full_meas_model, 
                full_meas_optim, 
                full_meas_scaler, 
                A_full
                )
            make_checkpoint_multiflow_v1(
                f'{config["ckpt_path"]}/media/ckp_{step}.tar', 
                step, 
                epoch, 
                media_model, 
                media_optim, 
                media_scaler, 
                A_media
            )
            print(f"Checkpoints saved at step {step}, epoch {epoch}")
    
    wandb.finish()