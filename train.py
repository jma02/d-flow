import os
import numpy as np
from tqdm import tqdm

import torch
import wandb
from torch import Tensor
from torch.nn import MSELoss

from unet import Unet
from flow import OptimalTransportFlow, sample_images
from utils import *


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
    def loss_fn(batch: Tensor) -> Tensor:
        t = torch.rand(batch.shape[0], device=batch.device)
        x0 = torch.randn_like(batch)

        xt = flow.step(t, x0, batch)
        pred_vel = model(xt, t)
        true_vel = flow.target(t, x0, batch)

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
    os.makedirs('samples', exist_ok=True)
    # initialize wandb
    config = {
        'sigma_min': 1e-2,
        'min_lr': 1e-8,
        'max_lr': 5e-4,
        'warmup_steps': 45000,
        'epochs': 50,
        'max_steps': 400000,
        'batch_size': 64,
        'log_freq': 100,
        'num_workers': 2,
    }

    wandb.init(project="dflow", config=config)
    device = 'cuda'

    model = Unet().to(device)
    # Wait until our gpus get python development headers
    # model = torch.compile(model)

    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )
    flow = OptimalTransportFlow(config['sigma_min'])
    loss_fn = get_loss_fn(model, flow)
    
    optim = torch.optim.Adam(model.parameters(), lr=config['min_lr'])
    train_loader, _ = get_loaders(config)
    scaler = torch.amp.GradScaler()

    ckpt = None
    if ckpt is not None:
        step, curr_epoch, model, optim, scaler, ema_model = load_checkpoint(ckpt, model, optim, scaler, ema_model)
        print(f'Loaded checkpoint [step {step} ({curr_epoch})]')
    else:
        step = 0
        curr_epoch = 0

    accumulation_steps = 2

    for epoch in tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs"):
        model.train()
        ema_model.train()
        
        epoch_loss = 0
        num_batches = 0

        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            if i % accumulation_steps == 0:
                optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optim)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()

                ema_model.update_parameters(model)

                for g in optim.param_groups:
                    lr = get_lr(config, step)
                    g['lr'] = lr

                true_loss = loss.item() * accumulation_steps
                if (step + 1) % config['log_freq'] == 0:
                    print(f'Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f} | Lr: {lr:.3e}')
                    
                    # Log training metrics to wandb
                    wandb.log({
                        "loss": true_loss,
                        "grad_norm": grad.item(),
                        "learning_rate": lr,
                        "step": step,
                        "epoch": epoch
                    })

                epoch_loss += true_loss
                num_batches += 1
                step += 1
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch
        })
        
        model.eval()
        ema_model.eval()
        with torch.no_grad():
            print(f'Generating samples at epoch {epoch}')
            shape = (4, 1, 64, 64)

            gen_x = sample_images(model, shape, num_steps=2)
            gen_x_ema = sample_images(ema_model, shape, num_steps=2)
            gen_x = gen_x[-1]
            gen_x_ema = gen_x_ema[-1]
            
            assert gen_x.shape == shape

            image = make_im_grid(gen_x, (2,2))
            ema_image = make_im_grid(gen_x_ema, (2, 2))
            image.save(f'samples/{epoch}.png')
            ema_image.save(f'samples/ema_{epoch}.png')

            # Log sample images to wandb
            wandb.log({
                "samples": wandb.Image(image),
                "epoch": epoch,
                "ema_samples": wandb.Image(ema_image),
            })
    
    make_checkpoint(f'ckp_{step}.tar', step, epoch, model, optim, scaler, ema_model)
    
    wandb.finish()