import os
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import MSELoss

from get_loaders import get_loaders
from unet_v2 import UnetV2
from flow import OptimalTransportFlow, sample_images
from utils import make_im_grid, make_checkpoint, load_checkpoint
import argparse

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
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--image-size', type=int, default=128, help='Size of the input images')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='circles', help='Dataset to use')

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
        'num_workers': 16,
        'image_size': args.image_size,
        'problem': args.problem,
    }

    device = args.device
    model = UnetV2(ch=32).to(device)
    model = torch.compile(model)

    flow = OptimalTransportFlow(config['sigma_min'])
    loss_fn = get_loss_fn(model, flow)
    
    optim = torch.optim.Adam(model.parameters(), lr=config['min_lr'])
    # after loading the data we change working directory
    train_loader, _ = get_loaders(config)
    os.makedirs(f"problems/{config['problem']}", exist_ok=True)
    os.chdir(f"problems/{config['problem']}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    scaler = torch.amp.GradScaler()

    ckpt = args.ckpt
    if ckpt is not None:
        step, curr_epoch, model, optim, scaler, _ = load_checkpoint(ckpt, model, optim, scaler, None)
        print(f'Loaded checkpoint [step {step} ({curr_epoch})]')
    else:
        step = 0
        curr_epoch = 0

    pbar = tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        epoch_loss = 0
        num_batches = 0

        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optim)
            scaler.update()

            for g in optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr

            true_loss = loss.item()
            if (step + 1) % config['log_freq'] == 0:
                pbar.set_postfix_str(f'Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f} | Lr: {lr:.3e}')
                
            epoch_loss += true_loss
            num_batches += 1
            step += 1
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
     
        model.eval()
        with torch.no_grad():
            shape = (1, 1, args.image_size, args.image_size)

            gen_x = sample_images(model, shape, num_steps=2, device=device)
            gen_x = gen_x[-1]
            
            assert gen_x.shape == shape

            image = make_im_grid(gen_x, (1,1))
            image.save(f'samples/{epoch}.png')

        if epoch % 10 == 0 or epoch == config['epochs']:
            make_checkpoint(f'checkpoints/ckp_{step}.tar', step, epoch, model, optim, scaler, ema_model=None)
    