import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import cmocean
from torch import Tensor
from torch.nn import MSELoss

from unet_v2 import UnetV2NoTime
from utils import make_checkpoint, load_checkpoint
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
    parser = argparse.ArgumentParser(description="Train a unet to parametrize an inverse operator.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='eit-shepp-logan', help='Dataset to use')

    args = parser.parse_args()
    problem = args.problem
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
        'problem': args.problem,
    }

    device = args.device
    model = UnetV2NoTime(ch=32).to(device)
    model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=config['min_lr'])
    # after loading the data we change working directory
    dataset = torch.load(
        f"data/{problem}-multiflow-128.pt"
    ) if "eit" in problem else torch.load(
        f"data/{problem}-multiflow-3-24-128.pt"
    )

    train_data = dataset["train"]
    val_data = dataset["val"]

    train_x = train_data["dtn_map"].float() if "eit" in problem else train_data["sub_meas"].float()
    train_y = train_data["media"].float()

    val_x = val_data["dtn_map"].float() if "eit" in problem else val_data["sub_meas"].float()
    val_y = val_data["media"].float()

    train_x = train_x.unsqueeze(1)
    train_y = train_y.unsqueeze(1)
    val_x = val_x.unsqueeze(1)
    val_y = val_y.unsqueeze(1)

    train = TensorDataset(train_x.detach().clone(), train_y.detach().clone())
    test = TensorDataset(val_x.detach().clone(), val_y.detach().clone())

    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    os.makedirs(f"problems/inverse-operator/{config['problem']}-3-24", exist_ok=True)
    os.chdir(f"problems/inverse-operator/{config['problem']}-3-24")
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

    loss_fn = MSELoss()
    pbar = tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        epoch_loss = 0
        num_batches = 0

        for i, (x,y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device):
                output = model(x)
                loss = loss_fn(output ,y)

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
                pbar.set_postfix({'loss': f'{true_loss:.5f}', 'grad': f'{grad.item():.5f}', 'lr': f'{lr:.3e}'})
                
            epoch_loss += true_loss
            num_batches += 1
            step += 1
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
     
        model.eval()
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x = x[0].unsqueeze(0).to(device)
            y = y[0].unsqueeze(0).to(device)
            output = model(x)
            
            # create plot of the input, output, and born output
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            plt.suptitle(f'Epoch {epoch}', fontsize=16)

            x = x.squeeze(0).squeeze(0)
            output = output.squeeze()
            y = y.squeeze(0).squeeze(0)

          
            im = axs[0, 0].imshow(x.cpu().numpy(), cmap=cmocean.cm.balance)
            fig.colorbar(im, ax=axs[0, 0])
            axs[0, 0].set_title('DtN Input') if "eit" in problem else axs[0, 0].set_title('Subsampled Measurements Input')

            output_np = output.cpu().numpy()
            y_np = y.cpu().numpy()
            vmin = min(output_np.min(), y_np.min())
            vmax = max(output_np.max(), y_np.max())

            # UNet Output with shared scale
            im = axs[0, 1].imshow(output_np, cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axs[0, 1])
            axs[0, 1].set_title('UNet Output')

            # Ground Truth with same scale
            im = axs[1, 0].imshow(y_np, cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axs[1, 0])
            axs[1, 0].set_title('Ground Truth')

            # Error plot (separate scale is fine)
            im = axs[1, 1].imshow((output - y).abs().cpu().numpy(), cmap='hot')
            fig.colorbar(im, ax=axs[1, 1])
            axs[1, 1].set_title('Error (Final Output - Ground Truth)') 

            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png')
            plt.close(fig)

 

        if epoch % 10 == 0 or epoch == config['epochs']:
            make_checkpoint(f'checkpoints/ckp_{step}.tar', step, epoch, model, optim, scaler, ema_model=None)
    