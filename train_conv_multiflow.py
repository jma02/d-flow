import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
from torch.nn import MSELoss

from unet_v2 import UnetV2 
from flow import OptimalTransportFlow
from utils import make_checkpoint
import matplotlib.pyplot as plt
import cmocean
import argparse
from torchdiffeq import odeint

# torch.manual_seed(159753)
# np.random.seed(159753)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def get_loss_fn(model: UnetV2, flow: OptimalTransportFlow):
    def loss_fn(x0: Tensor, t: Tensor, target: Tensor) -> Tensor:
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
    # parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--image-size', type=int, default=128, help='Size of the input images')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--problem', type=str, default='eit-shepp-logan', help='Dataset to use')

    args = parser.parse_args()
    config = {
        'sigma_min': 1e-2,
        'min_lr': 1e-7,
        'max_lr': 5e-3,
        'warmup_steps': 22500,
        'epochs': args.num_epochs,
        'max_steps': 400000,
        'batch_size': args.batch_size,
        'log_freq': 1, 
        'num_workers': 32,
        'image_size': args.image_size,
        'problem': args.problem,
    }

    no_conv_model = UnetV2(ch=32).to("cuda:0")
    conv_1_model = UnetV2(ch=32).to("cuda:0")
    conv_2_model = UnetV2(ch=32).to("cuda:1")
    conv_3_model = UnetV2(ch=32).to("cuda:1")

    no_conv_model = torch.compile(no_conv_model)
    conv_1_model = torch.compile(conv_1_model)
    conv_2_model = torch.compile(conv_2_model)
    conv_3_model = torch.compile(conv_3_model)

    flow_cuda0 = OptimalTransportFlow(config['sigma_min'])
    flow_cuda1 = OptimalTransportFlow(config['sigma_min'])
    no_conv_loss_fn = get_loss_fn(no_conv_model, flow_cuda0)
    conv_1_loss_fn = get_loss_fn(conv_1_model, flow_cuda0)
    conv_2_loss_fn = get_loss_fn(conv_2_model, flow_cuda1)
    conv_3_loss_fn = get_loss_fn(conv_3_model, flow_cuda1)

    no_conv_optim = torch.optim.Adam(no_conv_model.parameters(), lr=config['min_lr'])
    conv_1_optim = torch.optim.Adam(conv_1_model.parameters(), lr=config['min_lr'])
    conv_2_optim = torch.optim.Adam(conv_2_model.parameters(), lr=config['min_lr'])
    conv_3_optim = torch.optim.Adam(conv_3_model.parameters(), lr=config['min_lr'])

    # after loading the data we change working directory
    data = torch.load(f'data/{config["problem"]}-conv-dataset-128.pt')
    train_data = data['train']
    conv_cfg = {
            1: {'size': 11, 'sigma': 4},     
            2: {'size': 21, 'sigma': 12},    
            3: {'size': 45, 'sigma': 25},    
    }
    train_no_conv = train_data['no_conv']
    train_no_conv = 2.0 * (train_no_conv - train_no_conv.min()) / (train_no_conv.max() - train_no_conv.min()) - 1.0
    train_conv_1 = train_data[f"kernel_size{conv_cfg[1]['size']}_sigma{conv_cfg[1]['sigma']}"]
    train_conv_1 = 2.0 * (train_conv_1 - train_conv_1.min()) / (train_conv_1.max() - train_conv_1.min()) - 1.0
    train_conv_2 = train_data[f"kernel_size{conv_cfg[2]['size']}_sigma{conv_cfg[2]['sigma']}"]
    train_conv_2 = 2.0 * (train_conv_2 - train_conv_2.min()) / (train_conv_2.max() - train_conv_2.min()) - 1.0
    train_conv_3 = train_data[f"kernel_size{conv_cfg[3]['size']}_sigma{conv_cfg[3]['sigma']}"]
    train_conv_3 = 2.0 * (train_conv_3 - train_conv_3.min()) / (train_conv_3.max() - train_conv_3.min()) - 1.0

    train = TensorDataset(
        train_no_conv.detach().clone(),
        train_conv_1.detach().clone(),
        train_conv_2.detach().clone(),
        train_conv_3.detach().clone(),
    )
    train_loader = DataLoader(
        train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    run_name = "conv-multiflow-v1"
    os.makedirs(f"problems/{run_name}/{config['problem']}/", exist_ok=True)
    os.chdir(f"problems/{run_name}/{config['problem']}/")
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    cuda_0_scaler = torch.amp.GradScaler()
    cuda_1_scaler = torch.amp.GradScaler()
    

    step = 0
    curr_epoch = 0

    pbar = tqdm(range(curr_epoch, config['epochs'] + 1), desc="Epochs")
    for epoch in pbar:
        no_conv_model.train()
        conv_1_model.train()
        conv_2_model.train()
        conv_3_model.train()

        for i, (no_conv, conv_1, conv_2, conv_3) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False, total=len(train_loader)):
            no_conv = no_conv.to("cuda:0")
            conv_1 = conv_1.to("cuda:0")
            conv_2 = conv_2.to("cuda:1")
            conv_3 = conv_3.to("cuda:1")
            t = torch.rand(no_conv.shape[0], device="cpu")
            x0 = torch.randn_like(no_conv, device="cpu")

            no_conv_optim.zero_grad(set_to_none=True)
            conv_1_optim.zero_grad(set_to_none=True)
            conv_2_optim.zero_grad(set_to_none=True)
            conv_3_optim.zero_grad(set_to_none=True)

            t_cuda_0 = t.clone().to("cuda:0")
            x0_cuda_0 = x0.clone().to("cuda:0")
            with torch.autocast(device_type="cuda"):
                no_conv_loss = no_conv_loss_fn(x0_cuda_0, t_cuda_0, no_conv) 
                conv_1_loss = conv_1_loss_fn(x0_cuda_0, t_cuda_0, conv_1) 
            

            t_cuda_1 = t.clone().to("cuda:1")
            x0_cuda_1 = x0.clone().to("cuda:1") 
    
            with torch.autocast(device_type="cuda"):
                conv_2_loss = conv_2_loss_fn(x0_cuda_1, t_cuda_1, conv_2) 
                conv_3_loss = conv_3_loss_fn(x0_cuda_1, t_cuda_1, conv_3) 
            
            cuda_0_scaler.scale(no_conv_loss).backward()
            cuda_0_scaler.scale(conv_1_loss).backward()
            cuda_1_scaler.scale(conv_2_loss).backward()
            cuda_1_scaler.scale(conv_3_loss).backward()


            cuda_0_scaler.unscale_(no_conv_optim)
            cuda_0_scaler.unscale_(conv_1_optim)
            cuda_1_scaler.unscale_(conv_2_optim)
            cuda_1_scaler.unscale_(conv_3_optim)

            no_conv_grad = torch.nn.utils.clip_grad_norm_(no_conv_model.parameters(), max_norm=1.0)
            conv_1_grad = torch.nn.utils.clip_grad_norm_(conv_1_model.parameters(), max_norm=1.0)
            conv_2_grad = torch.nn.utils.clip_grad_norm_(conv_2_model.parameters(), max_norm=1.0)
            conv_3_grad = torch.nn.utils.clip_grad_norm_(conv_3_model.parameters(), max_norm=1.0)

            cuda_0_scaler.step(no_conv_optim)
            cuda_0_scaler.step(conv_1_optim)
            cuda_1_scaler.step(conv_2_optim)
            cuda_1_scaler.step(conv_3_optim)

            cuda_0_scaler.update()
            cuda_1_scaler.update()

            for g in no_conv_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr
            for g in conv_1_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr
            for g in conv_2_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr
            for g in conv_3_optim.param_groups:
                lr = get_lr(config, step)
                g['lr'] = lr

            true_no_conv_loss = no_conv_loss.item() 
            true_conv_1_loss = conv_1_loss.item()
            true_conv_2_loss = conv_2_loss.item()
            true_conv_3_loss = conv_3_loss.item()
            if (step + 1) % config['log_freq'] == 0:
                pbar.set_postfix({
                    'Step': step,
                    'NoConvLoss': f'{true_no_conv_loss:.3f}',
                    'Conv1Loss': f'{true_conv_1_loss:.3f}',
                    'Conv2Loss': f'{true_conv_2_loss:.3f}',
                    'Conv3Loss': f'{true_conv_3_loss:.3f}',
                    # 'LR': f'{lr:.1e}',
                    # 'NoConvGrad': f'{no_conv_grad.item():.3f}',
                    # 'Conv1Grad': f'{conv_1_grad.item():.3f}',
                    # 'Conv2Grad': f'{conv_2_grad.item():.3f}',
                    # 'Conv3Grad': f'{conv_3_grad.item():.3f}',
                    # 'x0_cuda0': f'{x0_cuda_0.mean().item():.3f},{x0_cuda_0.std().item():.3f}',
                    # 'x0_cuda1': f'{x0_cuda_1.mean().item():.3f},{x0_cuda_1.std().item():.3f}',
                    # 't_cuda0': f'{t_cuda_0.mean().item():.3f},{t_cuda_0.std().item():.3f}',
                    # 't_cuda1': f'{t_cuda_1.mean().item():.3f},{t_cuda_1.std().item():.3f}',
                    'x0_cuda0_isnan': torch.isnan(x0_cuda_0).any().item(),
                    't_cuda0_isnan': torch.isnan(t_cuda_0).any().item(),
                    'x0_cuda1_isnan': torch.isnan(x0_cuda_1).any().item(),
                    't_cuda1_isnan': torch.isnan(t_cuda_1).any().item(),
                })

            step += 1
        no_conv_model.eval()
        conv_1_model.eval()
        conv_2_model.eval()
        conv_3_model.eval()
        with torch.no_grad():
            x0 = torch.randn(1, 1, config['image_size'], config['image_size'], device="cpu")
            t_cuda_0 = torch.linspace(0.0, 1.0, 5, device="cuda:0")
            x0_cuda_0 = x0.clone().to("cuda:0")
            x0_cuda_1 = x0.clone().to("cuda:1")
            t_cuda_1 = torch.linspace(0.0, 1.0, 5, device="cuda:1")
            no_conv_sample = odeint(
                func = lambda t, x: no_conv_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t_cuda_0,
                y0 = x0_cuda_0,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            conv_1_sample = odeint(
                func = lambda t, x: conv_1_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t_cuda_0,
                y0 = x0_cuda_0,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            conv_2_sample = odeint(
                func = lambda t, x: conv_2_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t_cuda_1,
                y0 = x0_cuda_1,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            # x0 = x0.clone().to("cuda:1")
            # t = torch.linspace(0.0, 1.0, 5, device="cuda:1")

            conv_3_sample = odeint(
                func = lambda t, x: conv_3_model(x, t.expand(x.shape[0]).to(x.device)),
                t = t_cuda_1,
                y0 = x0_cuda_1,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5,
            )[-1].squeeze().cpu()

            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            plt.suptitle(f'Epoch {epoch}', fontsize=16)
            # place all plots on the same color scale
            vmin = min(no_conv_sample.min(), conv_1_sample.min(), conv_2_sample.min(), conv_3_sample.min())
            vmax = max(no_conv_sample.max(), conv_1_sample.max(), conv_2_sample.max(), conv_3_sample.max())
            axs[0].imshow(no_conv_sample.squeeze(), cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axs[0].set_title('No Conv')
            axs[0].axis('off')

            axs[1].imshow(conv_1_sample.squeeze(), cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axs[1].set_title(f'Conv Size {conv_cfg[1]["size"]} Sigma {conv_cfg[1]["sigma"]}')
            axs[1].axis('off')  

            axs[2].imshow(conv_2_sample.squeeze(), cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axs[2].set_title(f'Conv Size {conv_cfg[2]["size"]} Sigma {conv_cfg[2]["sigma"]}')
            axs[2].axis('off')

            axs[3].imshow(conv_3_sample.squeeze(), cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)
            axs[3].set_title(f'Conv Size {conv_cfg[3]["size"]} Sigma {conv_cfg[3]["sigma"]}')
            axs[3].axis('off')

            # log the plot locally
            plt.savefig(f'samples/sample_epoch_{epoch}.png')
            plt.close()

    
    
        if epoch % 10 == 0 or epoch == config['epochs']:
            os.makedirs('checkpoints/no_conv', exist_ok=True)
            os.makedirs('checkpoints/conv_1', exist_ok=True)
            os.makedirs('checkpoints/conv_2', exist_ok=True)
            os.makedirs('checkpoints/conv_3', exist_ok=True)
            make_checkpoint(
                f'checkpoints/no_conv/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=no_conv_model, 
                optim=no_conv_optim, 
                scaler=cuda_0_scaler, 
            )
            make_checkpoint(
                f'checkpoints/conv_1/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=conv_1_model, 
                optim=conv_1_optim, 
                scaler=cuda_0_scaler, 
            )
            make_checkpoint(
                f'checkpoints/conv_2/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=conv_2_model, 
                optim=conv_2_optim, 
                scaler=cuda_1_scaler, 
            )
            make_checkpoint(
                f'checkpoints/conv_3/ckp_{step}.tar', 
                step=step, 
                epoch=epoch, 
                model=conv_3_model, 
                optim=conv_3_optim, 
                scaler=cuda_1_scaler, 
            )
