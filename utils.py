import numpy as np
import wandb
import torch
from torchvision import datasets
import torchvision.transforms.v2 as v2
import cmocean

from einops import rearrange


# unloader = v2.Compose([v2.Lambda(lambda t: (t + 1) * 0.5),
#                        v2.Lambda(lambda t: t.permute(0, 2, 3, 1)),
#                        v2.Lambda(lambda t: t * 255.)])


unloader = v2.Compose([v2.Lambda(lambda t: (t + 1) * 0.5),
                        v2.Lambda(lambda t: t.permute(0, 2, 3, 1))])

def make_im_grid(x0: torch.Tensor, xy: tuple=(1, 10)):
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    
    im = im.numpy() 
    
    # Apply cmocean ice colormap to single channel images
    if C == 1:
        # Remove channel dimension for single channel
        im = im.squeeze(-1)  # Shape: (B, H, W)
        # Apply colormap to each image
        cmap = cmocean.cm.ice
        im_colored = np.zeros((B, H, W, 3))
        for i in range(B):
            im_colored[i] = cmap(im[i])[:, :, :3]  # Remove alpha channel
        im = (im_colored * 255).astype(np.uint8)
    else:
        im = (im * 255).astype(np.uint8)
    
    # Create grid
    im = rearrange(im, '(x y) h w c -> (x h) (y w) c', x=B//x, y=B//y)
    im = v2.ToPILImage()(im)
    return im


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    checkpoint = {
        'epoch': int(epoch),
        'step': int(step),
        'model_state_dict': model.state_dict(),
    }

    if optim is not None:
        checkpoint['optim_state_dict'] = optim.state_dict()

    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)


def make_checkpoint_multiflow_v1(path, step, epoch, model, linear_map, optim=None, scaler=None):
    checkpoint = {
        'epoch': int(epoch),
        'step': int(step),
        'model_state_dict': model.state_dict(),
        'linear_map': linear_map,
    }

    if optim is not None:
        checkpoint['optim_state_dict'] = optim.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)




def load_checkpoint_multiflow_v1(path, model, optim=None, scaler=None):
    # Load checkpoint to CPU first to avoid device mismatches
    checkpoint = torch.load(path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    # Create a new state dict to handle the 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
        new_state_dict[name] = v
        
    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict, strict=False)
    
    step = int(checkpoint['step'])
    epoch = int(checkpoint['epoch'])

    # It's better practice for the calling script to handle model.eval()
    # model.eval() 

    if optim and 'optim_state_dict' in checkpoint:
        # Create a new state dict here too 
        new_optim_state_dict = OrderedDict()
        for k, v in optim.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
            new_optim_state_dict[name] = v
        optim.load_state_dict(new_optim_state_dict)

    if scaler and 'scaler_state_dict' in checkpoint:
        new_scaler_state_dict = OrderedDict()
        for k, v in scaler.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
            new_scaler_state_dict[name] = v
        scaler.load_state_dict(new_scaler_state_dict)

    return step, epoch, model, optim, scaler, checkpoint['linear_map']

def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    # Load checkpoint to CPU first to avoid device mismatches
    checkpoint = torch.load(path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    # Create a new state dict to handle the 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
        new_state_dict[name] = v
        
    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict, strict=False)
    
    step = int(checkpoint['step'])
    epoch = int(checkpoint['epoch'])

    # It's better practice for the calling script to handle model.eval()
    # model.eval() 

    if optim and 'optim_state_dict' in checkpoint:
        # Create a new state dict here too 
        new_optim_state_dict = OrderedDict()
        for k, v in optim.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
            new_optim_state_dict[name] = v
        optim.load_state_dict(new_optim_state_dict)


    # I don't think we'll use this
    if ema_model and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        # Create a new state dict here too
        new_scaler_state_dict = OrderedDict()
        for k, v in scaler.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            name = name[10:] if name.startswith('_orig_mod.') else name  # remove `orig_mod.`
            new_scaler_state_dict[name] = v
        scaler.load_state_dict(new_scaler_state_dict)

    return step, epoch, model, optim, scaler, ema_model