import math

import numpy as np

import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


def make_skip_connection(dim_in, dim_out):
    if dim_in == dim_out:
        return nn.Identity()
    return nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)

def make_block(dim_in, dim_out, num_groups, dropout=0):
    return nn.Sequential(nn.GroupNorm(num_groups=num_groups, num_channels=dim_in), 
                         nn.SiLU(),
                         nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
                         nn.Conv2d(dim_in, dim_out, 3, 1, 1))


class ConditioningBlock(nn.Module):
    def __init__(self, dim_out, emb_dim):
        super().__init__()
        dim = 2 * dim_out 
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim)
        )
    
    def forward(self, emb):
        emb = self.proj(emb)[:, :, None, None]
        return emb
    

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim, num_groups=32, dropout=0.1, attn=False):
        super().__init__()

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)
        self.cond_block = ConditioningBlock(dim_out, emb_dim)

    def forward(self, x, emb):
        emb = self.cond_block(emb)

        h = self.block1(x)
        # scale shifting
        out_norm, out_rest = self.block2[0], self.block2[1:]
        scale, shift = emb.chunk(2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return h


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, downscale_freq_shift: 'float' = 0, max_period: int = 10000):
    assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'
    assert embedding_dim % 2 == 0, 'Even embedding dimensions only!'
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb