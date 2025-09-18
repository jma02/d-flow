from typing import List

import torch
import torch.nn as nn

from unet_v2.modules import (
    get_timestep_embedding,
    Downsample,
    Upsample,
    ResBlock,
)


class UnetV2(nn.Module):
    def __init__(
            self,
            ch: int = 128, 
            ch_mul: List[int] = [1, 2, 2, 2], 
            groups: int = 32,
            dropout: float = 0.1,
        ):
        super().__init__()

        self.ch = ch
        self.ch_mul = ch_mul
        self.dropout = dropout
        self.groups = groups

        self.temb_dim = self.ch * 4
        
        self.input_proj = nn.Conv2d(1, self.ch, 3, 1, 1)

        self.time_proj = nn.Sequential(nn.Linear(self.ch, self.temb_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.temb_dim, self.temb_dim))
        
        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])

        self.make_paths()
        
        self.final = nn.Sequential(nn.GroupNorm(num_groups=groups, num_channels=2*self.ch),
                                   nn.SiLU(),
                                   nn.Conv2d(2*self.ch, 1, 3, 1, 1))
    


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        assert t.shape == (x.shape[0],), 't should be a (batch_size,)-shaped array'
    
        temb = get_timestep_embedding(t, self.ch)
        emb = self.time_proj(temb)

        initial_proj = self.input_proj(x)
        h = initial_proj
        
        down_path = []

        for block_group in self.down:
            h = block_group[0](h, emb)
            h = block_group[1](h, emb)
            down_path.append(h)

            if len(block_group) > 2:
                h = block_group[2](h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h, emb)

        for block_group in self.up:
            h = torch.cat((h, down_path.pop()), dim=1)
            h = block_group[0](h, emb)
            h = block_group[1](h, emb)

            if len(block_group) > 2:
                h = block_group[2](h)

        x = torch.cat((h, initial_proj), dim=1)
        return self.final(x)
     
    

    def make_transition(self, res, down):
        dim = self.ch * self.ch_mul[res]

        if down:
            is_last_res = (res == (len(self.ch_mul) - 1))
            if is_last_res:
                return Downsample(dim, dim)
            
            dim_out = self.ch * self.ch_mul[res + 1]
            return Downsample(dim, dim_out)
        
        is_first_res = (res == 0)
        if is_first_res:
            return Upsample(dim, dim)
        
        dim_out = self.ch * self.ch_mul[res - 1]
        return Upsample(dim, dim_out)
            
        
    def make_res(self, res, down):
        dim = self.ch * self.ch_mul[res]
        transition = self.make_transition(res, down)
        
        if down:
            block1 = ResBlock(dim, dim, self.temb_dim, self.groups, self.dropout)
            block2 = ResBlock(dim, dim, self.temb_dim, self.groups, self.dropout)
        else:
            block1 = ResBlock(2 * dim, dim, self.temb_dim, self.groups, self.dropout)
            block2 = ResBlock(dim, dim, self.temb_dim, self.groups, self.dropout)
        
        return nn.ModuleList([block1, block2, transition])
 


    def make_paths(self):
        num_res = len(self.ch_mul)

        for res in range(num_res):
            is_last_res = (res == (num_res - 1))
            
            down_blocks = self.make_res(res, down=True)
            up_blocks = self.make_res(res, down=False)

            if is_last_res:
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]

            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)

        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList([
            ResBlock(nch, nch, self.temb_dim, self.groups, self.dropout),
            ResBlock(nch, nch, self.temb_dim, self.groups, self.dropout),
        ])
