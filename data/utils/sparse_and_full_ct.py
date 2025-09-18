# Assuming access to a dataset of media, we will generate data pairs of sparse CT measurements and "full" CT measurements
import numpy as np
import argparse

import torch
import os
import sys

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from transforms import radon

parser = argparse.ArgumentParser(description="Generate data pairs of sparse CT measurements and full CT measurements.")
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
parser.add_argument('--problem', type=str, default='shepp-logan', help='Dataset to use')
parser.add_argument('--im_size', type=int, default=128, help='This should correspond to a dataset we actually have')
parser.add_argument('--n_sub', type=int, default=1, help='number of sparse angles')
parser.add_argument('--n_full', type=int, default=24, help='number of full angles')
parser.add_argument('--multiflow', type=bool, default=False, help='whether to use multiflow (default: False)')

args = parser.parse_args()

dataset_source = torch.load(f"data/{args.problem}-dataset-{args.im_size}.pt")

# let's set a full measurement to be 24 angles
n_sub = args.n_sub
n_full = args.n_full
# for now we are assuming that n_sub | n_full, but this is not necessarily the case
# this is important because if n_sub does not divide n_full,
# it is not necessary that the angle set of n_sub is a subset of the angle set of n_full

dataset = {}

for split in ['train', 'val', 'test']:
    images = dataset_source[split]  # shape (N, 1, H, W)
    N, C, H, W = images.shape
    full_meas = []
    sub_meas = []

    # [0, pi]

    batch_size = 1024
    for i in tqdm(range(0, N, batch_size), desc=f"Generating {split} data"):
        imgs = images[i:i+batch_size].squeeze(1)  # shape (B, H, W)
        full_radon = radon.radon_transform(imgs, N=n_full)  # shape (n_full, detector_width)
        full_meas.append(full_radon)
        
        sub_radon = radon.radon_transform(imgs, N=n_sub)  # shape (n_sub, detector_width)
        sub_meas.append(sub_radon)
    
    full_meas = torch.cat(full_meas, dim=0)  # shape (N, n_full, detector_width)
    sub_meas = torch.cat(sub_meas, dim=0)      # shape (N, n_sub, detector_width)
    
    dataset[split] = {
        'full_meas': full_meas,
        'sub_meas': sub_meas
    }
    if args.multiflow:
        dataset[split]['media'] = images 
    print(dataset[split]['full_meas'].shape, dataset[split]['sub_meas'].shape)
save_name = f"data/{args.problem}-multiflow-v1-{n_sub}-{n_full}-{args.im_size}-multiflow.pt" if args.multiflow else f"data/{args.problem}-sparse-and-full-{n_sub}-{n_full}-{args.im_size}.pt"
torch.save(dataset, save_name)
print(f"Saved dataset to {save_name}")