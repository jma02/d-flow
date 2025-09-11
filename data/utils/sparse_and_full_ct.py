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

    for i in tqdm(range(N)):
        img = images[i].squeeze(0)  # shape (H, W)
        full_radon = radon.radon_transform(img, N=n_full)  # shape (n_full, detector_width)
        full_meas.append(full_radon)
        
        sub_radon = radon.radon_transform(img, N=n_sub)  # shape (n_sub, detector_width)
        sub_meas.append(sub_radon)
    
    full_meas = torch.stack(full_meas)  # shape (N, n_full, detector_width)
    sub_meas = torch.stack(sub_meas)      # shape (N, n_sub, detector_width)
    
    dataset[split] = {
        'full_meas': full_meas,
        'sub_meas': sub_meas
    }
torch.save(dataset, f"data/{args.problem}-sparse-and-full-{n_sub}-{n_full}-{args.im_size}.pt")