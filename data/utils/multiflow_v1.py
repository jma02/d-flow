# let's create data triplets for sparse ct, full ct, and media
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
sparse_and_full = torch.load(f"data/{args.problem}-sparse-and-full-{args.n_sub}-{args.n_full}-{args.im_size}.pt")
dataset = {}

for split in ['train', 'val', 'test']:
    dataset[split] = {
        'full_meas': sparse_and_full[split]['full_meas'],
        'sub_meas': sparse_and_full[split]['sub_meas'],
        'media': dataset_source[split]
    }

print(f"Full measurements shape: {dataset['train']['full_meas'].shape}, "
      f"Sub measurements shape: {dataset['train']['sub_meas'].shape}, "
      f"Media shape: {dataset['train']['media'].shape}")

torch.save(dataset, f"data/{args.problem}-multiflow-v1-{args.n_sub}-{args.n_full}-{args.im_size}.pt")