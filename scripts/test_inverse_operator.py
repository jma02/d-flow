import torch
from unet_v2 import UnetV2NoTime
from torch.utils.data import DataLoader, TensorDataset
from utils import load_checkpoint
import argparse

parser = argparse.ArgumentParser(description="Test a trained unet inverse operator.")
parser.add_argument('--problem', type=str, default='eit-shepp-logan', help='Dataset to use')
args = parser.parse_args()
problem = args.problem

dataset = torch.load(f'data/{problem}-multiflow-128.pt') if 'eit' in problem else torch.load(f'data/{problem}-multiflow-1-24-128.pt')
test_dataset = dataset['test']
inputs = test_dataset['dtn_map'] if 'eit' in problem else test_dataset['sub_meas']
inputs = TensorDataset(inputs.unsqueeze(1).float())
test_loader = DataLoader(inputs, batch_size=32, shuffle=False, pin_memory=True, drop_last=False)
device = 'cuda'
idx = 37750
checkpoint_path = f'problems/inverse-operator/{problem}/checkpoints/ckp_{idx}.tar'
model = UnetV2NoTime(ch=32).to(device)
_, _, model, _, _, _= load_checkpoint(model=model, path=checkpoint_path)
model.eval()
preds = []
with torch.no_grad():
    for (x,) in test_loader:
        x = x.to(device)
        output = model(x)
        preds.append(output.cpu())

preds = torch.cat(preds, dim=0).squeeze(1)
gt = test_dataset['media']

l2_rel = torch.norm(preds - gt, dim=(1,2)) / torch.norm(gt, dim=(1,2))
l1_rel = torch.norm(preds - gt, p=1, dim=(1,2)) / torch.norm(gt, p=1, dim=(1,2))

print(f'L2 relative error: {l2_rel.mean():.4f} ± {l2_rel.std():.4f}')
print(f'L1 relative error: {l1_rel.mean():.4f} ± {l1_rel.std():.4f}')