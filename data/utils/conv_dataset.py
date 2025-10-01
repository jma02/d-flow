import torch
import argparse
from tqdm import tqdm

# point it at a dataset and let it rip
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default="eit-shepp-logan", help='Dataset')
args = parser.parse_args()
problem = args.problem
data = torch.load(f'data/{problem}-dataset-128.pt')

# edit sigma, kernel size here
params = [
    {'size': 11, 'sigma': 4},     
    {'size': 21, 'sigma': 12},    
    {'size': 45, 'sigma': 25},    
]

def gaussian_kernel(size=5, sigma=2):
    """Create 2D Gaussian kernel"""
    coords = torch.arange(size) - (size-1)/2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    k = torch.exp(-(x**2 + y**2)/(2*sigma**2))
    return k / k.sum()


new_dataset = {

}
for split in ["train", "val", "test"]:
    dataset = data[split]
    conv_dataset = {
        "no_conv": dataset
    }
    for param in params:
        size, sigma = param['size'], param['sigma'] 
        assert size % 2 == 1, "Kernel size must be odd"
        conv_data = []
        for i, img in tqdm(enumerate(dataset), desc=f"Convolving {problem} {split} data with kernel size {size}, sigma {sigma}", total=len(dataset)):
            kernel = gaussian_kernel(size=size, sigma=sigma).unsqueeze(0).unsqueeze(0) 
            conv_img = torch.nn.functional.conv2d(img.unsqueeze(0), kernel, padding=size//2)
            conv_data.append(conv_img.squeeze(0))
        conv_dataset[f'kernel_size{size}_sigma{sigma}'] = torch.stack(conv_data)
    new_dataset[split] = conv_dataset

torch.save(new_dataset, f'data/{problem}-conv-dataset-128.pt')