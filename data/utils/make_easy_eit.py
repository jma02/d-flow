import torch
import random
import numpy as np
import tqdm
from math import floor
import argparse

def is_valid_circle(center, radius, circles, min_distance=16):
    for existing_circle in circles:
        existing_center, existing_radius = existing_circle
        distance = ((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)**0.5
        if distance < radius + existing_radius + min_distance:
            return False
    return True


def create_circles_dataset(num_samples=5000, im_size=128):
    dataset = torch.zeros((num_samples, im_size, im_size))

    scattering_indices = np.linspace(4, 5, 5)
    for sample_idx in tqdm.tqdm(range(num_samples), desc=f"Creating easy eit circles dataset"):
        img = torch.ones((im_size, im_size), dtype=torch.float32)
  
        num_circles = 1
        circles = []

        for _ in range(num_circles):
            while True:
                center_x, center_y = random.randint(floor(im_size*.2), floor(im_size*.8)), random.randint(floor(im_size*.2), floor(im_size*.8))
                radius = random.randint(floor(im_size*.1), floor(im_size*.15))
                if is_valid_circle((center_x, center_y), radius, circles, min_distance=floor(im_size*.065)):
                    break

            circles.append(((center_x, center_y), radius))

            # Create a grid of coordinates
            x = torch.arange(im_size).view(-1, 1)
            y = torch.arange(im_size).view(1, -1)

            # Calculate the distance from the center
            distance = (x - center_x)**2 + (y - center_y)**2
            # Create a mask for the circle
            circle_mask = distance <= radius**2

            img[circle_mask] = np.random.choice(scattering_indices, replace=True)

        # Store the image in the dataset
        dataset[sample_idx] = img

    return dataset

num_samples=20000
im_size = 128
dataset = create_circles_dataset(num_samples, im_size=im_size)
dataset = dataset.unsqueeze(1) # Add channel dimension

train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
dataset = {
    'train': dataset[:train_size],
    'val': dataset[train_size:train_size + val_size],
    'test': dataset[train_size + val_size:]
}

torch.save(dataset, f"data/eit-circles-easy-dataset-{im_size}.pt")
print(f"Dataset saved as eit-circles-easy-dataset-{im_size}.pt")