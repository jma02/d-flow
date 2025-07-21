import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import cmocean as cmo
def is_valid_circle(center, radius, circles, min_distance=20):
    for existing_circle in circles:
        existing_center, existing_radius = existing_circle
        distance = ((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)**0.5
        if distance < radius + existing_radius + min_distance:
            return False
    return True


def create_circles_dataset(num_samples=5000, im_size=128):
    # Initialize a tensor to store all images
    dataset = torch.zeros((num_samples, im_size, im_size))

    scattering_indices = np.linspace(0,1,20)
    for sample_idx in range(num_samples):
        img = torch.zeros((im_size, im_size), dtype=torch.float32)
        num_circles = random.randint(1, 5)  # Random number of circles per image
        circles = []

        for _ in range(num_circles):
            while True:
                center_x, center_y = random.randint(0, im_size-1), random.randint(0, im_size-1)
                radius = random.randint(im_size//10, im_size//8)
                if is_valid_circle((center_x, center_y), radius, circles, min_distance=3):
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
im_size = 64
print(f"Creating dataset: {num_samples} {im_size} x {im_size} images")
dataset = create_circles_dataset(num_samples, im_size=im_size)
dataset = dataset.unsqueeze(1) # Add channel dimension
plt.imshow(dataset[0].squeeze(), cmap=cmo.cm.ice)
plt.savefig('circle-dataset-sample.png', format='png')

train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size
dataset = {
    'train': dataset[:train_size],
    'val': dataset[train_size:train_size + val_size],
    'test': dataset[train_size + val_size:]
}

torch.save(dataset, "data/dataset.pt")
print("Dataset saved as dataset.pt")