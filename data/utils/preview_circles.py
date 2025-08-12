import cmocean.cm as cmo
import torch
import matplotlib.pyplot as plt

data = torch.load('data/dataset-16.pt')
dataset = data['train']

# Ensure dataset is 4D: (B, C, H, W)
if dataset.dim() == 3:
    dataset = dataset.unsqueeze(1)

num_images = min(100, dataset.shape[0])
rows = int(num_images**0.5)
cols = (num_images + rows - 1) // rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
axes = axes.flatten()

for i in range(num_images):
    img = dataset[i].squeeze()
    axes[i].imshow(img.cpu().numpy(), cmap=cmo.dense)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
# Hide unused axes
for j in range(num_images, len(axes)):
    axes[j].axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('circle_mosaic.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
