# A Python implementation of the MATLAB code provided at https://github.com/matthiaschung/Random-Shepp-Logan-Phantom

import numpy as np
import torch
import argparse
from tqdm import tqdm

def randomSheppLogan(n=512, default = False, phantom_type='msl', pad=4, M=1):
    phantom = shepp_logan(phantom_type)
    if phantom_type == 'eit-high-contrast':
        images = np.ones(((n + 2 * pad)**2, M))
    else:
        images = np.zeros(((n + 2 * pad)**2, M))

    pix = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(pix, -pix)
    
    if pad > 0:
        if phantom_type == 'eit-high-contrast':
            z1 = np.ones((n + 2 * pad, pad))
            z2 = np.ones((pad, n))
        else:
            z1 = np.zeros((n + 2 * pad, pad))
            z2 = np.zeros((pad, n))
         
    for i in range(M):
        if default:
            curr_phantom = phantom
        else:
            curr_phantom = modify(phantom, phantom_type=phantom_type)
        
        image = generateImage(curr_phantom, n, X, Y, phantom_type=phantom_type)
        if pad > 0:
            image = np.block([[z1, np.vstack([z2, image, z2]), z1]])
        
        images[:, i] = image.flatten()
    
    return images
    
def generateImage(e, n, X, Y, phantom_type='msl'):
    # initialize image
    if phantom_type == 'eit-high-contrast':
        # for EIT we want the background to be 1
        image = np.ones((n, n))            
    else: 
        image = np.zeros((n, n))           
    e[:, 1] = e[:, 1] ** 2              # update elements in phantom a^2
    e[:, 2] = e[:, 2] ** 2              # update elements in phantom b^2      
    e[:, 5] = e[:, 5] * np.pi / 180     # convert angles to radians
    cosp = np.cos(e[:, 5])              # take cosine of angles
    sinp = np.sin(e[:, 5])              # take sine of angles

    for k in range(e.shape[0]):
        # correct by center of ellipse
        x = X - e[k, 3]                 # x offset
        y = Y - e[k, 4]                 # y offset
        ellipse = ((x * cosp[k] + y * sinp[k]) ** 2) / e[k, 1] + ((y * cosp[k] - x * sinp[k]) ** 2) / e[k, 2]
        idx = np.where(ellipse <= 1)    # pixel indices within ellipse
        if k < 5:                       # set density within ellipse
            image[idx] = e[k, 0]
        else:                           # add density in ellipse to pixel values
            image[idx] = image[idx] + e[k, 0]
    
    return image


def modify(phantom, phantom_type='msl'):
    # number of ellipses
    m = phantom.shape[0]

    # Generate random scaling
    scale = 1 - np.random.rand() * 2 / 9
    phantom[:, 1:5] = scale * phantom[:, 1:5]

    # Random rotation
    rotation = 2 * 45 * (np.random.rand() - 0.5)
    phantom[:, 5] = rotation + phantom[:, 5]

    # Random translation
    translate = 0.2 * np.random.rand(1, 2)
    phantom[:, 3:5] = translate + phantom[:, 3:5]

    # random density relative to density
    density = 2 * 0.1 * (np.random.rand(m, 1) - 0.5)
    phantom[:, 0] = density.flatten() * phantom[:, 0] + phantom[:, 0]
    if phantom_type != 'eit-high-contrast':
        # clip if not doing EIT
        phantom[:, 0] = np.clip(phantom[:, 0], 0, 1)

    # Remove random tumors 
    obj = 4
    idx = np.random.choice(m-obj, size=np.random.randint(0, m - obj), replace=False)
    phantom = np.delete(phantom, idx+obj, axis=0)

    return phantom

def shepp_logan(phantom_type='msl'):
    if phantom_type == 'sl':
        """
        column 1. A additive intensity of the ellipse
        column 2. The length of the horizontal semi-axis of the ellipse
        column 3. The length of the vertical semi-axis of the ellipse
        column 4. The x-coordinate of the center of the ellipse
        column 5. The y-coordinate of the center of the ellipse
        column 6. The angle of rotation between the horizontal semi-axis 
        #         of the ellipse and the x-axis of the image
        """
        # Standard Shepp-Logan phantom parameters
        phantom = np.array([
            [1, 0.69, 0.92, 0, 0, 0],               # outer skull cap boundaries
            [0.02, 0.6624, 0.8740, 0, -0.0184, 0],  # inner skull cap boundaries
            [0, 0.11, 0.31, 0.22, 0, -18],          # ventricle left 
            [0, 0.16, 0.41, -0.22, 0, 18],          # ventricle right
            [0.01, 0.21, 0.25, 0, 0.35, 0],         # tumor 1
            [0.01, 0.046, 0.046, 0, 0.1, 0],        # tumor 2
            [0.01, 0.046, 0.046, 0, -0.1, 0],       # tumor 3
            [0.01, 0.046, 0.023, -0.08, -0.605, 0], # tumor 4
            [0.01, 0.023, 0.023, 0, -0.606, 0],     # tumor 5 
            [0.01, 0.023, 0.046, 0.06, -0.605, 0]   # tumor 6
        ])
    elif  phantom_type== 'msl':
        # modified, same as shepp-logan except intensities are changed to yield
        # higher contrast in the image
        phantom = np.array([
            [1, 0.69, 0.92, 0, 0, 0],
            [0.2, 0.6624, 0.8740, 0, -0.0184, 0],
            [0, 0.11, 0.31, 0.22, 0, -18],
            [0, 0.16, 0.41, -0.22, 0, 18],
            [0.1, 0.21, 0.25, 0, 0.35, 0],
            [0.1, 0.046, 0.046, 0, 0.1, 0],
            [0.1, 0.046, 0.046, 0, -0.1, 0],
            [0.1, 0.046, 0.023, -0.08, -0.605, 0],
            [0.1, 0.023, 0.023, 0, -0.606, 0],
            [0.1, 0.023, 0.046, 0.06, -0.605, 0]
        ])
    elif  phantom_type== 'eit-high-contrast':
        # modified, here we use very high contrast to represent conductivities in the EIT problem 
        phantom = np.array([
            [5, 0.69, 0.92, 0, 0, 0],
            [4, 0.6624, 0.8740, 0, -0.0184, 0],
            [3, 0.11, 0.31, 0.22, 0, -18],
            [3, 0.16, 0.41, -0.22, 0, 18],
            [2, 0.21, 0.25, 0, 0.35, 0],
            [2, 0.046, 0.046, 0, 0.1, 0],
            [2, 0.046, 0.046, 0, -0.1, 0],
            [2, 0.046, 0.023, -0.08, -0.605, 0],
            [2, 0.023, 0.023, 0, -0.606, 0],
            [2, 0.023, 0.046, 0.06, -0.605, 0]
        ])
    else:
        raise ValueError("No valid phantom type selected.")

    return phantom


def main():
    parser = argparse.ArgumentParser(description="Generate Shepp-Logan phantoms.")
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--problem', type=str, default='eit-hc', help='msl, sl, or eit-hc')
    # msl is for ct, this here is just for logging
    pm = {
        'msl' : 'ct',
        'eit-hc' : 'eit'
    }

    args = parser.parse_args()

    im_size = args.im_size
    n = im_size - 10 # Size of the image
    pad = 5  # Padding size
    M = 1 # number of randomizations
    N = 20000 # number of shepp logans

    problem = args.problem

    phantom_list = [torch.tensor(randomSheppLogan(n=n, pad=pad, M=M, phantom_type=problem)[:, M-1], dtype=torch.float32).view(1, n + 2*pad, n + 2*pad) 
                    for _ in tqdm(range(N), desc=f"Generating {pm[problem]} phantoms")]
    
    # Stack all phantoms into a single tensor (N, 1, H, W)
    images = torch.stack(phantom_list, dim=0)

    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    dataset = {
        'train': images[:train_size],
        'val': images[train_size:train_size + val_size],
        'test': images[train_size + val_size:]
    }

    torch.save(dataset, f"data/{pm[problem]}-shepp-logan-dataset-{n+2*pad}.pt")
    print(f"Dataset saved as {pm[problem]}-shepp-logan-dataset-{n+2*pad}.pt")



if __name__ == "__main__":
    main()