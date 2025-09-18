# Code to generate Shepp-Logan phantoms. 
# A Python implementation of the MATLAB code provided at https://github.com/matthiaschung/Random-Shepp-Logan-Phantom

import numpy as np
import matplotlib.pyplot as plt


def randomSheppLogan(n=512, default = False, phantom_type='msl', pad=4, M=1):
    phantom = shepp_logan(phantom_type)
    images = np.zeros(((n + 2 * pad)**2, M))
    pix = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(pix, -pix)
    
    if pad > 0:
        z1 = np.zeros((n + 2 * pad, pad))
        z2 = np.zeros((pad, n))
         
    for i in range(M):
        if default:
            curr_phantom = phantom
        else:
            curr_phantom = modify(phantom)
        
        image = generateImage(curr_phantom, n, X, Y)
        if pad > 0:
            image = np.block([[z1, np.vstack([z2, image, z2]), z1]])
        
        images[:, i] = image.flatten()
    
    return images
    
def generateImage(e, n, X, Y):
    image = np.zeros((n, n))
    e[:, 1] = e[:, 1] ** 2
    e[:, 2] = e[:, 2] ** 2
    e[:, 5] = e[:, 5] * np.pi / 180
    cosp = np.cos(e[:, 5])
    sinp = np.sin(e[:, 5])

    for k in range(e.shape[0]):
        x = X - e[k, 3]
        y = Y - e[k, 4]
        ellipse = ((x * cosp[k] + y * sinp[k]) ** 2) / e[k, 1] + ((y * cosp[k] - x * sinp[k]) ** 2) / e[k, 2]
        idx = np.where(ellipse <= 1)
        if k < 5:
            image[idx] = e[k, 0]
        else:
            image[idx] = image[idx] + e[k, 0]
    
    return image


def modify(phantom):
    """
    Modify the parameters of the phantom to introduce random variations.
    
    Parameters:
    phantom : np.ndarray
              Original parameters of the phantom.

    Returns:
    phantom : np.ndarray
              Modified parameters of the phantom.
    """
    m = phantom.shape[0]

    # Generate random scaling
    scale = min(1 - (np.random.rand() * 2 / 9), 0.7)
    phantom[:, 1:5] = scale * phantom[:, 1:5]

    # Random rotation
    rotation = 2 * 45 * (np.random.rand() - 0.5)
    phantom[:, 5] = rotation + phantom[:, 5]

    # Random translation
    translate = 0.2 * np.random.rand(1, 2)
    phantom[:, 3:5] = translate + phantom[:, 3:5]

    # Randomize density
    density = 2 * 0.1 * (np.random.rand(m, 1) - 0.5)
    phantom[:, 0] = density.flatten() * phantom[:, 0] + phantom[:, 0]
    phantom[:, 0] = np.clip(phantom[:, 0], 0, 1)

    # Remove random ellipses
    obj = 4
    idx = np.random.choice(m-obj, size=np.random.randint(0, m - obj), replace=False)
    phantom = np.delete(phantom, idx+obj, axis=0)

    return phantom

def shepp_logan(phantom_type='msl'):
    """
    Load parameters for the default Shepp-Logan or Modified Shepp-Logan phantom.

    Parameters:
    type : str
           'sl' for standard Shepp-Logan or 'msl' for Modified Shepp-Logan.

    Returns:
    phantom : np.ndarray
              Parameters defining the ellipses in the phantom.
    """
    if phantom_type == 'sl':
        # Standard Shepp-Logan phantom parameters
        phantom = np.array([
            [1, 0.69, 0.92, 0, 0, 0],
            [0.02, 0.6624, 0.8740, 0, -0.0184, 0],
            [0, 0.11, 0.31, 0.22, 0, -18],
            [0, 0.16, 0.41, -0.22, 0, 18],
            [0.01, 0.21, 0.25, 0, 0.35, 0],
            [0.01, 0.046, 0.046, 0, 0.1, 0],
            [0.01, 0.046, 0.046, 0, -0.1, 0],
            [0.01, 0.046, 0.023, -0.08, -0.605, 0],
            [0.01, 0.023, 0.023, 0, -0.606, 0],
            [0.01, 0.023, 0.046, 0.06, -0.605, 0]
        ])
    elif  phantom_type== 'msl':
        # Modified Shepp-Logan phantom parameters
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
    else:
        raise ValueError("No valid phantom type selected.")

    return phantom