import torch

def randomSheppLogan(n=512, default=False, phantom_type='msl', pad=4, M=1, dtype=torch.float64, device='cpu'):
    """
    Generate random Shepp-Logan phantoms.
    
    Parameters:
    n : int
        Size of the grid (n x n).
    default : bool
        If True, use default phantom without modifications.
    phantom_type : str
        'sl' for standard Shepp-Logan or 'msl' for Modified Shepp-Logan.
    pad : int
        Padding around the image.
    M : int
        Number of phantoms to generate.
    dtype : torch.dtype
        Data type for the output tensor.
    device : str or torch.device
        Device to place tensors on.
        
    Returns:
    images : torch.Tensor
        Generated phantom images, flattened.
    """
    phantom_params = shepp_logan(phantom_type, dtype=dtype, device=device)
    images = torch.zeros(((n + 2 * pad)**2, M), dtype=dtype, device=device)
    pix = torch.linspace(-1, 1, n, dtype=dtype, device=device)
    X, Y = torch.meshgrid(pix, -pix, indexing='ij')
    
    if pad > 0:
        z1 = torch.zeros((n + 2 * pad, pad), dtype=dtype, device=device)
        z2 = torch.zeros((pad, n), dtype=dtype, device=device)
         
    for i in range(M):
        if default:
            curr_phantom = phantom_params
        else:
            curr_phantom = modify(phantom_params, device=device)
        
        image = generateImage(curr_phantom, n, X, Y, dtype=dtype, device=device)
        if pad > 0:
            # Create padded image using torch.cat
            top_row = torch.cat([z1, torch.cat([z2, image, z2], dim=0), z1], dim=1)
            image = top_row
        
        images[:, i] = image.flatten()
    
    return images

def generateImage(e, n, X, Y, dtype=torch.float64, device='cpu'):
    """
    Generate a Shepp-Logan phantom image from ellipse parameters.
    
    Parameters:
    e : torch.Tensor
        Ellipse parameters.
    n : int
        Image size.
    X, Y : torch.Tensor
        Coordinate grids.
    dtype : torch.dtype
        Data type for the output tensor.
    device : str or torch.device
        Device to place tensors on.
        
    Returns:
    image : torch.Tensor
        Generated phantom image.
    """
    image = torch.zeros((n, n), dtype=dtype, device=device)
    e = e.clone()  # Don't modify the original
    e[:, 1] = e[:, 1] ** 2
    e[:, 2] = e[:, 2] ** 2
    e[:, 5] = e[:, 5] * torch.pi / 180
    cosp = torch.cos(e[:, 5])
    sinp = torch.sin(e[:, 5])

    for k in range(e.shape[0]):
        x = X - e[k, 3]
        y = Y - e[k, 4]
        ellipse = ((x * cosp[k] + y * sinp[k]) ** 2) / e[k, 1] + ((y * cosp[k] - x * sinp[k]) ** 2) / e[k, 2]
        idx = ellipse <= 1
        if k < 5:
            image[idx] = e[k, 0]
        else:
            image[idx] = image[idx] + e[k, 0]
    
    return image


def phantom(n, P, dtype=torch.float64, device='cpu'):
    """
    Create a Shepp-Logan phantom discretized on an n x n grid.

    Parameters:
    n : int
        Size of the grid (n x n).
    P : torch.Tensor
        Parameters defining the ellipses [density, x_size, y_size, x_center, y_center, angle].
    dtype : torch.dtype
        Data type for the output tensor.
    device : str or torch.device
        Device to place the tensor on.

    Returns:
    phantom : torch.Tensor
              The discretized phantom of size (n, n).
    """
    # Create coordinate grids
    x = torch.linspace(-1, 1, n, dtype=dtype, device=device)
    y = torch.linspace(-1, 1, n, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    phantom_img = torch.zeros((n, n), dtype=dtype, device=device)

    for i in range(P.shape[0]):
        density = P[i, 0]
        a = P[i, 1]  # x-radius
        b = P[i, 2]  # y-radius
        x0 = P[i, 3]  # x-center
        y0 = P[i, 4]  # y-center
        phi = P[i, 5] * torch.pi / 180.0  # Convert angle to radians

        # Apply rotation and translation
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        # Rotate and translate coordinates
        X_rot = cos_phi * (X - x0) + sin_phi * (Y - y0)
        Y_rot = -sin_phi * (X - x0) + cos_phi * (Y - y0)

        # Create ellipse mask
        ellipse = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1.0
        phantom_img = phantom_img + density * ellipse.to(dtype)

    return phantom_img


def modify(phantom, device='cpu'):
    """
    Modify the parameters of the phantom to introduce random variations.
    
    Parameters:
    phantom : torch.Tensor
              Original parameters of the phantom.
    device : str or torch.device
        Device to place tensors on.

    Returns:
    phantom : torch.Tensor
              Modified parameters of the phantom.
    """
    phantom = phantom.clone()  # Don't modify the original
    m = phantom.shape[0]

    # Generate random scaling
    scale = min(1 - (torch.rand(1, device=device) * 2 / 9), 0.7)
    phantom[:, 1:5] = scale * phantom[:, 1:5]

    # Random rotation
    rotation = 2 * 45 * (torch.rand(1, device=device) - 0.5)
    phantom[:, 5] = rotation + phantom[:, 5]

    # Random translation
    translate = 0.2 * torch.rand(1, 2, device=device)
    phantom[:, 3:5] = translate + phantom[:, 3:5]

    # Randomize density
    density = 2 * 0.1 * (torch.rand(m, 1, device=device) - 0.5)
    phantom[:, 0] = density.flatten() * phantom[:, 0] + phantom[:, 0]
    phantom[:, 0] = torch.clamp(phantom[:, 0], 0, 1)

    # Remove random ellipses
    obj = 4
    if m > obj:
        num_to_remove = torch.randint(0, m - obj, (1,), device=device).item()
        if num_to_remove > 0:
            idx = torch.randperm(m - obj, device=device)[:num_to_remove] + obj
            # Convert to list and sort in descending order to avoid index shifting
            idx_list = sorted(idx.tolist(), reverse=True)
            for i in idx_list:
                phantom = torch.cat([phantom[:i], phantom[i+1:]], dim=0)

    return phantom

def shepp_logan(phantom_type='msl', dtype=torch.float64, device='cpu'):
    """
    Load parameters for the default Shepp-Logan or Modified Shepp-Logan phantom.

    Parameters:
    phantom_type : str
           'sl' for standard Shepp-Logan or 'msl' for Modified Shepp-Logan.
    dtype : torch.dtype
        Data type for the output tensor.
    device : str or torch.device
        Device to place the tensor on.

    Returns:
    phantom : torch.Tensor
              Parameters defining the ellipses in the phantom.
    """
    if phantom_type == 'sl':
        # Standard Shepp-Logan phantom parameters
        phantom_params = [
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
        ]
    elif phantom_type == 'msl':
        # Modified Shepp-Logan phantom parameters
        phantom_params = [
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
        ]
    else:
        raise ValueError("No valid phantom type selected.")

    return torch.tensor(phantom_params, dtype=dtype, device=device)
