# -*- coding: utf-8 -*-
"""Radon transformations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transforms import fourier, util
import math

__all__ = ['radon_transform', 'iradon_transform']


def get_r_coords(diameter, num):
    if diameter % 2 == 0:
        radius = diameter / 2 - 0.5
        center = -0.5
        return np.linspace(-radius, radius, num) + center
    else:
        radius = (diameter - 1) / 2
        return np.linspace(-radius, radius, num)


def expand_diameter(diameter, K):
    expanded_diameter = int(diameter * K)
    if expanded_diameter % 2 == 1:
        expanded_diameter += 1
    return expanded_diameter


def get_kspace_radial(diameter, expanded_diameter, n_projections):
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, n_projections, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing='xy')
    x = np.round((r_grid * np.cos(a_grid)) * expanded_diameter / diameter) % expanded_diameter
    y = np.round((-r_grid * np.sin(a_grid)) * expanded_diameter / diameter) % expanded_diameter
    return x.astype(np.int32), y.astype(np.int32)


def radon_transform(image, N=50):
    K = 1.25
    oversamp = 1.25
    width = 4
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, N, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing='xy')
    x = r_grid * np.cos(a_grid)
    y = -r_grid * np.sin(a_grid)

    # Convert to torch tensors if input is torch
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        coord = torch.stack([torch.from_numpy(y).to(image.device).float(), 
                           torch.from_numpy(x).to(image.device).float()], dim=-1)
        r_tensor = torch.from_numpy(r).to(image.device).float()
    else:
        coord = np.stack([y, x], axis=-1)
        r_tensor = r

    kspace = fourier.nufft(image, coord, oversamp=oversamp, width=width)
    sinogram = fourier.nufft_adjoint(kspace, r_tensor[:, None], oshape=kspace.shape[:-1] + (diameter,),
                                   oversamp=oversamp, width=width) * diameter / expanded_diameter / np.sqrt(diameter)
    
    if is_torch:
        return sinogram.real * diameter
    else:
        return sinogram.real * diameter


def fft_radon_transform(image, N=50, expansion=6):
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, N)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)
    
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        kspace = torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1))
        slices = kspace[..., torch.from_numpy(y).to(image.device), torch.from_numpy(x).to(image.device)]
        sinogram = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(slices, dim=-1), dim=-1), dim=-1)
    else:
        kspace = np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))
        slices = kspace[..., y, x]
        sinogram = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(slices, axes=-1), axis=-1), axes=-1)
    
    return sinogram


def fft_radon_to_kspace(image, expansion=6):
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)
    
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        kspace = torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1))
    else:
        kspace = np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))
    
    return kspace


def fft_radon_to_image(kspace, size):
    is_torch = isinstance(kspace, torch.Tensor)
    
    if is_torch:
        image = torch.fft.fftshift(torch.fft.ifft2(kspace, dim=(-2, -1)), dim=(-2, -1))
    else:
        image = np.fft.fftshift(np.fft.ifft2(kspace, axes=(-2, -1)), axes=(-2, -1))
    
    diagonal = math.ceil(np.sqrt(2) * size)
    oshape = image.shape[:-2] + (diagonal, diagonal)
    image = util.resize(image, oshape)
    return unpad_image(image.real)


def pad_image(image):
    is_torch = isinstance(image, torch.Tensor)
    diagonal = np.sqrt(2) * max(image.shape[-2:])
    pad = [int(np.ceil(diagonal - s)) for s in image.shape[-2:]]
    new_center = [(s + p) // 2 for s, p in zip(image.shape[-2:], pad)]
    old_center = [s // 2 for s in image.shape[-2:]]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    
    if is_torch:
        # PyTorch padding format (last dim first)
        torch_pad = []
        for pb, pa in reversed(pad_width):
            torch_pad.extend([pb, pa])
        padded_image = F.pad(image, torch_pad, mode='constant', value=0)
    else:
        pad_width = [(0, 0) for i in image.shape[:-2]] + pad_width
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    return padded_image


def unpad_image(image):
    size = int(np.sqrt(image.shape[-1] ** 2 / 2))
    pad_left = (image.shape[-1] - size) // 2
    return image[..., pad_left:pad_left + size, pad_left:pad_left + size]


def get_fourier_filter(diameter, K, oversamp=1.25, width=4):
    size = expand_diameter(diameter, K)
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    r = get_r_coords(diameter, size) / diameter * size
    
    # Convert to torch if needed
    f_tensor = torch.from_numpy(f) if isinstance(diameter, torch.Tensor) else f
    r_tensor = torch.from_numpy(r) if isinstance(diameter, torch.Tensor) else r
    
    if isinstance(f_tensor, torch.Tensor):
        fourier_filter = 2 * fourier.nufft(torch.fft.fftshift(f_tensor), r_tensor[:, None],
                                         oversamp=oversamp, width=width).squeeze() * np.sqrt(size)
    else:
        fourier_filter = 2 * fourier.nufft(np.fft.fftshift(f), r[:, None],
                                         oversamp=oversamp, width=width).squeeze() * np.sqrt(size)
    
    return fourier_filter


def iradon_transform(sinogram, K=1.8):
    oversamp = 1.25
    width = 4
    diameter = sinogram.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    N = sinogram.shape[-2]
    r = get_r_coords(diameter, expanded_diameter)
    a = np.linspace(0, np.pi, N, endpoint=False)
    r_grid, a_grid = np.meshgrid(r, a, indexing='xy')
    x = r_grid * np.cos(a_grid)
    y = -r_grid * np.sin(a_grid)
    
    is_torch = isinstance(sinogram, torch.Tensor)
    fourier_filter = get_fourier_filter(diameter, K, oversamp=oversamp, width=width)
    
    if is_torch:
        r_tensor = torch.from_numpy(r).to(sinogram.device)
        coord = torch.stack([torch.from_numpy(y).to(sinogram.device), torch.from_numpy(x).to(sinogram.device)], dim=-1)
        if not isinstance(fourier_filter, torch.Tensor):
            fourier_filter = torch.from_numpy(fourier_filter).to(sinogram.device)
    else:
        r_tensor = r
        coord = np.stack([y, x], axis=-1)
    
    kspace = fourier.nufft(sinogram, r_tensor[:, None], oversamp=oversamp, width=width) * np.sqrt(diameter)
    image = fourier.nufft_adjoint(kspace * fourier_filter, coord,
                                oshape=sinogram.shape[:-2] + (diameter, diameter), oversamp=oversamp,
                                width=width) * diameter / expanded_diameter
    
    return unpad_image(image.real / N * np.pi / 2.)