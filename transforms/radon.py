# -*- coding: utf-8 -*-
"""Radon transformations."""

import torch
import torch.nn.functional as F
from transforms import fourier, util
import math

__all__ = ["radon_transform", "iradon_transform"]


def get_r_coords(diameter, num):
    if diameter % 2 == 0:
        radius = diameter / 2 - 0.5
        center = -0.5
        return torch.linspace(-radius, radius, num) + center
    else:
        radius = (diameter - 1) / 2
        return torch.linspace(-radius, radius, num)


def expand_diameter(diameter, K):
    expanded_diameter = int(diameter * K)
    if expanded_diameter % 2 == 1:
        expanded_diameter += 1
    return expanded_diameter


def get_kspace_radial(diameter, expanded_diameter, n_projections):
    r = get_r_coords(diameter, expanded_diameter)
    a = torch.linspace(0, torch.pi, n_projections)
    r_grid, a_grid = torch.meshgrid(r, a, indexing="xy")
    x = (
        torch.round((r_grid * torch.cos(a_grid)) * expanded_diameter / diameter)
        % expanded_diameter
    )
    y = (
        torch.round((-r_grid * torch.sin(a_grid)) * expanded_diameter / diameter)
        % expanded_diameter
    )
    return x.long(), y.long()


def radon_transform(image, N=50):
    image = torch.as_tensor(image)
    K = 1.25
    oversamp = 1.25
    width = 4
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    r = get_r_coords(diameter, expanded_diameter)
    a = torch.linspace(0, torch.pi, N)
    r_grid, a_grid = torch.meshgrid(r, a, indexing="xy")
    x = r_grid * torch.cos(a_grid)
    y = -r_grid * torch.sin(a_grid)

    coord = torch.stack([y, x], dim=-1)
    r_tensor = r

    kspace = fourier.nufft(image, coord, oversamp=oversamp, width=width)
    sinogram = (
        fourier.nufft_adjoint(
            kspace,
            r_tensor[:, None],
            oshape=kspace.shape[:-1] + (diameter,),
            oversamp=oversamp,
            width=width,
        )
        * diameter
        / expanded_diameter
        / torch.sqrt(torch.tensor(diameter, dtype=torch.float32))
    )

    return sinogram.real * diameter


def fft_radon_transform(image, N=50, expansion=6):
    image = torch.as_tensor(image)
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, N)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)

    kspace = torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1))
    slices = kspace[..., y, x]
    sinogram = torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(slices, dim=-1), dim=-1), dim=-1
    )

    return sinogram


def fft_radon_to_kspace(image, expansion=6):
    image = torch.as_tensor(image)
    image = pad_image(image)
    diameter = image.shape[-1]
    expanded_diameter = expand_diameter(diameter, expansion)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    image = util.resize(image, oshape)

    kspace = torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1))

    return kspace


def fft_radon_to_image(kspace, size):
    kspace = torch.as_tensor(kspace)

    image = torch.fft.fftshift(torch.fft.ifft2(kspace, dim=(-2, -1)), dim=(-2, -1))

    diagonal = math.ceil(torch.sqrt(torch.tensor(2.0)) * size)
    oshape = image.shape[:-2] + (diagonal, diagonal)
    image = util.resize(image, oshape)
    return unpad_image(image.real)


def pad_image(image):
    image = torch.as_tensor(image)
    diagonal = torch.sqrt(torch.tensor(2.0)) * max(image.shape[-2:])
    pad = [int(torch.ceil(diagonal - s)) for s in image.shape[-2:]]
    new_center = [(s + p) // 2 for s, p in zip(image.shape[-2:], pad)]
    old_center = [s // 2 for s in image.shape[-2:]]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]

    # PyTorch padding format (last dim first)
    torch_pad = []
    for pb, pa in reversed(pad_width):
        torch_pad.extend([pb, pa])
    padded_image = F.pad(image, torch_pad, mode="constant", value=0)

    return padded_image


def unpad_image(image):
    size = int(torch.sqrt(torch.tensor(image.shape[-1] ** 2 / 2)))
    pad_left = (image.shape[-1] - size) // 2
    return image[..., pad_left : pad_left + size, pad_left : pad_left + size]


def get_fourier_filter(diameter, K, oversamp=1.25, width=4):
    size = expand_diameter(diameter, K)
    n = torch.cat(
        (
            torch.arange(1, size / 2 + 1, 2, dtype=torch.int32),
            torch.arange(size / 2 - 1, 0, -2, dtype=torch.int32),
        )
    )
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n.float()) ** 2

    r = get_r_coords(diameter, size) / diameter * size

    fourier_filter = (
        2
        * fourier.nufft(
            torch.fft.fftshift(f), r[:, None], oversamp=oversamp, width=width
        ).squeeze()
        * torch.sqrt(torch.tensor(size, dtype=torch.float32))
    )

    return fourier_filter


def iradon_transform(sinogram, K=1.8):
    sinogram = torch.as_tensor(sinogram)
    oversamp = 1.25
    width = 4
    diameter = sinogram.shape[-1]
    expanded_diameter = expand_diameter(diameter, K)
    N = sinogram.shape[-2]
    r = get_r_coords(diameter, expanded_diameter)
    a = torch.linspace(0, torch.pi, N)
    r_grid, a_grid = torch.meshgrid(r, a, indexing="xy")
    x = r_grid * torch.cos(a_grid)
    y = -r_grid * torch.sin(a_grid)

    fourier_filter = get_fourier_filter(diameter, K, oversamp=oversamp, width=width)

    r_tensor = r
    coord = torch.stack([y, x], dim=-1)

    kspace = fourier.nufft(
        sinogram, r_tensor[:, None], oversamp=oversamp, width=width
    ) * torch.sqrt(torch.tensor(diameter, dtype=torch.float32))
    image = (
        fourier.nufft_adjoint(
            kspace * fourier_filter,
            coord,
            oshape=sinogram.shape[:-2] + (diameter, diameter),
            oversamp=oversamp,
            width=width,
        )
        * diameter
        / expanded_diameter
    )

    return unpad_image(image.real / N * torch.pi / 2.0)
