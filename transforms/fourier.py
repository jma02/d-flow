# -*- coding: utf-8 -*-
"""FFT and non-uniform FFT (NUFFT) functions."""

import torch
import numpy as np
from transforms import util, interp
from math import ceil

__all__ = ["fft", "ifft", "nufft"]


def fft(input, oshape=None, axes=None, center=True, norm=None):
    """FFT function that supports centering."""
    input = torch.as_tensor(input)

    if not torch.is_complex(input):
        input = input.to(torch.complex64)

    if center:
        output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        if axes is None:
            output = torch.fft.fftn(input, s=oshape, norm=norm)
        else:
            output = torch.fft.fftn(input, s=oshape, dim=axes, norm=norm)

    return output


def ifft(input, oshape=None, axes=None, center=True, norm=None):
    """IFFT function that supports centering."""
    input = torch.as_tensor(input)

    if not torch.is_complex(input):
        input = input.to(torch.complex64)

    if center:
        output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        if axes is None:
            output = torch.fft.ifftn(input, s=oshape, norm=norm)
        else:
            output = torch.fft.ifftn(input, s=oshape, dim=axes, norm=norm)

    return output


def nufft(input, coord, oversamp=1.25, width=4):
    """Non-uniform Fast Fourier Transform."""
    input = torch.as_tensor(input)
    coord = torch.as_tensor(coord)

    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    # Apodize
    output = _apodize(input, ndim, oversamp, width, beta)

    # Zero-pad
    output /= util.prod(input.shape[-ndim:]) ** 0.5
    output = util.resize(output, os_shape)

    # FFT
    output = fft(output, axes=range(-ndim, 0), norm=None)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp)
    output = interp.interpolate(
        output, coord, kernel="kaiser_bessel", width=width, param=beta
    )
    output /= width**ndim

    return output


def nufft_adjoint(input, coord, oshape, oversamp=1.25, width=4):
    """Adjoint non-uniform Fast Fourier Transform."""
    input = torch.as_tensor(input)
    coord = torch.as_tensor(coord)

    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output = interp.gridding(
        input, coord, os_shape, kernel="kaiser_bessel", width=width, param=beta
    )
    output /= width**ndim

    # IFFT
    output = ifft(output, axes=range(-ndim, 0), norm=None)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:]) ** 0.5

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta)

    return output


def _fftc(input, oshape=None, axes=None, norm=None):
    ndim = len(input.shape)
    axes = util._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)

    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)

    return output


def _ifftc(input, oshape=None, axes=None, norm=None):
    ndim = len(input.shape)
    axes = util._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)

    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)

    return output


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]

    scale = torch.ceil(
        oversamp * torch.tensor(shape[-ndim:], dtype=torch.float32, device=coord.device)
    ) / torch.tensor(shape[-ndim:], dtype=torch.float32, device=coord.device)
    shift = (
        torch.ceil(
            oversamp
            * torch.tensor(shape[-ndim:], dtype=torch.float32, device=coord.device)
        )
        // 2
    )

    return coord * scale + shift


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i) for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta):
    output = input

    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)

        # Use float dtype for arange, not complex
        idx = torch.arange(i, dtype=torch.float32, device=output.device)

        # Calculate apodization
        apod_arg = beta**2 - (np.pi * width * (idx - i // 2) / os_i) ** 2

        apod = torch.sqrt(apod_arg.clamp(min=0))
        apod = torch.where(apod == 0, torch.ones_like(apod), apod / torch.sinh(apod))
        # Handle NaN/Inf values
        apod = torch.where(torch.isfinite(apod), apod, torch.ones_like(apod))

        # Reshape for broadcasting
        reshape_dims = [1] * len(output.shape)
        reshape_dims[a] = i
        apod = apod.reshape(reshape_dims)

        output = output * apod

    return output


def estimate_shape(coord):
    """Estimate array shape from coordinates."""
    coord = torch.as_tensor(coord)
    ndim = coord.shape[-1]
    shape = [int(coord[..., i].max() - coord[..., i].min()) for i in range(ndim)]

    return shape
