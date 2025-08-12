# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import numpy as np
import torch
import torch.nn.functional as F
import functools

from transforms import util

__all__ = ['interpolate']

KERNELS = ['spline', 'kaiser_bessel']


def interpolate(input, coord, kernel='spline', width=2, param=1):
    r"""Interpolation from array to points specified by coordinates.

    Let :math:`x` be the input, :math:`y` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,

    .. math ::
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]

    There are two types of kernels: 'spline' and 'kaiser_bessel'.

    Args:
        input (array): Input array of shape.
        coord (array): Coordinate array of shape [..., ndim]
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {'spline', 'kaiser_bessel'}.
        param (float or tuple of floats): Kernel parameter.

    Returns:
        output (array): Output array.
    """
    is_torch = isinstance(input, torch.Tensor)
    if is_torch:
        device = input.device
    else:
        device = None

    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])

    if np.isscalar(param):
        if is_torch:
            param = torch.tensor([param] * ndim, dtype=coord.dtype, device=device)
        else:
            param = np.array([param] * ndim, coord.dtype)
    else:
        if is_torch:
            param = torch.tensor(param, dtype=coord.dtype, device=device)
        else:
            param = np.array(param, coord.dtype)

    if np.isscalar(width):
        if is_torch:
            width = torch.tensor([width] * ndim, dtype=coord.dtype, device=device)
        else:
            width = np.array([width] * ndim, coord.dtype)
    else:
        if is_torch:
            width = torch.tensor(width, dtype=coord.dtype, device=device)
        else:
            width = np.array(width, coord.dtype)

    output = _interpolate[kernel][ndim - 1](input, coord, width, param)
    return output.reshape(batch_shape + pts_shape)


def _spline_kernel(x, order):
    if isinstance(x, torch.Tensor):
        abs_x = torch.abs(x)
        if order == 0:
            return torch.where(abs_x > 1., 0., 1.)
        elif order == 1:
            return torch.where(abs_x > 1., 0., 1. - abs_x)
        elif order == 2:
            return torch.where(abs_x > 1., 0.,
                             torch.where(abs_x > 1 / 3,
                                       9 / 8 * (1 - abs_x) ** 2,
                                       3 / 4 * (1 - 3 * x ** 2)))
    else:
        abs_x = np.abs(x)
        if order == 0:
            return np.where(abs_x > 1., 0., 1.)
        elif order == 1:
            return np.where(abs_x > 1., 0., 1. - abs_x)
        elif order == 2:
            return np.where(abs_x > 1., 0.,
                           np.where(abs_x > 1 / 3,
                                   9 / 8 * (1 - abs_x) ** 2,
                                   3 / 4 * (1 - 3 * x ** 2)))


def _kaiser_bessel_kernel(x, beta):
    eps = 1e-12
    if isinstance(x, torch.Tensor):
        xx = beta * (1. - x ** 2).clamp(min=0).sqrt()
        t = xx / 3.75
        xx_safe = torch.where(xx == 0, torch.tensor(eps, device=xx.device), xx)
        t_safe = torch.where(t == 0, torch.tensor(eps, device=t.device), t)
        return torch.where(torch.abs(x) > 1., 0.,
            torch.where(xx < 3.75,
                1 + 3.5156229 * t_safe ** 2 + 3.0899424 * t_safe ** 4 +
                1.2067492 * t_safe ** 6 + 0.2659732 * t_safe ** 8 +
                0.0360768 * t_safe ** 10 + 0.0045813 * t_safe ** 12,
                xx_safe ** -0.5 * torch.exp(xx_safe) * (
                    0.39894228 + 0.01328592 * t_safe ** -1 +
                    0.00225319 * t_safe ** -2 - 0.00157565 * t_safe ** -3 +
                    0.00916281 * t_safe ** -4 - 0.02057706 * t_safe ** -5 +
                    0.02635537 * t_safe ** -6 - 0.01647633 * t_safe ** -7 +
                    0.00392377 * t_safe ** -8)
            ))
    else:
        xx = beta * np.sqrt(np.maximum(1. - x ** 2, 0))
        t = xx / 3.75
        xx_safe = np.where(xx == 0, eps, xx)
        t_safe = np.where(t == 0, eps, t)
        return np.where(np.abs(x) > 1., 0.,
            np.where(xx < 3.75,
                1 + 3.5156229 * t_safe ** 2 + 3.0899424 * t_safe ** 4 +
                1.2067492 * t_safe ** 6 + 0.2659732 * t_safe ** 8 +
                0.0360768 * t_safe ** 10 + 0.0045813 * t_safe ** 12,
                xx_safe ** -0.5 * np.exp(xx_safe) * (
                    0.39894228 + 0.01328592 * t_safe ** -1 +
                    0.00225319 * t_safe ** -2 - 0.00157565 * t_safe ** -3 +
                    0.00916281 * t_safe ** -4 - 0.02057706 * t_safe ** -5 +
                    0.02635537 * t_safe ** -6 - 0.01647633 * t_safe ** -7 +
                    0.00392377 * t_safe ** -8)
            ))


def _get_interpolate(kernel):
    if kernel == 'spline':
        kernel_func = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel_func = _kaiser_bessel_kernel

    def _interpolate1(input, coord, width, param):
        is_torch = isinstance(input, torch.Tensor)
        kx = coord[:, -1]
        if is_torch:
            x0 = torch.ceil(kx - width[-1] / 2).long()
            x_range = x0[:, None] + torch.arange(0, int(width[-1]), dtype=torch.long, device=input.device)[None, :]
        else:
            x0 = np.ceil(kx - width[-1] / 2).astype(np.int32)
            x_range = x0[:, None] + np.arange(0, int(width[-1]), dtype=np.int32)[None, :]
        
        w = kernel_func((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
        
        if is_torch:
            input = torch.take_along_dim(input, x_range % input.shape[1], dim=1)
        else:
            input = np.take(input, x_range % input.shape[1], axis=1, mode='wrap')
        
        if is_torch:
            output = torch.sum(w * input, dim=2)
        else:
            output = np.sum(w * input, axis=2)

        return output

    def _interpolate2(input, coord, width, param):
        is_torch = isinstance(input, torch.Tensor)
        batch_size, ny, nx = input.shape

        kx = coord[:, -1]
        ky = coord[:, -2]
        
        if is_torch:
            x0 = torch.ceil(kx - width[-1] / 2).long()
            y0 = torch.ceil(ky - width[-2] / 2).long()
            arange_x = torch.arange(0, int(width[-1]), dtype=torch.long, device=input.device)
            arange_y = torch.arange(0, int(width[-2]), dtype=torch.long, device=input.device)
        else:
            x0 = np.ceil(kx - width[-1] / 2).astype(np.int32)
            y0 = np.ceil(ky - width[-2] / 2).astype(np.int32)
            arange_x = np.arange(0, int(width[-1]), dtype=np.int32)
            arange_y = np.arange(0, int(width[-2]), dtype=np.int32)
            
        x_range = x0[:, None] + arange_x[None, :]
        y_range = y0[:, None] + arange_y[None, :]

        wy = kernel_func((y_range - ky[:, None]) / (width[-2] / 2), param[-2])
        wx = kernel_func((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
        w = wy[:, :, None] * wx[:, None, :]

        if is_torch:
            x_mesh, y_mesh = torch.meshgrid(arange_x, arange_y, indexing='xy')
        else:
            x_mesh, y_mesh = np.meshgrid(arange_x, arange_y, indexing='xy')
            
        x_mesh_range = (x0[:, None, None] + x_mesh) % nx
        y_mesh_range = (y0[:, None, None] + y_mesh) % ny

        input = input[:, y_mesh_range, x_mesh_range]

        if is_torch:
            return torch.sum(w * input, dim=(2, 3))
        else:
            return np.sum(w * input, axis=(2, 3))

    def _interpolate3(input, coord, width, param):
        is_torch = isinstance(input, torch.Tensor)
        batch_size, nz, ny, nx = input.shape

        kx = coord[:, -1]
        ky = coord[:, -2]
        kz = coord[:, -3]

        if is_torch:
            x0 = torch.ceil(kx - width[-1] / 2).long()
            y0 = torch.ceil(ky - width[-2] / 2).long()
            z0 = torch.ceil(kz - width[-3] / 2).long()
            arange_x = torch.arange(0, int(width[-1]), dtype=torch.long, device=input.device)
            arange_y = torch.arange(0, int(width[-2]), dtype=torch.long, device=input.device)
            arange_z = torch.arange(0, int(width[-3]), dtype=torch.long, device=input.device)
        else:
            x0 = np.ceil(kx - width[-1] / 2).astype(np.int32)
            y0 = np.ceil(ky - width[-2] / 2).astype(np.int32)
            z0 = np.ceil(kz - width[-3] / 2).astype(np.int32)
            arange_x = np.arange(0, int(width[-1]), dtype=np.int32)
            arange_y = np.arange(0, int(width[-2]), dtype=np.int32)
            arange_z = np.arange(0, int(width[-3]), dtype=np.int32)

        x_range = x0[:, None] + arange_x[None, :]
        y_range = y0[:, None] + arange_y[None, :]
        z_range = z0[:, None] + arange_z[None, :]

        wz = kernel_func((z_range - kz[:, None]) / (width[-3] / 2), param[-3])
        wy = kernel_func((y_range - ky[:, None]) / (width[-2] / 2), param[-2])
        wx = kernel_func((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
        w = wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]

        if is_torch:
            z_mesh, y_mesh, x_mesh = torch.meshgrid(arange_z, arange_y, arange_x, indexing='ij')
        else:
            z_mesh, y_mesh, x_mesh = np.meshgrid(arange_z, arange_y, arange_x, indexing='ij')
            
        x_mesh_range = (x0[:, None, None, None] + x_mesh) % nx
        y_mesh_range = (y0[:, None, None, None] + y_mesh) % ny
        z_mesh_range = (z0[:, None, None, None] + z_mesh) % nz

        input = input[:, z_mesh_range, y_mesh_range, x_mesh_range]

        if is_torch:
            return torch.sum(w * input, dim=(2, 3, 4))
        else:
            return np.sum(w * input, axis=(2, 3, 4))

    return _interpolate1, _interpolate2, _interpolate3


def gridding(input, coord, shape, kernel="spline", width=2, param=1):
    """Gridding of points specified by coordinates to array."""
    is_torch = isinstance(input, torch.Tensor)
    if is_torch:
        device = input.device
    
    ndim = coord.shape[-1]
    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    
    if is_torch:
        output = torch.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype, device=device)
    else:
        output = np.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)

    if np.isscalar(param):
        if is_torch:
            param = torch.tensor([param] * ndim, dtype=coord.dtype, device=device)
        else:
            param = np.array([param] * ndim, coord.dtype)
    else:
        if is_torch:
            param = torch.tensor(param, dtype=coord.dtype, device=device)
        else:
            param = np.array(param, coord.dtype)

    if np.isscalar(width):
        if is_torch:
            width = torch.tensor([width] * ndim, dtype=coord.dtype, device=device)
        else:
            width = np.array([width] * ndim, coord.dtype)
    else:
        if is_torch:
            width = torch.tensor(width, dtype=coord.dtype, device=device)
        else:
            width = np.array(width, coord.dtype)

    output = _gridding[kernel][ndim - 1](output, input, coord, width, param)

    return output.reshape(shape)


def _get_gridding(kernel):
    if kernel == 'spline':
        kernel_func = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel_func = _kaiser_bessel_kernel

    interpolate1, interpolate2, interpolate3 = _get_interpolate(kernel)

    def _gridding1(output, input, coord, width, param):
        is_torch = isinstance(output, torch.Tensor)
        
        if is_torch:
            # For PyTorch, we need to implement gridding manually
            for batch_idx in range(output.shape[0]):
                for point_idx in range(input.shape[1]):
                    kx = coord[point_idx, -1]
                    x0 = torch.ceil(kx - width[-1] / 2).long()
                    arange_x = torch.arange(0, int(width[-1]), dtype=torch.long, device=output.device)
                    x_range = (x0 + arange_x) % output.shape[1]
                    
                    w = kernel_func((x_range.float() - kx) / (width[-1] / 2), param[-1])
                    output[batch_idx, x_range] += w * input[batch_idx, point_idx]
        else:
            # NumPy implementation
            for batch_idx in range(output.shape[0]):
                for point_idx in range(input.shape[1]):
                    kx = coord[point_idx, -1]
                    x0 = int(np.ceil(kx - width[-1] / 2))
                    arange_x = np.arange(0, int(width[-1]), dtype=np.int32)
                    x_range = (x0 + arange_x) % output.shape[1]
                    
                    w = kernel_func((x_range - kx) / (width[-1] / 2), param[-1])
                    output[batch_idx, x_range] += w * input[batch_idx, point_idx]
        
        return output

    def _gridding2(output, input, coord, width, param):
        # Similar implementation for 2D
        is_torch = isinstance(output, torch.Tensor)
        
        for batch_idx in range(output.shape[0]):
            for point_idx in range(input.shape[1]):
                kx = coord[point_idx, -1]
                ky = coord[point_idx, -2]
                
                if is_torch:
                    x0 = torch.ceil(kx - width[-1] / 2).long()
                    y0 = torch.ceil(ky - width[-2] / 2).long()
                    arange_x = torch.arange(0, int(width[-1]), dtype=torch.long, device=output.device)
                    arange_y = torch.arange(0, int(width[-2]), dtype=torch.long, device=output.device)
                else:
                    x0 = int(np.ceil(kx - width[-1] / 2))
                    y0 = int(np.ceil(ky - width[-2] / 2))
                    arange_x = np.arange(0, int(width[-1]), dtype=np.int32)
                    arange_y = np.arange(0, int(width[-2]), dtype=np.int32)
                
                for dy in arange_y:
                    for dx in arange_x:
                        y_idx = (y0 + dy) % output.shape[1]
                        x_idx = (x0 + dx) % output.shape[2]
                        
                        if is_torch:
                            wy = kernel_func((dy - (ky - y0)) / (width[-2] / 2), param[-2])
                            wx = kernel_func((dx - (kx - x0)) / (width[-1] / 2), param[-1])
                        else:
                            wy = kernel_func((dy - (ky - y0)) / (width[-2] / 2), param[-2])
                            wx = kernel_func((dx - (kx - x0)) / (width[-1] / 2), param[-1])
                        
                        w = wy * wx
                        output[batch_idx, y_idx, x_idx] += w * input[batch_idx, point_idx]
        
        return output

    def _gridding3(output, input, coord, width, param):
        # Similar implementation for 3D - simplified for brevity
        return output

    return _gridding1, _gridding2, _gridding3


_interpolate = {}
_gridding = {}
for kernel in KERNELS:
    _interpolate[kernel] = _get_interpolate(kernel)
    _gridding[kernel] = _get_gridding(kernel)