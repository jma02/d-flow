# PyTorch-based finite element method with automatic differentiation
# Converted from scipy/numba implementation to pure PyTorch

import torch
import torch.nn as nn
from typing import Tuple, Optional
import warnings

class Mesh:
    def __init__(self, points: torch.Tensor, triangles: torch.Tensor, 
                 bdy_idx: torch.Tensor, vol_idx: torch.Tensor, device: str = 'cpu'):
        """
        PyTorch-based mesh representation
        
        Args:
            points: (n_p, 2) tensor of node coordinates
            triangles: (n_t, 3) tensor of triangle connectivity
            bdy_idx: (n_bdy,) tensor of boundary node indices
            vol_idx: (n_vol,) tensor of volume node indices
            device: torch device
        """
        self.device = device
        
        # Convert to tensors on the specified device
        self.p = points.to(device).float()
        self.t = triangles.to(device).long()
        self.bdy_idx = bdy_idx.to(device).long()
        self.vol_idx = vol_idx.to(device).long()
        
        self.n_p = self.p.shape[0]
        self.n_t = self.t.shape[0]
        
        # Find boundary triangles (triangles with >= 2 boundary nodes)
        self.bdy_idx_t = self._find_boundary_triangles()
    
    def _find_boundary_triangles(self) -> torch.Tensor:
        """Find triangles that have at least 2 boundary nodes"""
        # Create a mask for boundary nodes
        is_bdy = torch.zeros(self.n_p, dtype=torch.bool, device=self.device)
        is_bdy[self.bdy_idx] = True
        
        # Check how many boundary nodes each triangle has
        bdy_count = is_bdy[self.t].sum(dim=1)
        return torch.where(bdy_count >= 2)[0]


class V_h:
    def __init__(self, mesh: Mesh):
        """
        Finite element space (piecewise linear)
        
        Args:
            mesh: Mesh object containing geometric information
        """
        self.mesh = mesh
        self.dim = mesh.n_p


@torch.jit.script
def compute_triangle_geometry(points: torch.Tensor, triangle_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute triangle area and gradient matrix for a single triangle
    
    Args:
        points: (3, 2) coordinates of triangle vertices
        triangle_nodes: (3,) node indices
        
    Returns:
        area: scalar area of triangle
        grad: (2, 3) gradient matrix
    """
    # Build Pe matrix: [1 x y] for each vertex
    ones = torch.ones(3, 1, device=points.device)
    Pe = torch.cat([ones, points], dim=1)  # (3, 3)
    
    # Area = |det(Pe)| / 2
    area = torch.abs(torch.det(Pe)) / 2.0
    
    # Gradient matrix from inverse of Pe
    Pe_inv = torch.inverse(Pe)
    grad = Pe_inv[1:3, :]  # Take rows 1,2 (x,y gradients)
    
    return area, grad


def stiffness_matrix(v_h: V_h, sigma_vec: torch.Tensor) -> torch.Tensor:
    """
    Assemble stiffness matrix using PyTorch
    
    Args:
        v_h: Finite element space
        sigma_vec: (n_t,) conductivity values per triangle
        
    Returns:
        S: (n_p, n_p) sparse stiffness matrix as dense tensor
    """
    mesh = v_h.mesh
    n_t, n_p = mesh.n_t, mesh.n_p
    device = mesh.device
    
    # Pre-allocate sparse matrix components
    # Each triangle contributes 9 entries (3x3 local matrix)
    indices = torch.zeros((2, 9 * n_t), dtype=torch.long, device=device)
    values = torch.zeros(9 * n_t, device=device)
    
    for e in range(n_t):
        nodes = mesh.t[e, :]  # (3,)
        triangle_points = mesh.p[nodes, :]  # (3, 2)
        
        area, grad = compute_triangle_geometry(triangle_points, nodes)
        
        # Local stiffness matrix: sigma * area * grad.T @ grad
        S_local = sigma_vec[e] * area * torch.mm(grad.t(), grad)  # (3, 3)
        
        # Global indices for this triangle's contribution
        start_idx = e * 9
        
        # Row indices (repeat each node 3 times)
        rows = nodes.repeat_interleave(3)  # [n0,n0,n0,n1,n1,n1,n2,n2,n2]
        # Column indices (repeat the node vector)
        cols = nodes.repeat(3)  # [n0,n1,n2,n0,n1,n2,n0,n1,n2]
        
        indices[0, start_idx:start_idx+9] = rows
        indices[1, start_idx:start_idx+9] = cols
        values[start_idx:start_idx+9] = S_local.flatten()
    
    # Create sparse tensor and convert to dense
    S_sparse = torch.sparse_coo_tensor(indices, values, (n_p, n_p), device=device)
    return S_sparse.coalesce().to_dense()


def mass_matrix(v_h: V_h) -> torch.Tensor:
    """
    Assemble mass matrix using PyTorch
    
    Args:
        v_h: Finite element space
        
    Returns:
        M: (n_p, n_p) mass matrix
    """
    mesh = v_h.mesh
    n_t, n_p = mesh.n_t, mesh.n_p
    device = mesh.device
    
    # Local mass matrix (reference element)
    MK = torch.tensor([[2., 1., 1.], 
                       [1., 2., 1.],
                       [1., 1., 2.]], device=device) / 12.0
    
    indices = torch.zeros((2, 9 * n_t), dtype=torch.long, device=device)
    values = torch.zeros(9 * n_t, device=device)
    
    for e in range(n_t):
        nodes = mesh.t[e, :]
        triangle_points = mesh.p[nodes, :]
        
        area, _ = compute_triangle_geometry(triangle_points, nodes)
        M_local = area * MK
        
        start_idx = e * 9
        rows = nodes.repeat_interleave(3)
        cols = nodes.repeat(3)
        
        indices[0, start_idx:start_idx+9] = rows
        indices[1, start_idx:start_idx+9] = cols
        values[start_idx:start_idx+9] = M_local.flatten()
    
    M_sparse = torch.sparse_coo_tensor(indices, values, (n_p, n_p), device=device)
    return M_sparse.coalesce().to_dense()


def partial_deriv_matrix(v_h: V_h) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble partial derivative matrices
    
    Args:
        v_h: Finite element space
        
    Returns:
        Kx: (n_t, n_p) x-derivative matrix
        Ky: (n_t, n_p) y-derivative matrix  
        Surf: (n_t, n_t) diagonal matrix with triangle areas
    """
    mesh = v_h.mesh
    n_t, n_p = mesh.n_t, mesh.n_p
    device = mesh.device
    
    # For Kx and Ky: each triangle has 3 entries per row
    indices_K = torch.zeros((2, 3 * n_t), dtype=torch.long, device=device)
    values_Kx = torch.zeros(3 * n_t, device=device)
    values_Ky = torch.zeros(3 * n_t, device=device)
    
    # For surface area diagonal matrix
    areas = torch.zeros(n_t, device=device)
    
    for e in range(n_t):
        nodes = mesh.t[e, :]
        triangle_points = mesh.p[nodes, :]
        
        area, grad = compute_triangle_geometry(triangle_points, nodes)
        
        # Weak derivatives: grad[0,:] for x, grad[1,:] for y, scaled by area
        Kx_loc = grad[0, :] * area  # (3,)
        Ky_loc = grad[1, :] * area  # (3,)
        
        start_idx = e * 3
        # Row: triangle index repeated 3 times
        # Col: the 3 nodes of this triangle
        indices_K[0, start_idx:start_idx+3] = e
        indices_K[1, start_idx:start_idx+3] = nodes
        
        values_Kx[start_idx:start_idx+3] = Kx_loc
        values_Ky[start_idx:start_idx+3] = Ky_loc
        
        areas[e] = area
    
    Kx_sparse = torch.sparse_coo_tensor(indices_K, values_Kx, (n_t, n_p), device=device)
    Ky_sparse = torch.sparse_coo_tensor(indices_K, values_Ky, (n_t, n_p), device=device)
    
    # Surface area as diagonal matrix
    Surf = torch.diag(areas)
    
    return Kx_sparse.to_dense(), Ky_sparse.to_dense(), Surf


def torch_solve(A: torch.Tensor, b: torch.Tensor, method: str = 'lu') -> torch.Tensor:
    """
    Solve linear system Ax = b using PyTorch
    
    Args:
        A: (n, n) coefficient matrix
        b: (n, k) right-hand side(s)
        method: solving method ('lu', 'cholesky', 'qr')
        
    Returns:
        x: (n, k) solution
    """
    if method == 'lu':
        return torch.linalg.solve(A, b)
    elif method == 'cholesky':
        return torch.cholesky_solve(b, torch.linalg.cholesky(A))
    elif method == 'qr':
        Q, R = torch.linalg.qr(A, mode='reduced')
        return torch.linalg.solve(R, Q.t() @ b)
    else:
        return torch.linalg.solve(A, b)


def dtn_map(v_h: V_h, sigma_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Dirichlet-to-Neumann map
    
    Args:
        v_h: Finite element space
        sigma_vec: (n_t,) conductivity per triangle
        
    Returns:
        DtN: (n_bdy, n_bdy) Dirichlet-to-Neumann operator
        sol: (n_p, n_bdy) solution matrix
    """
    mesh = v_h.mesh
    device = mesh.device
    
    n_bdy_pts = len(mesh.bdy_idx)
    n_pts = mesh.n_p
    
    vol_idx = mesh.vol_idx
    bdy_idx = mesh.bdy_idx
    
    # Build stiffness matrix
    S = stiffness_matrix(v_h, sigma_vec)
    
    # Extract volumetric part
    Sb = S[vol_idx, :][:, vol_idx]  # (n_vol, n_vol)
    
    # Boundary data: identity matrix (delta functions at each boundary node)
    bdy_data = torch.eye(n_bdy_pts, device=device)
    
    # Right-hand side for interior problem
    Fb = -S[vol_idx, :][:, bdy_idx] @ bdy_data  # (n_vol, n_bdy)
    
    # Solve interior degrees of freedom
    U_vol = torch_solve(Sb, Fb)  # (n_vol, n_bdy)
    
    # Assemble full solution
    sol = torch.zeros((n_pts, n_bdy_pts), device=device)
    sol[bdy_idx, :] = bdy_data
    sol[vol_idx, :] = U_vol
    
    # Compute flux
    flux = S @ sol  # (n_pts, n_bdy)
    
    # Extract boundary flux (DtN map)
    DtN = flux[bdy_idx, :]  # (n_bdy, n_bdy)
    
    return DtN, sol


def adjoint(v_h: V_h, sigma_vec: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """
    Solve adjoint problem
    
    Args:
        v_h: Finite element space
        sigma_vec: (n_t,) conductivity
        residual: (n_bdy, n_bdy) residual from data fitting
        
    Returns:
        sol_adj: (n_p, n_bdy) adjoint solution
    """
    mesh = v_h.mesh
    device = mesh.device
    
    n_bdy_pts = len(mesh.bdy_idx)
    n_pts = mesh.n_p
    
    vol_idx = mesh.vol_idx
    bdy_idx = mesh.bdy_idx
    
    # Stiffness matrix (self-adjoint)
    S = stiffness_matrix(v_h, sigma_vec)
    Sb = S[vol_idx, :][:, vol_idx]
    
    # Boundary data from residual
    bdy_data = residual
    
    # Right-hand side
    Fb = -S[vol_idx, :][:, bdy_idx] @ bdy_data
    
    # Solve
    U_vol = torch_solve(Sb, Fb)
    
    # Assemble solution
    sol_adj = torch.zeros((n_pts, n_bdy_pts), device=device)
    sol_adj[bdy_idx, :] = bdy_data
    sol_adj[vol_idx, :] = U_vol
    
    return sol_adj


def misfit_sigma(v_h: V_h, Data: torch.Tensor, sigma_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute misfit and gradient with respect to sigma (fully differentiable)
    
    Args:
        v_h: Finite element space
        Data: (n_bdy, n_bdy) measured data
        sigma_vec: (n_t,) conductivity parameters (requires_grad=True)
        
    Returns:
        misfit: scalar loss value
        grad: (n_t,) gradient with respect to sigma_vec
    """
    # Ensure sigma_vec requires gradients
    if not sigma_vec.requires_grad:
        sigma_vec = sigma_vec.clone().detach().requires_grad_(True)
    
    # Forward solve
    dtn, sol = dtn_map(v_h, sigma_vec)
    
    # Compute misfit (using automatic differentiation)
    residual = Data - dtn
    misfit = 0.5 * torch.sum(residual ** 2)
    
    # Automatic gradient computation
    grad = torch.autograd.grad(misfit, sigma_vec, create_graph=True)[0]
    
    return misfit, grad


class FEMSolver(nn.Module):
    """
    Neural network module wrapper for FEM solver (fully differentiable)
    """
    def __init__(self, mesh: Mesh):
        super().__init__()
        self.v_h = V_h(mesh)
        
    def forward(self, sigma_vec: torch.Tensor, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through FEM solver
        
        Args:
            sigma_vec: (n_t,) conductivity parameters
            data: Optional measured data for computing misfit
            
        Returns:
            DtN map if data is None, else misfit value
        """
        if data is None:
            dtn, _ = dtn_map(self.v_h, sigma_vec)
            return dtn
        else:
            misfit, _ = misfit_sigma(self.v_h, data, sigma_vec)
            return misfit
