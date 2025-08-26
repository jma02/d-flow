# PyTorch-based EIT problem solver with automatic differentiation
# Converted from scipy/numba implementation to pure PyTorch

import torch
import torch.nn as nn
from typing import Tuple, Optional

from fem import Mesh, V_h, stiffness_matrix, mass_matrix, partial_deriv_matrix, torch_solve


class EITSolver(nn.Module):
    """
    PyTorch-based EIT (Electrical Impedance Tomography) solver
    Fully differentiable implementation
    """
    
    def __init__(self, v_h: V_h, device: str = 'cpu'):
        super().__init__()
        self.v_h = v_h
        self.device = device
        self.build_matrices()
    
    def build_matrices(self):
        """Build time-independent matrices"""
        # Mass matrix
        self.Mass = mass_matrix(self.v_h)
        
        # Partial derivative matrices
        Kx, Ky, M_w = partial_deriv_matrix(self.v_h)
        
        # Create derivative operators Dx = M_w^(-1) * Kx
        M_w_diag = torch.diag(M_w)
        M_w_inv = torch.diag(1.0 / M_w_diag)
        
        self.Dx = M_w_inv @ Kx
        self.Dy = M_w_inv @ Ky
        self.M_w = M_w
        
        # Index tensors for boundary/volume separation
        self.vol_idx = self.v_h.mesh.vol_idx
        self.bdy_idx = self.v_h.mesh.bdy_idx
        
    def update_matrices(self, sigma_vec: torch.Tensor):
        """Update conductivity-dependent matrices"""
        # Build stiffness matrix
        self.S = stiffness_matrix(self.v_h, sigma_vec)
        
        # Extract sub-matrices for efficient solving
        self.S_ii = self.S[self.vol_idx, :][:, self.vol_idx]  # interior-interior
        self.S_ib = self.S[self.vol_idx, :][:, self.bdy_idx]  # interior-boundary
    
    def dtn_map(self, sigma_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Dirichlet-to-Neumann map
        
        Args:
            sigma_vec: (n_t,) conductivity per triangle
            
        Returns:
            DtN: (n_bdy, n_bdy) Dirichlet-to-Neumann operator
            sol: (n_p, n_bdy) solution matrix
        """
        self.update_matrices(sigma_vec)
        
        n_bdy_pts = len(self.bdy_idx)
        n_pts = self.v_h.mesh.n_p
        device = sigma_vec.device
        
        # Boundary data: identity matrix (delta functions at each boundary node)
        bdy_data = torch.eye(n_bdy_pts, device=device, dtype=torch.float32)
        
        # Right-hand side for interior problem
        Fb = -self.S_ib @ bdy_data
        
        # Solve interior degrees of freedom
        U_vol = torch_solve(self.S_ii, Fb)
        
        # Assemble full solution
        sol = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=torch.float32)
        sol[self.bdy_idx, :] = bdy_data
        sol[self.vol_idx, :] = U_vol
        
        # Compute flux
        flux = self.S @ sol
        
        # Extract boundary flux (DtN map)
        DtN = flux[self.bdy_idx, :]
        
        return DtN, sol
    
    def adjoint(self, sigma_vec: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Solve adjoint problem
        
        Args:
            sigma_vec: (n_t,) conductivity
            residual: (n_bdy, n_bdy) residual from data fitting
            
        Returns:
            sol_adj: (n_p, n_bdy) adjoint solution
        """
        n_bdy_pts = len(self.bdy_idx)
        n_pts = self.v_h.mesh.n_p
        device = sigma_vec.device
        
        # Boundary data from residual
        bdy_data = residual
        
        # Right-hand side for adjoint problem
        Fb = -self.S_ib @ bdy_data
        
        # Solve interior degrees of freedom
        U_vol = torch_solve(self.S_ii, Fb)
        
        # Assemble adjoint solution
        sol_adj = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=torch.float32)
        sol_adj[self.bdy_idx, :] = bdy_data
        sol_adj[self.vol_idx, :] = U_vol
        
        return sol_adj
    
    def misfit(self, Data: torch.Tensor, sigma_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute misfit and gradient (fully differentiable)
        
        Args:
            Data: (n_bdy, n_bdy) measured data
            sigma_vec: (n_t,) conductivity parameters
            
        Returns:
            misfit: scalar loss value
            grad: (n_t,) gradient with respect to sigma_vec
        """
        # Ensure sigma_vec requires gradients
        if not sigma_vec.requires_grad:
            sigma_vec = sigma_vec.clone().detach().requires_grad_(True)
        
        # Forward solve
        dtn, sol = self.dtn_map(sigma_vec)
        
        # Compute residual
        residual = -(Data - dtn)
        
        # Compute misfit using automatic differentiation
        misfit_val = torch.sqrt(torch.sum(residual ** 2))
        
        # Automatic gradient computation
        grad = torch.autograd.grad(misfit_val, sigma_vec, create_graph=True)[0]
        
        return misfit_val, grad
    
    def forward(self, sigma_vec: torch.Tensor, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for nn.Module interface
        
        Args:
            sigma_vec: (n_t,) conductivity parameters
            data: Optional measured data for computing misfit
            
        Returns:
            DtN map if data is None, else misfit value
        """
        if data is None:
            dtn, _ = self.dtn_map(sigma_vec)
            return dtn
        else:
            misfit_val, _ = self.misfit(data, sigma_vec)
            return misfit_val


@torch.jit.script
def stiffness_matrix_torch_jit(points: torch.Tensor, triangles: torch.Tensor, 
                               sigma_vec: torch.Tensor, n_p: int, n_t: int) -> torch.Tensor:
    """
    JIT-compiled stiffness matrix assembly using PyTorch
    
    Args:
        points: (n_p, 2) node coordinates
        triangles: (n_t, 3) triangle connectivity
        sigma_vec: (n_t,) conductivity per triangle
        n_p: number of points
        n_t: number of triangles
        
    Returns:
        S: (n_p, n_p) stiffness matrix
    """
    device = points.device
    
    # Pre-allocate sparse matrix components
    indices = torch.zeros((2, 9 * n_t), dtype=torch.long, device=device)
    values = torch.zeros(9 * n_t, device=device)
    
    for e in range(n_t):
        nodes = triangles[e, :]  # (3,)
        triangle_points = points[nodes, :]  # (3, 2)
        
        # Build Pe matrix: [1 x y] for each vertex
        ones = torch.ones(3, 1, device=device)
        Pe = torch.cat([ones, triangle_points], dim=1)  # (3, 3)
        
        # Area = |det(Pe)| / 2
        area = torch.abs(torch.det(Pe)) / 2.0
        
        # Gradient matrix from inverse of Pe
        Pe_inv = torch.inverse(Pe)
        grad = Pe_inv[1:3, :]  # Take rows 1,2 (x,y gradients)
        
        # Local stiffness matrix: sigma * area * grad.T @ grad
        S_local = sigma_vec[e] * area * torch.mm(grad.t(), grad)  # (3, 3)
        
        # Global indices for this triangle's contribution
        start_idx = e * 9
        
        # Row indices (repeat each node 3 times)
        rows = nodes.repeat_interleave(3)
        # Column indices (repeat the node vector)
        cols = nodes.repeat(3)
        
        indices[0, start_idx:start_idx+9] = rows
        indices[1, start_idx:start_idx+9] = cols
        values[start_idx:start_idx+9] = S_local.flatten()
    
    # Create sparse tensor and convert to dense
    S_sparse = torch.sparse_coo_tensor(indices, values, (n_p, n_p), device=device)
    return S_sparse.coalesce().to_dense()


def stiffness_matrix_torch(v_h: V_h, sigma_vec: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for JIT-compiled stiffness matrix assembly
    
    Args:
        v_h: Finite element space
        sigma_vec: (n_t,) conductivity values per triangle
        
    Returns:
        S: (n_p, n_p) stiffness matrix
    """
    mesh = v_h.mesh
    return stiffness_matrix_torch_jit(mesh.p, mesh.t, sigma_vec, mesh.n_p, mesh.n_t)


# Utility functions for creating EIT problems
def create_eit_problem(mesh: Mesh, device: str = 'cpu') -> EITSolver:
    """
    Create an EIT solver for a given mesh
    
    Args:
        mesh: Mesh object
        device: torch device
        
    Returns:
        EIT solver instance
    """
    v_h = V_h(mesh)
    return EITSolver(v_h, device)

