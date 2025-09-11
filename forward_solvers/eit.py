# Code from https://github.com/Forgotten/EIT/
# Converted from numba to torch

import torch
import torch.linalg as torchla

from fem import partial_deriv_matrix, mass_matrix

class EIT:
    def __init__(self, v_h):
        self.v_h = v_h
        self.build_matrices()

    def update_matrices(self, sigma_vec):

        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx

        S = stiffness_matrix(self.v_h, sigma_vec)
        self.S = S
        # Convert to dense for indexing operations
        S_dense = S.to_dense()
        self.S_ii = S_dense[vol_idx,:][:,vol_idx]
        self.S_ib = S_dense[vol_idx,:][:,bdy_idx]

    def build_matrices(self):

        self.Mass = mass_matrix(self.v_h)
        Kx, Ky, M_w = partial_deriv_matrix(self.v_h)

        # Create diagonal matrix from M_w diagonal - convert to dense for operations
        M_w_dense = M_w.to_dense()
        M_w_diag_inv = torch.diag(1.0 / torch.diag(M_w_dense))
        
        self.Dx = M_w_diag_inv @ Kx.to_dense()
        self.Dy = M_w_diag_inv @ Ky.to_dense()
        self.M_w = M_w_dense

    def dtn_map(self, sigma_vec):
        # do this here

        self.update_matrices(sigma_vec)

        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts = self.v_h.mesh.p.shape[0]
    
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
    
        # the boundary data are just direct deltas at each node - match dtype
        bdy_data = torch.eye(n_bdy_pts, device=self.v_h.mesh.device, dtype=self.S_ii.dtype)
        
        # building the rhs for the linear system
        Fb = -self.S_ib @ bdy_data
            
        # solve interior dof
        U_vol = torch.linalg.solve(self.S_ii, Fb)
        
        # allocate the space for the full solution - match dtype
        sol = torch.zeros((n_pts, n_bdy_pts), device=self.v_h.mesh.device, dtype=self.S_ii.dtype)
        
        # write the corresponding values back to the solution
        sol[bdy_idx,:] = bdy_data
        sol[vol_idx,:] = U_vol

        # computing the flux
        flux = self.S.to_dense() @ sol

        # extracting the boundary data of the flux 
        DtN = flux[bdy_idx, :]

        return DtN, sol

    def adjoint(self, sigma_vec, residual):

        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts = self.v_h.mesh.p.shape[0]
    
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        # the boundary data are just direct deltas at each node
        bdy_data = residual
        
        # building the rhs for the linear system
        Fb = -self.S_ib @ bdy_data
        
        # solve interior dof
        U_vol = torch.linalg.solve(self.S_ii, Fb)
        
        # allocate the space for the full solution - match dtype
        sol_adj = torch.zeros((n_pts, n_bdy_pts), device=self.v_h.mesh.device, dtype=self.S_ii.dtype)
        
        # write the corresponding values back to the solution
        sol_adj[bdy_idx,:] = bdy_data
        sol_adj[vol_idx,:] = U_vol

        return sol_adj 

    def misfit(self, Data, sigma_vec):
        # compute the misfit 

        # compute dtn and sol for given sigma
        dtn, sol = self.dtn_map(sigma_vec)

        # compute the residual
        residual = -(Data - dtn)

        # compute the adjoint fields
        sol_adj = self.adjoint(sigma_vec, residual)

        Sol_adj_x = self.Dx @ sol_adj
        Sol_adj_y = self.Dy @ sol_adj

        Sol_x = self.Dx @ sol
        Sol_y = self.Dy @ sol

        grad = self.M_w @ torch.sum(Sol_adj_x*Sol_x + Sol_adj_y*Sol_y, dim=1, keepdim=True)

        return torch.sqrt(torch.sum(torch.square(residual))), grad


def stiffness_matrix(v_h, sigma_vec):
    ''' S = stiffness_matrix_numba(v_h, sigma_vec)
        function to assemble the stiffness matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
               sigma_vec: values of sigma at each 
               triangle
    '''
    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    # ensure that sigma_vec is a tensor it it isn't already
    if not isinstance(sigma_vec, torch.Tensor):
        sigma_vec = torch.tensor(sigma_vec, dtype=torch.float64, device=v_h.mesh.device)
    sigma_vec = sigma_vec.reshape((-1,))
    # we define the arrays for the indicies and the values 
    idx_i = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    idx_j = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    vals = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.float64, device=v_h.mesh.device)

    # we fill the entries with a jitted function
    fill_entries_matrix(idx_i, idx_j, vals, t, p, 
                        sigma_vec, int(t.shape[0]))

    # we add all the indices to make the matrix
    indices = torch.stack([idx_i.reshape((-1,)), idx_j.reshape((-1,))], dim=0)
    S_coo = torch.sparse_coo_tensor(
        indices, 
        vals.reshape((-1,)), 
        size=(v_h.dim, v_h.dim),
        device=v_h.mesh.device
    )

    return S_coo.coalesce().to_sparse_csr()


@torch.jit.script
def fill_array(idx: torch.Tensor, e: int, matrix: torch.Tensor):
    for ii in range(3):
        for jj in range(3):
            idx[e, 3*ii+jj] = matrix[ii, jj]


@torch.jit.script
def fill_entries_matrix(idx_i: torch.Tensor, idx_j: torch.Tensor, vals: torch.Tensor, 
                       t: torch.Tensor, p: torch.Tensor, sigma_vec: torch.Tensor, size_t: int):

    for e in range(size_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = torch.cat((torch.ones((3,1), dtype=torch.float64, device=p.device), 
                       p[nodes,:]), dim=-1)
        # area of triangle e = half of parallelogram area
        Area = torch.abs(torchla.det(Pe))/2
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = torchla.inv(Pe)
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]
        # element matrix from slopes b,c in grad
        grad_t_c = grad.T.clone()
        grad_c = grad.clone()
        
        S_local = (sigma_vec[e]*Area)*torch.mm(grad_t_c, grad_c)

        # add S_local  to 9 entries of global K
        fill_array(idx_i, e, torch.ones((3,1), dtype=torch.int64, device=p.device)*nodes)
        fill_array(idx_j, e, (torch.ones((3,1), dtype=torch.int64, device=p.device)*nodes).T)
        vals[e,:] = S_local.reshape((9,))

