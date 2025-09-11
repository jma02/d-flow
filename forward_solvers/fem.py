# Code from https://github.com/Forgotten/EIT/
# Converted from numba/scipy to torch

import torch
import torch.linalg as torchla

class Mesh:
    def __init__(self, points, triangles, bdy_idx, vol_idx):
        # self.p    array with the node points (sorted)
        #           type : torch.Tensor dim: (n_p, 2)
        # self.n_p  number of node points
        #           type : int
        # self.t    array with indices of points per segment
        #           type : torch.Tensor dim: (n_s, 3)
        # self.n_t  number of triangles
        #           type : int
        # self.bc.  array with the indices of boundary points
        #           type : torch.Tensor dim: (2)

        self.p = points
        self.t = triangles

        self.n_p = self.p.shape[0]
        self.n_t = self.t.shape[0]

        self.bdy_idx = bdy_idx
        self.vol_idx = vol_idx

        # boundary indices for the triangles in the 
        # boundary 

        # faster search in the for loop
        bdy_idx_set = set(self.bdy_idx.tolist())
        
        self.bdy_idx_t = set()
        for e in range(self.t.shape[0]):  # integration over one triangular element at a time
            nodes = self.t[e, :]
            if   (nodes[0].item() in bdy_idx_set)\
               + (nodes[1].item() in bdy_idx_set)\
               + (nodes[2].item() in bdy_idx_set) >= 2:
                self.bdy_idx_t.add(e)

        # device property for convenience
        self.device = self.p.device


class V_h:
    def __init__(self, mesh):
        # self.mesh Mesh object containg geometric info type: Mesh
        # self.sim  dimension of the space              type: in

        self.mesh = mesh
        self.dim = mesh.n_p


def stiffness_matrix(v_h, sigma_vec):
    ''' S = stiffness_matrix(v_h, sigma_vec)
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

    # we define the arrays for the indicies and the values 
    idx_i = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    idx_j = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    vals = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.float64, device=v_h.mesh.device)

    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = torch.cat([torch.ones((3,1), device=v_h.mesh.device), p[nodes,:]], dim=-1)
        # area of triangle e = half of parallelogram area
        Area = torch.abs(torchla.det(Pe))/2
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = torchla.inv(Pe)
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]
        # element matrix from slopes b,c in grad
        S_local = (sigma_vec[e]*Area)*torch.mm(grad.T, grad)
        
        # add S_local  to 9 entries of global K
        idx_i[e,:] = (torch.ones((3,1), device=v_h.mesh.device)*nodes).T.reshape((9,))
        idx_j[e,:] = (torch.ones((3,1), device=v_h.mesh.device)*nodes).reshape((9,))
        vals[e,:] = S_local.reshape((9,))

    # we add all the indices to make the matrix
    indices = torch.stack([idx_i.reshape((-1,)), idx_j.reshape((-1,))], dim=0)
    S_coo = torch.sparse_coo_tensor(
        indices, 
        vals.reshape((-1,)), 
        size=(v_h.dim, v_h.dim),
        device=v_h.mesh.device
    )

    # Convert to CSR format (PyTorch equivalent of scipy's lil_matrix for efficient operations)
    return S_coo.coalesce().to_sparse_csr()


#####################################################
def mass_matrix(v_h):
    ''' M = mass_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
    '''

    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    idx_i = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    idx_j = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.int64, device=v_h.mesh.device)
    vals = torch.zeros((v_h.mesh.n_t, 9), dtype=torch.float64, device=v_h.mesh.device)

    # local mass matrix (so we don't need to compute it at each iteration)
    MK = 1/12*torch.tensor([[ 2., 1., 1.], 
                           [ 1., 2., 1.],
                           [ 1., 1., 2.]], device=v_h.mesh.device)

    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = torch.cat([torch.ones((3,1), device=v_h.mesh.device), p[nodes,:]], dim=-1)
        # area of triangle e = half of parallelogram area
        Area = torch.abs(torchla.det(Pe))/2
    
        M_local = Area*MK
        
        # add S_local  to 9 entries of global K
        idx_i[e,:] = (torch.ones((3,1), device=v_h.mesh.device)*nodes).T.reshape((9,))
        idx_j[e,:] = (torch.ones((3,1), device=v_h.mesh.device)*nodes).reshape((9,))
        vals[e,:] = M_local.reshape((9,))

    # we add all the indices to make the matrix
    indices = torch.stack([idx_i.reshape((-1,)), idx_j.reshape((-1,))], dim=0)
    M_coo = torch.sparse_coo_tensor(
        indices, 
        vals.reshape((-1,)), 
        size=(v_h.dim, v_h.dim),
        device=v_h.mesh.device
    )

    return M_coo.coalesce().to_sparse_csr()


def projection_v_w(v_h):
    ''' M = mass_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
    '''

    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    idx_i = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.int64, device=v_h.mesh.device)
    idx_j = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.int64, device=v_h.mesh.device)
    vals = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.float64, device=v_h.mesh.device)

    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = torch.cat([torch.ones((3,1), device=v_h.mesh.device), p[nodes,:]], dim=-1)
        # area of triangle e = half of parallelogram area
        Area = torch.abs(torchla.det(Pe))/2

        # add S_local  to 9 entries of global K
        idx_i[e,:] = nodes
        idx_j[e,:] = e*torch.ones((3,), device=v_h.mesh.device)
        vals[e,:] = torch.ones((3,), device=v_h.mesh.device)*Area/3

    # we add all the indices to make the matrix
    indices = torch.stack([idx_i.reshape((-1,)), idx_j.reshape((-1,))], dim=0)
    M_coo = torch.sparse_coo_tensor(
        indices, 
        vals.reshape((-1,)), 
        size=(v_h.dim, v_h.mesh.n_t),
        device=v_h.mesh.device
    )

    return M_coo.coalesce().to_sparse_csr()


def partial_deriv_matrix(v_h):
    ''' Kx, Ky, Surf = partial_deriv_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
        output: Kx matrix to compute weak derivatives
                Ky matrix to compute weak derivative
                M_t mass matrix in W (piece-wise constant matrices)
    '''
    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    # number of triangles
    n_t = v_h.mesh.n_t

    idx_i = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.int64, device=v_h.mesh.device)
    idx_j = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.int64, device=v_h.mesh.device)
    vals_x = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.float64, device=v_h.mesh.device)
    vals_y = torch.zeros((v_h.mesh.n_t, 3), dtype=torch.float64, device=v_h.mesh.device)
    vals_s = torch.zeros((v_h.mesh.n_t, 1), dtype=torch.float64, device=v_h.mesh.device)

    # Assembly the matrix
    for e in range(n_t):  #
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = torch.cat([torch.ones((3,1), device=v_h.mesh.device), p[nodes,:]], dim=-1)
        # area of triangle e = half of parallelogram area
        Area = torch.abs(torchla.det(Pe))/2
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = torchla.inv(Pe)
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]

        Kx_loc = grad[0,:]*Area
        Ky_loc = grad[1,:]*Area

        vals_x[e,:] = Kx_loc
        vals_y[e,:] = Ky_loc

        vals_s[e] = Area

        # saving the indices
        idx_i[e,:] = e*torch.ones((3,), device=v_h.mesh.device)
        idx_j[e,:] = nodes

    indices = torch.stack([idx_i.reshape((-1,)), idx_j.reshape((-1,))], dim=0)
    
    Kx_coo = torch.sparse_coo_tensor(
        indices, 
        vals_x.reshape((-1,)), 
        size=(n_t, p.shape[0]),
        device=v_h.mesh.device
    )

    Ky_coo = torch.sparse_coo_tensor(
        indices, 
        vals_y.reshape((-1,)), 
        size=(n_t, p.shape[0]),
        device=v_h.mesh.device
    )

    # Create diagonal matrix for surface areas
    surf_indices = torch.stack([torch.arange(n_t, device=v_h.mesh.device), 
                               torch.arange(n_t, device=v_h.mesh.device)], dim=0)
    surf = torch.sparse_coo_tensor(
        surf_indices, 
        vals_s.reshape((-1,)), 
        size=(n_t, n_t),
        device=v_h.mesh.device
    )

    return Kx_coo.coalesce().to_sparse_csr(), Ky_coo.coalesce().to_sparse_csr(), surf.coalesce().to_sparse_csr()


def torch_solve(A, b):
    """
    We need to convert to dense to use the torch linalg solver, as
    this enables autodifferentiation :(
    """
    if A.is_sparse:
        # For sparse matrices, convert to dense for solving
        A_dense = A.to_dense()
    else:
        A_dense = A
    
    # this is what spsolve does, but this will error out if matrices are singular
    solution = torch.linalg.lstsq(A_dense, b, rcond=None)
    return solution.solution


def dtn_map(v_h, sigma_vec):

    n_bdy_pts = len(v_h.mesh.bdy_idx)
    n_pts = v_h.mesh.p.shape[0]

    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx

    # build the stiffness matrix
    S = stiffness_matrix(v_h, sigma_vec)
    
    # Convert to dense for indexing (like scipy lil_matrix behavior)
    S_dense = S.to_dense()
    
    # reduced Stiffness matrix (only volumetric dof)
    Sb = S_dense[vol_idx,:][:,vol_idx]
    
    # the boundary data are just direct deltas at each node - match dtype
    bdy_data = torch.eye(n_bdy_pts, device=v_h.mesh.device, dtype=S_dense.dtype)
    
    # building the rhs for the linear system
    Fb = -S_dense[vol_idx,:][:,bdy_idx] @ bdy_data
    
    # solve interior dof
    U_vol = torch_solve(Sb, Fb)
    
    # allocate the space for the full solution - match dtype
    sol = torch.zeros((n_pts, n_bdy_pts), device=v_h.mesh.device, dtype=S_dense.dtype)
    
    # write the corresponding values back to the solution
    sol[bdy_idx,:] = bdy_data
    sol[vol_idx,:] = U_vol

    # computing the flux
    flux = S_dense @ sol

    # extracting the boundary data of the flux 
    DtN = flux[bdy_idx, :]

    return DtN, sol


def adjoint(v_h, sigma_vec, residual):

    n_bdy_pts = len(v_h.mesh.bdy_idx)
    n_pts = v_h.mesh.p.shape[0]

    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx

    # build the stiffness matrix
    # given that the operator is self-adjoint
    S = stiffness_matrix(v_h, sigma_vec)
    
    # Convert to dense for indexing
    S_dense = S.to_dense()
    
    # reduced Stiffness matrix (only volumetric dof)
    Sb = S_dense[vol_idx,:][:,vol_idx]
    
    # the boundary data are just direct deltas at each node
    bdy_data = residual
    
    # building the rhs for the linear system
    Fb = -S_dense[vol_idx,:][:,bdy_idx] @ bdy_data
    
    # solve interior dof
    U_vol = torch_solve(Sb, Fb)
    
    # allocate the space for the full solution - match dtype
    sol_adj = torch.zeros((n_pts, n_bdy_pts), device=v_h.mesh.device, dtype=S_dense.dtype)
    
    # write the corresponding values back to the solution
    sol_adj[bdy_idx,:] = bdy_data
    sol_adj[vol_idx,:] = U_vol

    return sol_adj


def misfit_sigma(v_h, Data, sigma_vec):
    # compute the misfit 

    # compute dtn and sol for given sigma
    dtn, sol = dtn_map(v_h, sigma_vec)

    # compute the residual
    residual = -(Data - dtn)

    # compute the adjoint fields
    sol_adj = adjoint(v_h, sigma_vec, residual)

    # compute the derivative matrices (weakly)
    Kx, Ky, M_w = partial_deriv_matrix(v_h)

    # Convert sparse matrices to dense for solve operations
    M_w_dense = M_w.to_dense()
    
    Sol_adj_x = torch_solve(M_w_dense, torch.sparse.mm(Kx, sol_adj))
    Sol_adj_y = torch_solve(M_w_dense, torch.sparse.mm(Ky, sol_adj))

    Sol_x = torch_solve(M_w_dense, torch.sparse.mm(Kx, sol))
    Sol_y = torch_solve(M_w_dense, torch.sparse.mm(Ky, sol))

    grad = M_w_dense @ torch.sum(Sol_adj_x*Sol_x + Sol_adj_y*Sol_y, dim=1, keepdim=True)

    return 0.5*torch.sum(torch.square(residual)), grad
