import numpy as np
from typing import Tuple

import cellsmap.analyses.utils.numerics.fp_solvers as fps

def gradient_flow_term(U:np.ndarray, D:np.ndarray, xArrays:list) -> np.ndarray:
    '''
    Compute the gradient flow term -D(x) grad(U) for a given potential U
    and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Inputs:
    - U: np.ndarray, potential evaluated on a grid (dimensions n1 x n2 x ... x nN)
    - D: np.ndarray, diagonal terms of diffusion matrix (dimensions N x n1 x n2 x ... nN)
    - xArrays: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*xArray, indexing = 'ij')

    Outputs:
    - flow_term: np.ndarray, gradient flow term = -D(x) grad[U] (dimensions N x n1 x n2 x ... x nN)
    '''

    N = len(xArrays) # number of dimensions 
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(N)] # grid spacing

    # compute gradient of generalized potential U
    gradU = np.zeros(D.shape)
    for i in range(D.shape[0]):
        gradU[i] = np.gradient(U,dx[i],axis=i,edge_order=2)

    # compute gradient flow term = -D(x) grad(U)
    flow_term = np.zeros(D.shape)
    for i in range(D.shape[0]):
        flow_term[i] = -D[i]*gradU[i]

    return flow_term

def expand_to_matrix(D:np.ndarray) -> np.ndarray:
    """
    Expand the vector-valued function D into the matrix-valued function D_mat.

    Inputs:
    - D: np.ndarray, the array representing the N diagonal terms of the diffusion matrix evaluated on an ND meshgrid
        (dimensions N x n1 x n2 x ... x nN)
    
    Outputs:
    - D_mat: np.ndarray, the matrix-valued function D(x) = diag[D1(x), D2(x), ..., DN(x)] evaluated on an ND meshgrid
        (dimensions N x N x n1 x n2 x ... x nN)
    """

    N = D.shape[0] # number of dimensions of state space
    N_grid = D.shape[1:] # grid dimensions (number of grid points in each dimension)

    if N_grid.__class__ == int: # if 1D grid, convert to tuple
        N_grid = (N_grid,)

    D_mat = np.zeros((N, N) + N_grid) # initialize matrix-valued function
    
    # fill in diagonal terms
    for i in range(N):
        D_mat[i, i] = D[i]
    
    return D_mat

def probability_flux(P:np.ndarray, f: np.ndarray, D:np.ndarray, xArrays:list) -> np.ndarray:
    '''
    Compute the probability flux term f(x)P(x) - div(D(x) P(x)) for a given stationary
    probability density P, drift vector field f, and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Inputs:
    - P: np.ndarray, stationary probability density evaluated on a grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on a grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix (dimensions N x n1 x n2 x ... nN)
    - xArrays: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*xArray, indexing = 'ij')

    Outputs:
    - p_flux: np.ndarray, probability flux term of Fokker-Planck equation (dimensions N x n1 x n2 x ... x nN)
        J(x) = f(x)P(x) - div(D(x) P(x))
    '''

    N = len(xArrays) # number of dimensions
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))] # grid spacing

    fP = np.zeros_like(f)
    divD = np.zeros_like(f)

    # compute 1) divergence of matrix D(x) and 
    # 2) the vector-scalar product f(x) P(x)
    # at each grid point
    for i in range(N):
        divD[i] = np.gradient(D[i], dx[i], axis=i, edge_order=2)
        fP[i] = f[i] * P
    divD = np.sum(divD, axis=0)

    # compute gradient of P(x)
    grad_P = np.gradient(P, *dx, edge_order=2)
    if N == 1: # expand array dimensions for 1D case
        grad_P = grad_P.reshape((-1,1))

    # expand D(x) to matrix form (need for einsum)
    D_full = expand_to_matrix(D)
    
    # Generalized Einstein summation to comput second part of divergence term: D(x) grad(P(x))
    # using einsum to do matrix multiplication in a generalized way at each grid point
    einsum_str = 'ij' + ''.join([chr(107 + j) for j in range(N)]) + ',j' + ''.join([chr(107 + j) for j in range(N)]) + '->i' + ''.join([chr(107 + j) for j in range(N)])

    # compute probability flux term
    p_flux = fP - divD * P - np.einsum(einsum_str, D_full, grad_P) 
    
    return p_flux

def grad_flux_decomposition(f:np.ndarray, D:np.ndarray, xArrays:list, tol:float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the gradient/flux decomposition of the drift vector field f(x) for 
    stochastic dynamics with diagonal diffusion matrix D(x):
        f(x) = [gradient flow] + [flux term]
    where
        [gradient flow] = -D(x) grad(U) 
        [diffusion geometry] = div(D(x))
        [flux term] = (f(x)P(x) - div(D(x) P(x)))/P(x)
    
    Arguments:
        f               drift vector field evaluated on a grid (N x n1 x n2 x ... x nN array)
        D               diagonal terms of diffusion matrix evaluated on a grid (N x n1 x n2 x ... nN array)
        xArrays         arrays [x1,x2,..xN] such that f and D have been evaluated on np.meshgrid(*xArray, indexing = 'ij')
        tol             set P<tol to tol to avoid log(0) and divide by 0 errors

    Returns:
        U                   generalized potential (n1 x n2 x ... x nN array)
        gradient_term       gradient term (N x n1 x n2 x ... x nN array)
        flux_term           flux term (N x n1 x n2 x ... x nN array)
    '''

    N = len(xArrays)
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(N)]
    grid_len = [len(xArrays[i]) for i in range(N)]
    
    if N == 1: # expand array dimensions for 1D case
        f = f[None,:]
        D = D[None,:]

    # solve stationary Fokker-Planck equation
    stationary_fp = fps.SteadyFP(grid_len, dx) # initialize stationary Fokker-Planck solver
    P = stationary_fp.solve(f,D) # solve for stationary probability density
    P[P<tol] = tol # set values less than tol to tol to avoid log(0) and divide by 0 errors
    P = P/np.sum(P) # normalize

    # get generalized potential
    U = -np.log(P)

    # compute gradient flow term
    gradient_term = gradient_flow_term(U,D,xArrays)

    # compute divergence of D(x)
    if N == 1:
        divD = np.gradient(D,dx,edge_order=2)
    else:
        divD = np.zeros(D.shape)
        for i in range(D.shape[0]):
            divD[i] = np.gradient(D[i],dx[i],axis=i,edge_order=2)
        divD = np.sum(divD,axis=0)

    # compute flux term
    flux_term = probability_flux(P,f,D,xArrays)/P

    return U, gradient_term, divD, flux_term

def invert_D(D:np.ndarray) -> np.ndarray:
    if len(D.shape) == 2:
        return np.linalg.inv(D)
    else:
        if len(D.shape) == 4:
            N = D.shape[0]
            N_grid = D.shape[2:]
            D_ = D.reshape((N,N,-1))
        elif len(D.shape) == 3:
            N = D.shape[0]
            N_grid = None
            D_ = D.copy()
        D_inv = np.zeros(D_.shape)
        for i in range(D_.shape[-1]):
            D_inv[:,:,i] = np.linalg.inv(D_[:,:,i])
        if N_grid is not None:
            return D_inv.reshape((N,N) + N_grid)
        else:
            return D_inv

def entropy_production(J:np.ndarray, D:np.ndarray, P:np.ndarray, x:list) -> float:
    N = len(x)
    D_inv = invert_D(D)

    # Generalized Einstein summation for matrix multiplication
    einsum_str = 'ij' + ''.join([chr(107 + j) for j in range(N)]) + ',j' + ''.join([chr(107 + j) for j in range(N)]) + '->i' + ''.join([chr(107 + j) for j in range(N)])
    D_inv_J = np.einsum(einsum_str, D_inv, J)

    einsum_ip = 'i' + ''.join([chr(107 + j) for j in range(N)]) + ',i' + ''.join([chr(107 + j) for j in range(N)]) + '->' + ''.join([chr(107 + j) for j in range(N)])
    inner_prod = np.einsum(einsum_ip,D_inv_J,J/P)
    
    weighted_entropy_prod = inner_prod * P
    dx = np.prod([x[i][1] - x[i][0] for i in range(N)])
    integral = np.sum(weighted_entropy_prod) * dx

    return integral



