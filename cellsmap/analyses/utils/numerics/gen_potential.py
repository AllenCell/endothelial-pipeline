import numpy as np
from typing import Tuple

import cellsmap.analyses.utils.numerics.fp_solvers as fps

def generalized_potential(P:np.ndarray, grid=None, tol:float=1e-8) -> np.ndarray:
    '''
    Compute the generalized potential U = -ln(P) corresponding to 
    stationary probability density P.

    Arguments:
        P       stationary probability density (callable function or ND array of dimensions n1 x n2 x ... x nN)
        grid    meshgrid (indexing = 'ij') of points in the state space (required if P is callable)
        tol     set P<tol to tol to avoid log(0) errors

    Returns:
        U       generalized potential (ND array of dimensions n1 x n2 x ... x nN) 
    '''

    if callable(P):
        if grid is None:
            raise ValueError('grid must be provided if P is a callable function')
        P = P(*grid)
    
    P[P<tol] = tol

    return -np.log(P)

def gradient_flow_term(U:np.ndarray, D, xArrays:list, ndim:int=2) -> np.ndarray:
    '''
    Compute the gradient flow term -D(x) grad(U) for a given potential U
    and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Arguments:
        U               potential evaluated on a grid (ND array of dimensions n1 x n2 x ... x nN)
        D               diagonal terms of diffusion matrix (callable function R^N -> R^N or N x n1 x n2 x ... nN array)
        xArrays         arrays [x1,x2,..xN] such that U has been evaluated on np.meshgrid(*xArray, indexing = 'ij')
        isConstant      True if D is a constant matrix, False otherwise

    Returns:
        -D(x) gradU       gradient flow term ((N+1)D array of dimensions N x n1 x n2 x ... x nN)
    '''
    if ndim == 1:
        grid = [xArrays]
        dx = [xArrays[1] - xArrays[0]]
    else:
        grid = np.meshgrid(*xArrays,indexing='ij')
        dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]

    if callable(D):
        D_vals = D(*grid)
    else:
        D_vals = D.copy()

    gradU = np.zeros(D_vals.shape)
    for i in range(D_vals.shape[0]):
        gradU[i] = np.gradient(U,dx[i],axis=i,edge_order=2)

    flow_term = np.zeros(D_vals.shape)
    for i in range(D_vals.shape[0]):
        flow_term[i] = -D_vals[i]*gradU[i]

    return flow_term

def expand_to_matrix(D_vals:np.ndarray) -> np.ndarray:
    """
    Expand the vector-valued function D_vals into the matrix-valued function D_vals.

    Arguments:
        D_flat : ndarray
            The (n, N_1, N_2, ..., N_n) array representing the diagonal terms of the diffusion matrix.
    Returns:
        D_vals : ndarray
            The (n,n, N_1, N_2,...,N_n) matrix-valued function.
    """
    n = D_vals.shape[0]
    N_grid = D_vals.shape[1:]
    if N_grid.__class__ == int:
        N_grid = (N_grid,)
    D_mat = np.zeros((n, n) + N_grid)
    
    for i in range(n):
        D_mat[i, i] = D_vals[i]
    
    return D_mat

def probability_flux(P:np.ndarray, f, D, xArrays:list) -> np.ndarray:
    '''
    Compute the probability flux term f(x)P(x) - div(D(x) P(x)) for a given stationary
    probability density P, drift vector field f, and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Arguments:
        P               stationary probability density evaluated on a grid (ND array of dimensions n1 x n2 x ... x nN)
        f               drift vector field (callable function R^N -> R^N or N x n1 x n2 x ... x nN array)
        D               diagonal terms of diffusion matrix (callable function R^N -> R^N or N x n1 x n2 x ... nN array)
        xArrays         arrays [x1,x2,..xN] such that P has been evaluated on np.meshgrid(*xArray, indexing = 'ij')
        isConstant      True if D is a constant matrix, False otherwise

    Returns:
        f(x)P(x) - div(D(x) P(x))       probability flux term ((N+1)D array of dimensions N x n1 x n2 x ... x nN)
    '''

    ndim = len(xArrays)
    grid = np.meshgrid(*xArrays,indexing='ij')
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]

    if callable(f):
        f_vals = f(*grid)
    else:
        f_vals = f.copy()

    if callable(D):
        D_vals = D(*grid)
    else:
        D_vals = D.copy()

    fP = np.zeros_like(f_vals)
    divD = np.zeros_like(f_vals)

    for i in range(D_vals.shape[0]):
        divD[i] = np.gradient(D_vals[i], dx[i], axis=i, edge_order=2)
        fP[i] = f_vals[i] * P

    grad_P = np.gradient(P, *dx, edge_order=2)
    if ndim == 1:
        grad_P = grad_P.reshape((-1,1))
    D_full = expand_to_matrix(D_vals)
    divD = np.sum(divD, axis=0)

    # Generalized Einstein summation
    einsum_str = 'ij' + ''.join([chr(107 + j) for j in range(ndim)]) + ',j' + ''.join([chr(107 + j) for j in range(ndim)]) + '->i' + ''.join([chr(107 + j) for j in range(ndim)])

    return fP - divD * P - np.einsum(einsum_str, D_full, grad_P)

def grad_flux_decomposition(f, D, xArrays:list, ndim:int=2, tol:float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the gradient/flux decomposition of the drift vector field f(x) for 
    stochastic dynamics with diagonal diffusion matrix D(x):
        f(x) = [gradient flow] + [flux term]
    where
        [gradient flow] = -D(x) grad(U) 
        [diffusion geometry] = div(D(x))
        [flux term] = (f(x)P(x) - div(D(x) P(x)))/P(x)
    
    Arguments:
        f               drift vector field (callable function R^N -> R^N or N x n1 x n2 x ... x nN array)
        D               diagonal terms of diffusion matrix (callable function R^N -> R^N or N x n1 x n2 x ... nN array)
        xArrays         arrays [x1,x2,..xN] such that f and D have been (or will be) evaluated on np.meshgrid(*xArray, indexing = 'ij')
        tol             set P<tol to tol to avoid log(0) and divide by 0 errors
        isConstant      True if D is a constant matrix, False otherwise

    Returns:
        U                   generalized potential (n1 x n2 x ... x nN array)
        gradient_term       gradient term (N x n1 x n2 x ... x nN array)
        flux_term           flux term (N x n1 x n2 x ... x nN array)
    '''

    if ndim == 1:
        grid = [xArrays]
        grid_len = len(xArrays)
        dx = xArrays[1] - xArrays[0]
    else:
        grid = np.meshgrid(*xArrays,indexing='ij')
        dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]
        grid_len = [len(xArrays[i]) for i in range(len(xArrays))]
    
    if callable(f):
        f_vals = f(*grid)
    else:
        f_vals = f.copy()
        if ndim == 1:
            f_vals = f_vals.reshape((1,-1))
    
    if callable(D):
        D_vals = D(*grid)
    else:
        D_vals = D.copy()
        if ndim == 1:
            D_vals = D_vals.reshape((1,-1))

    # solve stationary Fokker-Planck equation
    stationary_fp = fps.SteadyFP(grid_len, dx)
    P = stationary_fp.solve(f_vals,D_vals)
    P[P<tol] = tol
    P = P/np.sum(P)

    # get generalized potential
    U = -np.log(P)

    # compute gradient flow term
    gradient_term = gradient_flow_term(U,D_vals,xArrays,ndim=ndim)

    # compute divergence of D(x)
    if ndim == 1:
        divD = np.gradient(D_vals,dx,edge_order=2)
    else:
        divD = np.zeros(D_vals.shape)
        for i in range(D.shape[0]):
            divD[i] = np.gradient(D[i],dx[i],axis=i,edge_order=2)
        divD = np.sum(divD,axis=0)

    # compute flux term
    flux_term = probability_flux(P,f_vals,D_vals,xArrays)/P

    return U, gradient_term, divD, flux_term

def invert_D(D:np.ndarray) -> np.ndarray:
    if len(D.shape) == 2:
        return np.linalg.inv(D)
    else:
        if len(D.shape) == 4:
            ndim = D.shape[0]
            N_grid = D.shape[2:]
            D_ = D.reshape((ndim,ndim,-1))
        elif len(D.shape) == 3:
            ndim = D.shape[0]
            N_grid = None
            D_ = D.copy()
        D_inv = np.zeros(D_.shape)
        for i in range(D_.shape[-1]):
            D_inv[:,:,i] = np.linalg.inv(D_[:,:,i])
        if N_grid is not None:
            return D_inv.reshape((ndim,ndim) + N_grid)
        else:
            return D_inv

def entropy_production(J:np.ndarray, D:np.ndarray, P:np.ndarray, x:list) -> float:
    ndim = len(x)
    D_inv = invert_D(D)

    # Generalized Einstein summation for matrix multiplication
    einsum_str = 'ij' + ''.join([chr(107 + j) for j in range(ndim)]) + ',j' + ''.join([chr(107 + j) for j in range(ndim)]) + '->i' + ''.join([chr(107 + j) for j in range(ndim)])
    D_inv_J = np.einsum(einsum_str, D_inv, J)

    einsum_ip = 'i' + ''.join([chr(107 + j) for j in range(ndim)]) + ',i' + ''.join([chr(107 + j) for j in range(ndim)]) + '->' + ''.join([chr(107 + j) for j in range(ndim)])
    inner_prod = np.einsum(einsum_ip,D_inv_J,J/P)
    weighted_entropy_prod = inner_prod * P
    dx = np.prod([x[i][1] - x[i][0] for i in range(ndim)])
    integral = np.sum(weighted_entropy_prod) * dx

    return integral



