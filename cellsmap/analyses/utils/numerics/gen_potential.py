import numpy as np
from numba import njit
from scipy.integrate import simpson
from typing import Tuple

import cellsmap.analyses.utils.numerics.fp_solvers as fps

def gradient_flow_term(U:np.ndarray, D:np.ndarray, x:list) -> np.ndarray:
    '''
    Compute the gradient flow term -D(x) grad(U) for a given potential U
    and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Inputs:
    - U: np.ndarray, potential evaluated on an ND grid (dimensions n1 x n2 x ... x nN)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*x, indexing = 'ij')

    Outputs:
    - flow_term: np.ndarray, gradient flow term = -D(x) grad[U] (dimensions N x n1 x n2 x ... x nN)
    '''

    N = len(x) # number of dimensions 
    dx = [x[i][1] - x[i][0] for i in range(N)] # grid spacing

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

def probability_flux(P:np.ndarray, f: np.ndarray, D:np.ndarray, x:list) -> np.ndarray:
    '''
    Compute the probability flux term f(x)P(x) - div(D(x) P(x)) for a given stationary
    probability density P, drift vector field f, and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Inputs:
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*x, indexing = 'ij')

    Outputs:
    - p_flux: np.ndarray, probability flux term of Fokker-Planck equation (dimensions N x n1 x n2 x ... x nN)
        J(x) = f(x)P(x) - div(D(x) P(x))
    '''

    N = len(x) # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(len(x))] # grid spacing

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

def grad_flux_decomposition(f:np.ndarray, D:np.ndarray, x:list, tol:float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the gradient/flux decomposition of the drift vector field f(x) for 
    stochastic dynamics with diagonal diffusion matrix D(x):
        f(x) = [gradient flow] + [diffusion geometry] + [flux term]
    where
        [gradient flow] = -D(x) grad(U) 
        [diffusion geometry] = div(D(x))
        [flux term] = (f(x)P(x) - div(D(x) P(x)))/P(x)
    
    Inputs:
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*x, indexing = 'ij')
    - tol: float, set P = tol where P<tol to avoid log(0) and divide by 0 errors (P is stationary probability density)

    Returns:
    - U: np.ndarray, generalized potential U := -ln P evaluated on a grid (n1 x n2 x ... x nN array)
    - gradient_term: np.ndarray, gradient flow term = -D(x) grad(U) (dimensions N x n1 x n2 x ... x nN)
    - diffustion_geometry: np.ndarray, contribution from multiplicative noise (dimensions n1 x n2 x ... x nN)
        - computed as the remainder of f - gradient_term - flux_term
    - flux_term: np.ndarray, flux term = (f(x)P(x) - div(D(x) P(x)))/P(x) (dimensions N x n1 x n2 x ... x nN)
    '''

    N = len(x)
    dx = [x[i][1] - x[i][0] for i in range(N)]
    grid_len = [len(x[i]) for i in range(N)]
    
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
    gradient_term = gradient_flow_term(U,D,x)

    # compute flux term
    flux_term = probability_flux(P,f,D,x)/P

    # remainder is term from gradient of multiplicative noise tensor
    diffusion_geometry = f - gradient_term - flux_term

    return U, gradient_term, diffusion_geometry, flux_term

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

@njit
def compute_D_gradU_f(U, f, D, dx):
    '''
    Compute the term D(x) grad(U) + f(x) for a given potential U, drift vector field f, and diagonal diffusion matrix D(x).

    Uses numba's njit to speed up computation.

    Inputs:
    - U: np.ndarray, potential evaluated on an ND grid (dimensions n1 x n2 x ... x nN)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - dx: list, grid spacing in each dimension

    Outputs:
    - D_gradU_f: np.ndarray, D(x) grad(U) + f(x) (dimensions N x n1 x n2 x ... x nN)
    '''
    N = len(dx) # number of dimensions

    D_gradU_f = np.zeros_like(f) # initialize array to store D(x) grad(U) + f(x)
    # take advantage of diagonal matrix structure: element i of D(x)*grad(U(x)) is D[i]*gradU[i]
    for i in range(N):
        # grad(U): numerical gradient without using np.gradient
        gradU_i = np.zeros_like(U)
        gradU_i[1:-1] = (U[2:] - U[:-2])/(2*dx[i])
        gradU_i[0] = (U[1] - U[0])/dx[i]
        gradU_i[-1] = (U[-1] - U[-2])/dx[i]

        # compute D(x) grad(U) + f(x)
        D_gradU_f[i] = D[i] * gradU_i + f[i]

    return D_gradU_f

def entropy_production_NEW(P:np.ndarray, f:np.ndarray, D:np.ndarray, x:list) -> float:
    '''
    Compute the entropy production rate for a given stationary probability P(x), 
    drift vector field f(x), diffusion matrix D(x), and probability flux J(x).

    Inputs:
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U, f, and J have been evaluated on np.meshgrid(*x, indexing = 'ij')
    
    Outputs:
    - integral: float, entropy production rate
        - computed numerically as the integral of ||D grad U + F||^2 * P(x) over the grid x, where
            U(x) = -log P(x) is the generalized potential
        - Simpson's rule is used for numerical integration
    '''
    N = len(x) # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(N)] # grid spacing

    # D * gradient of U + f, where U = -log P
    D_gradU_f = compute_D_gradU_f(-np.log(P), f, D, dx)

    # # take advantage of diagonal matrix structure: element i of D(x)*grad(U(x)) is D[i]*gradU[i]
    # D_gradU_f = np.zeros_like(f)
    # for i in range(N):
    #     D_gradU_f[i] = D[i]*np.gradient(-np.log(P),dx[i],axis=i,edge_order=2) + f[i]

    # compute inner product of D grad U + F with itself: ||D grad U + F||^2
    # multiplied by the stationary probability density P(x) = exp(-U(x))
    # variable named epr to initialize integral, this is actually the integrand
    epr = np.sum(D_gradU_f**2, axis=0) * P
    
    # epr = integral{ ||D grad U + F||^2 * P(x) }
    # numerical integration via Simpson's rule
    # integrate over each dimension
    for i in range(N):
        epr = simpson(epr,dx=dx[i],axis=0)

    print("epr:",epr)
    return epr

def entropy_production(J:np.ndarray, D:np.ndarray, P:np.ndarray, x:list) -> float:
    '''
    Compute the entropy production rate for a given probability flux J, diagonal diffusion matrix D(x),
    and stationary probability density P(x).

    Inputs:
    - J: np.ndarray, probability flux evaluated on an ND grid (dimensions N x n1 x n2 x ... x nN)
        - computed as J(x) = f(x)P(x) - div(D(x) P(x)), using probability_flux() function
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - x: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*x, indexing = 'ij')
    
    Outputs:
    - integral: float, entropy production rate
        - computed numerically as the integral of (J D^{-1} J)/P over the grid x
    '''
    N = len(x)
    D_inv = invert_D(D)

    # Generalized Einstein summation for matrix multiplication on a meshgrid
    einsum_str = 'ij' + ''.join([chr(107 + j) for j in range(N)]) + ',j' + ''.join([chr(107 + j) for j in range(N)]) + '->i' + ''.join([chr(107 + j) for j in range(N)])
    D_inv_J = np.einsum(einsum_str, D_inv, J)

    # compute inner product of D_inv_J and J/P
    einsum_ip = 'i' + ''.join([chr(107 + j) for j in range(N)]) + ',i' + ''.join([chr(107 + j) for j in range(N)]) + '->' + ''.join([chr(107 + j) for j in range(N)])
    inner_prod = np.einsum(einsum_ip,D_inv_J,J/P)
    
    # weight numerical integration by P(x)?
    weighted_entropy_prod = inner_prod * P
    dx = np.prod([x[i][1] - x[i][0] for i in range(N)])
    integral = np.sum(weighted_entropy_prod) * dx

    return integral



