import numpy as np
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

def compute_J_terms(P:np.ndarray, f:np.ndarray, D:np.ndarray, dx:list, additive_noise:bool) -> np.ndarray:
    '''
    Compute the terms needed for the stationary probability flux J(x) = f(x)P(x) - div(D(x) P(x))
    for a given drift vector field f and diagonal diffusion matrix D with stationary
    probability P (solution to stationary Fokker-Planck Equation).

    Terms computed:
    - f(x)P(x)
    - div(D(x)) * P
    - D(x) grad(P)


    Inputs:
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - dx: list, grid spacing in each dimension
    - additive_noise: bool, if True, assume additive noise (default), else multiplicative noise
        - if additive noise, div(D) = 0, as D is constant

    Outputs:
    - fP: np.ndarray, f(x)P(x) (dimensions N x n1 x n2 x ... x nN)
    - divD_P: np.ndarray, div(D(x)) * P (dimensions n1 x n2 x ... x nN)
    - D_gradP: np.ndarray, D(x) grad(P) (dimensions N x n1 x n2 x ... x nN)
    '''
    N = len(dx) # number of dimensions

    fP = np.zeros_like(f) # initialize array to store f(x)P(x)
    divD_P = np.zeros_like(f) # initialize array to store div(D(x)) * P(x)
    D_gradP = np.zeros_like(f) # initialize array to store D(x) grad(P)
    # take advantage of diagonal matrix structure: element i of D(x)*grad(P(x)) is D[i]*gradP[i]
    for i in range(N):
        # f(x)P(x)
        fP[i] = f[i] * P

        # div(D): sum of gradients of diagonal terms
        # multiply by P(x) to get div(D(x)) * P(x) at end of loop
        if not additive_noise:
            divD_P[i] = divD_P[i] + np.gradient(D[i], dx[i], axis=i, edge_order=2)

        # grad(P): numerical gradient
        gradP_i = np.gradient(P,dx[i],axis=i,edge_order=2)

        # compute D(x) grad(P)
        D_gradP[i] = D[i] * gradP_i

    if not additive_noise:
        # multiply div(D) by P(x)
        divD_P = divD_P * P

    return fP, divD_P, D_gradP
    

def probability_flux(P:np.ndarray, f: np.ndarray, D:np.ndarray, x:list, additive_noise:bool) -> np.ndarray:
    '''
    Compute the probability flux term J(x) = f(x)P(x) - div(D(x) P(x)) for a given stationary
    probability density P, drift vector field f, and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Inputs:
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U and D have been evaluated on np.meshgrid(*x, indexing = 'ij')
    - additive_noise: bool, if True, assume additive noise (default), else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)

    Outputs:
    - p_flux: np.ndarray, probability flux term of Fokker-Planck equation (dimensions N x n1 x n2 x ... x nN)
        J(x) = f(x)P(x) - div(D(x) P(x))
    '''

    N = len(x) # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(N)] # grid spacing

    # D grad P
    fP, divD_P, D_gradP = compute_J_terms(P, f, D, dx, additive_noise)

    # compute probability flux term
    p_flux = fP - divD_P - D_gradP

    return p_flux

def grad_flux_decomposition(f:np.ndarray, D:np.ndarray, x:list, additive_noise:bool, tol:float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    - additive_noise: bool, if True, assume additive noise, else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)

    Returns:
    - U: np.ndarray, generalized potential U := -ln P evaluated on a grid (n1 x n2 x ... x nN array)
    - gradient_term: np.ndarray, gradient flow term = -D(x) grad(U) (dimensions N x n1 x n2 x ... x nN)
    - diffustion_geometry: np.ndarray, contribution from multiplicative noise (dimensions n1 x n2 x ... x nN)
        - computed as the remainder of f - gradient_term - flux_term
    - flux_term: np.ndarray, flux term = (f(x)P(x) - div(D(x) P(x)))/P(x) (dimensions N x n1 x n2 x ... x nN)
    '''

    N = len(x)
    dx = [x[i][1] - x[i][0] for i in range(N)]
    Nbins = [len(x[i]) for i in range(N)]
    
    if N == 1: # expand array dimensions for 1D case
        f = f[None,:]
        D = D[None,:]

    # numerical solution of Fokker-Planck equation
    # initialize stationary Fokker-Planck solver
    if N == 1:
        stationary_fp = fps.SteadyFP(Nbins[0], dx[0])
    else:
        stationary_fp = fps.SteadyFP(Nbins, dx) 

    P = stationary_fp.solve(f,D) # solve for stationary probability density
    P[P<tol] = tol # set values less than tol to tol to avoid log(0) and divide by 0 errors
    P = P/np.sum(P) # normalize

    # get generalized potential
    U = -np.log(P)

    # compute gradient flow term
    gradient_term = gradient_flow_term(U,D,x)

    # compute flux term
    flux_term = probability_flux(P,f,D,x,additive_noise)/P

    # remainder is term from gradient of multiplicative noise tensor
    diffusion_geometry = f - gradient_term - flux_term

    return U, gradient_term, diffusion_geometry, flux_term

def compute_D_gradU_f(U, f, D, dx):
    '''
    Compute the term D(x) grad(U) + f(x) for a given potential U, drift vector field f, and diagonal diffusion matrix D(x).

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
        # grad(U): numerical gradient
        gradU_i = np.gradient(U,dx[i],axis=i,edge_order=2)

        # compute D(x) grad(U) + f(x)
        D_gradU_f[i] = D[i] * gradU_i + f[i]

    return D_gradU_f

def entropy_production(P:np.ndarray, f:np.ndarray, D:np.ndarray, x:list, additive_noise:bool) -> float:
    '''
    Compute the entropy production rate for a given stationary probability P(x) via the Fokker-Planck equation
    with drift vector field f(x) and diffusion matrix D(x).

    Inputs:
    - P: np.ndarray, stationary probability density evaluated on an ND grid (n1 x n2 x ... x nN array)
    - f: np.ndarray, drift vector field evaluated on an ND grid (N x n1 x n2 x ... x nN array)
    - D: np.ndarray, diagonal terms of diffusion matrix evaluated on an ND grid (dimensions N x n1 x n2 x ... nN)
    - x: list, arrays [x1,x2,..xN] such that U, f, and J have been evaluated on np.meshgrid(*x, indexing = 'ij')
    - additive_noise: bool, if True, assume additive noise (default), else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)
    
    Outputs:
    - integral: float, entropy production rate
        - computed numerically as the integral of ||D grad U + F||^2 * P(x) over the grid x, where
            U(x) = -log P(x) is the generalized potential
        - Simpson's rule is used for numerical integration
    '''


    J = probability_flux(P,f,D,x,additive_noise) # compute probability flux term J(x) = f(x)P(x) - div(D(x) P(x))

    # initialize with inner product of f and J
    epr = np.sum(f*J, axis=0)

    # # D * gradient of U + f, where U = -log P
    # D_gradU_f = compute_D_gradU_f(-np.log(P), f, D, dx)

    # # compute inner product of D grad U + F with itself: ||D grad U + F||^2
    # # multiplied by the stationary probability density P(x) = exp(-U(x))
    # # variable named epr to initialize integral, this is actually the integrand
    # epr = np.sum(D_gradU_f**2, axis=0) * P
    
    # epr = integral{ ||D grad U + F||^2 * P(x) }
    # numerical integration via Simpson's rule
    # integrate over each dimension

    N = len(x) # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(N)] # grid spacing

    for i in range(N):
        epr = simpson(epr,dx=dx[i],axis=0)

    return epr



