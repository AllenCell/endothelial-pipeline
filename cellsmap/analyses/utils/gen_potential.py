import numpy as np
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

def generalized_potential(P,grid=None,tol=1e-8):
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

def gradient_flow_term(U,D,xArrays,isConstant=False):
    '''
    Compute the gradient flow term -D(x) grad(U) + div(D(x)) for a given potential U
    and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,DN(x)].

    Arguments:
        U               potential evaluated on a grid (ND array of dimensions n1 x n2 x ... x nN)
        D               diagonal terms of diffusion matrix (callable function R^N -> R^N or N x n1 x n2 x ... nN array)
        xArrays         arrays [x1,x2,..xN] such that U has been evaluated on np.meshgrid(*xArray, indexing = 'ij')
        isConstant      True if D is a constant matrix, False otherwise

    Returns:
        -D(x) gradU + div(D(x))       gradient flow term ((N+1)D array of dimensions N x n1 x n2 x ... x nN)
    '''

    grid = np.meshgrid(*xArrays,indexing='ij')
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]

    if callable(D):
        D = D(*grid)

    divD = np.zeros(D.shape)
    if not isConstant: # if constant, div(D) = 0
        for i in range(D.shape[0]):
            divD[i] = np.gradient(D[i],dx[i],axis=i,edge_order=2)
    divD = np.sum(divD,axis=0)

    gradU = np.zeros(D.shape)
    for i in range(D.shape[0]):
        gradU[i] = np.gradient(U,dx[i],axis=i,edge_order=2)

    flow_term = np.zeros(D.shape)
    for i in range(D.shape[0]):
        flow_term[i] = -D[i]*gradU[i]

    return flow_term + divD

def probability_flux(P,f,D,xArrays):
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

    grid = np.meshgrid(*xArrays,indexing='ij')
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]

    if callable(f):
        f = f(*grid)

    if callable(D):
        D = D(*grid)

    divDP = np.zeros(D.shape)
    fP = np.zeros(D.shape)
    for i in range(D.shape[0]):
        divDP[i] = np.gradient(D[i]*P,dx[i],axis=i,edge_order=2)
        fP[i] = f[i]*P

    divDP = np.sum(divDP,axis=0)

    return fP - divDP

def grad_flux_decomposition(f,D,xArrays,tol=1e-8,isConstant=False):
    '''
    Compute the gradient/flux decomposition of the drift vector field f(x) for 
    stochastic dynamics with diagonal diffusion matrix D(x):
        f(x) = [gradient flow] + [flux term]
    where
        [gradient flow] = -D(x) grad(U) + div(D(x))
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

    if len(xArrays) == 1:
        grid = xArrays[0]
    else:
        grid = np.meshgrid(*xArrays,indexing='ij')
    dx = [xArrays[i][1] - xArrays[i][0] for i in range(len(xArrays))]
    grid_len = [len(xArrays[i]) for i in range(len(xArrays))]

    if callable(f):
        f_vals = f(*grid)
    else:
        f_vals = f.copy()
    
    if callable(D):
        D_vals = D(*grid)
    else:
        D_vals = D.copy()

    # solve stationary Fokker-Planck equation
    stationary_fp = fps.SteadyFP(grid_len, dx)
    P = stationary_fp.solve(f_vals,D_vals)
    P[P<tol] = tol

    # get generalized potential
    U = generalized_potential(P)

    # compute gradient flow term
    gradient_term = gradient_flow_term(U,D_vals,xArrays,isConstant=isConstant)

    # compute flux term
    flux_term = f_vals - gradient_term

    return U,gradient_term,flux_term



