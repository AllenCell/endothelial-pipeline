import numpy as np
from scipy.integrate import simpson

from src.endo_pipeline.library.analyze.diffae_feature_dynamics.model_eval import (
    get_normalization_constant,
)
from src.endo_pipeline.library.analyze.numerics import fp_solvers as fps


def gradient_flow_term(potential: np.ndarray, diffusion: np.ndarray, x: list) -> np.ndarray:
    """
    Compute the gradient flow term -D(x) grad(U) for a given potential U
    and diagonal diffusion matrix D(x)=diag[D1(x),D2(x),...,Dd(x)].

    Inputs:
    - potential: np.ndarray, generalized potential U evaluated
        on a d-D grid (n1 x n2 x ... x nd array)
    - diffusion: np.ndarray, diagonal terms of diffusion matrix
        evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    - x: list, arrays [x1,x2,..xd] such that diff.
        has been evaluated on np.meshgrid(*x, indexing = 'ij')

    Outputs:
    - flow_term: np.ndarray, gradient flow term = -D(x) grad(U)
        (dimensions d x n1 x n2 x ... x nd)
    """

    d = len(x)  # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(d)]  # grid spacing

    # compute gradient of generalized potential U
    grad_u = np.zeros(diffusion.shape)
    for i in range(diffusion.shape[0]):
        grad_u[i] = np.gradient(potential, dx[i], axis=i, edge_order=2)

    # compute gradient flow term = -D(x) grad(U)
    flow_term = np.zeros(diffusion.shape)
    for i in range(diffusion.shape[0]):
        flow_term[i] = -diffusion[i] * grad_u[i]

    return flow_term


def compute_flux_terms(
    p: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    dx: list,
    additive_noise: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the terms needed for the stationary probability flux
        J(x) = f(x)P(x) - div(D(x) P(x))
    for a given drift vector field f and diagonal diffusion
    matrix D with stationary probability P
    (solution to stationary Fokker-Planck Equation).

    Terms computed:
    - f(x)P(x)
    - div(D(x)) * P
    - D(x) grad(P)


    Inputs:
    - p: np.ndarray, stationary probability density evaluated on
        a d-D grid (n1 x n2 x ... x nd array)
    - drift: np.ndarray, drift vector field evaluated on
        a d-D grid (d x n1 x n2 x ... x nd array)
    - diffusion: np.ndarray, diagonal terms of diffusion matrix
        evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    - dx: list, grid spacing in each dimension
    - additive_noise: bool, if True, assume additive noise (default),
        else multiplicative noise
        - if additive noise, div(D) = 0, as D is constant

    Outputs:
    - f_p: np.ndarray, f(x)P(x) (dimensions N x n1 x n2 x ... x nN)
    - div_d_p: np.ndarray, div(D(x)) * P (dimensions n1 x n2 x ... x nN)
    - d_grad_p: np.ndarray, D(x) grad(P) (dimensions N x n1 x n2 x ... x nN)
    """
    d = len(dx)  # number of dimensions

    f_p = np.zeros_like(drift)  # initialize array to store f(x)P(x)
    div_d_p = np.zeros_like(drift)  # initialize array to store div(D(x)) * P(x)
    d_grad_p = np.zeros_like(drift)  # initialize array to store D(x) grad(P)
    # take advantage of diagonal matrix structure:
    # element i of D(x)*grad(P(x)) is D[i]*gradP[i]
    for i in range(d):
        # f(x)P(x)
        f_p[i] = drift[i] * p

        # div(D): sum of gradients of diagonal terms
        # multiply by P(x) to get div(D(x)) * P(x) at end of loop
        if not additive_noise:
            div_d_p[i] = div_d_p[i] + np.gradient(diffusion[i], dx[i], axis=i, edge_order=2)

        # grad(P): numerical gradient
        grad_p_i = np.gradient(p, dx[i], axis=i, edge_order=2)

        # compute D(x) grad(P)
        d_grad_p[i] = diffusion[i] * grad_p_i

    if not additive_noise:
        # multiply div(D) by P(x)
        div_d_p = div_d_p * p

    return f_p, div_d_p, d_grad_p


def probability_flux(
    p: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    x: list,
    additive_noise: bool,
) -> np.ndarray:
    """
    Compute the probability flux term
        J(x) = f(x)P(x) - div(D(x) P(x))
    for a given stationary probability density P,
    drift vector field f, and diagonal diffusion matrix
    D(x)=diag[D1(x),D2(x),...,Dd(x)].

    Inputs:
    - p: np.ndarray, stationary probability density evaluated on
        a d-D grid (n1 x n2 x ... x nd array)
    - drift: np.ndarray, drift vector field evaluated on
        a d-D grid (d x n1 x n2 x ... x nd array)
    - diffusion: np.ndarray, diagonal terms of diffusion matrix
        evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    - x: list, arrays [x1,x2,..xd] such that drift and diff. have
        been evaluated on np.meshgrid(*x, indexing = 'ij')
    - additive_noise: bool, if True, assume additive noise (default),
        else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)

    Outputs:
    - p_flux: np.ndarray, probability flux term of the
        Fokker-Planck equation (dimensions d x n1 x n2 x ... x nd)
        J(x) = f(x)P(x) - div(D(x) P(x))
    """

    d = len(x)  # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(d)]  # grid spacing

    # D grad P
    f_p, div_d_p, d_grad_p = compute_flux_terms(p, drift, diffusion, dx, additive_noise)

    # compute probability flux term
    p_flux = f_p - div_d_p - d_grad_p

    return p_flux


def grad_flux_decomposition(
    drift: np.ndarray,
    diffusion: np.ndarray,
    x: list,
    additive_noise: bool,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gradient/flux decomposition of the drift vector field f(x) for
    stochastic dynamics with diagonal diffusion matrix D(x):
        f(x) = [gradient flow] + [diffusion geometry] + [flux term]
    where
        [gradient flow] = -D(x) grad(U),
        [diffusion geometry] = div(D(x)),
        [flux term] = (f(x)P(x) - div(D(x) P(x)))/P(x).

    Inputs:
    - drift: np.ndarray, drift vector field evaluated on
        a d-D grid (d x n1 x n2 x ... x nd array)
    - diffuion: np.ndarray, diagonal terms of diffusion matrix
        evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    - x: list, arrays [x1,x2,..xd] such that drift and diff.
        have been evaluated on np.meshgrid(*x, indexing = 'ij')
    - tol: float, set p = tol where p<tol to avoid log(0)
        and divide by 0 errors (p is stationary probability density)
    - additive_noise: bool, if True, assume additive noise,
        else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)

    Returns:
    - potential: np.ndarray, generalized potential
        U := -ln P evaluated on a grid (n1 x n2 x ... x nd array)
    - gradient_term: np.ndarray, gradient flow term = -D(x) grad(U)
        (dimensions d x n1 x n2 x ... x nd)
    - diffustion_geometry: np.ndarray, contribution from
        multiplicative noise (dimensions n1 x n2 x ... x nd)
        - computed as the remainder of f - gradient_term - flux_term
    - flux_term: np.ndarray,
        flux term = (f(x)P(x) - div(D(x) P(x)))/P(x)
        (dimensions d x n1 x n2 x ... x nd)
    """

    d = len(x)
    dx = [x[i][1] - x[i][0] for i in range(d)]
    num_bins = [len(x[i]) for i in range(d)]

    if d == 1:  # expand array dimensions for 1D case
        drift = drift[None, :]
        diffusion = diffusion[None, :]

    # numerical solution of Fokker-Planck equation
    # initialize stationary Fokker-Planck solver
    stationary_fp = fps.SteadyFP(num_bins, dx)

    # solve for stationary probability density
    p = stationary_fp.solve(drift, diffusion)
    p[p < tol] = tol  # set values less than tol to tol to avoid log(0) and divide by 0 errors
    c = get_normalization_constant(p, dx)  # compute normalization constant
    p = p / c  # normalize

    # get generalized potential
    potential = -np.log(p)

    # compute gradient flow term
    gradient_term = gradient_flow_term(potential, diffusion, x)

    # compute flux term
    flux = probability_flux(
        p, drift, diffusion, x, additive_noise
    )  # compute probability flux term J(x) = f(x)P(x) - div(D(x) P(x))
    flux_term = flux / p  # flux term in decomposition is J/P

    # remainder is term from gradient of multiplicative noise tensor
    diffusion_geometry = drift - gradient_term - flux_term

    return potential, gradient_term, diffusion_geometry, flux_term


def entropy_production(
    p: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    x: list,
    additive_noise: bool,
) -> float:
    """
    Compute the entropy production rate for a given stationary probability
    P(x) via the Fokker-Planck equation
    with drift vector field f(x) and diffusion matrix D(x).

    Inputs:
    - p: np.ndarray, stationary probability density evaluated on
        a d-D grid (n1 x n2 x ... x nd array)
    - drift: np.ndarray, drift vector field evaluated on
        a d-D grid (d x n1 x n2 x ... x nd array)
    - diffusion: np.ndarray, diagonal terms of diffusion matrix
        evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    - x: list, arrays [x1,x2,..xd] such that drift and diff.
        have been evaluated on np.meshgrid(*x, indexing = 'ij')
    - additive_noise: bool, if True, assume additive noise (default),
        else multiplicative noise
        - additive noise: D = const, non-additive noise: D = D(x)

    Outputs:
    - integral: float, entropy production rate
        - computed numerically as the integral of
            ||D grad U + F||^2 * P(x)
          over the grid x, where U(x) = -log P(x)
          is the generalized potential
        - Simpson's rule is used for numerical integration
    """

    flux = probability_flux(
        p, drift, diffusion, x, additive_noise
    )  # compute probability flux term J(x) = f(x)P(x) - div(D(x) P(x))

    # epr = integral{ <F(x), J(x)> } dx
    # initialize with inner product of f and J
    epr = np.sum(drift * flux, axis=0)

    # numerical integration via Simpson's rule
    d = len(x)  # number of dimensions
    dx = [x[i][1] - x[i][0] for i in range(d)]  # grid spacing

    for i in range(d):  # integrate over each dimension
        epr = simpson(epr, dx=dx[i], axis=0)

    return epr
