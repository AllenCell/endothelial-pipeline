import numpy as np
from scipy.integrate import simpson

from .binning import get_normalization_constant
from .fp_solvers import SteadyFP


def gradient_flow_term(potential: np.ndarray, diffusion: np.ndarray, x: list) -> np.ndarray:
    """
    Compute the gradient flow term -D(x) grad(U) for a given potential U and
    diagonal diffusion matrix `D =diag[D1, D2, ... , Dd]`.

    Parameters
    ----------
    potential
        Generalized potential U evaluated on a d-D grid (n1 x n2 x ... x nd
        array)
    diffusion
        Diagonal terms of diffusion matrix evaluated on a d-D grid (dimensions d
        x n1 x n2 x ... nd)
    x
        List of arrays [x1,x2,..xd] that correspond to the grid points in each
        dimension.
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
    Compute the terms needed for the stationary probability flux.

    The probability flux term is given by

        J(x) = f(x)P(x) - div(D(x) P(x))

    for a given drift vector field `f` and diagonal diffusion matrix `D
    =diag[D1, D2, ... , Dd]` with stationary probability `P` (solution to
    stationary Fokker-Planck Equation).

    This method computes the three terms needed for the probability flux term,
    which are also the terms needed for the gradient/flux decomposition of the
    drift vector field:
        - f(x)P(x)
        - div(D(x)) * P
        - D(x) grad(P)

    Parameters
    ----------
    p
        Stationary probability density evaluated on a d-D grid (n1 x n2 x ... x nd array)
    drift
        Drift vector field evaluated on a d-D grid (d x n1 x n2 x ... x nd array)
    diffusion
        Diagonal terms of diffusion matrix evaluated on a d-D grid (dimensions d x n1 x n2 x ... nd)
    dx
        List of grid spacing in each dimension
    additive_noise
        If True, assume additive noise (default), else multiplicative noise - if additive noise, div(D) = 0, as D is constant
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
    Compute the probability flux term.


    The probability flux term is given by

        J(x) = f(x)P(x) - div(D(x) P(x))

    for a given drift vector field `f` and diagonal diffusion matrix `D
    =diag[D1, D2, ... , Dd]` with stationary probability `P` (solution to
    stationary Fokker-Planck Equation).

    Parameters
    ----------
    p
        Stationary probability density evaluated on a d-D grid (n1 x n2 x ... x
        nd array)
    drift
        Drift vector field evaluated on a d-D grid (d x n1 x n2 x ... x nd
        array)
    diffusion
        Diagonal terms of diffusion matrix evaluated on a d-D grid (dimensions d
        x n1 x n2 x ... nd)
    x
        List of arrays [x1,x2,..xd] that correspond to the grid points in each
        dimension.
    additive_noise
        If True, assume additive noise (default), else multiplicative noise - if
        additive noise, div(D) = 0, as D is constant
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
    Compute the gradient/flux decomposition.

    The gradient/flux decomposition of the drift vector field f(x) for
    stochastic dynamics with diagonal diffusion matrix D(x) is given by
        f(x) = [gradient flow] + [diffusion geometry] + [flux term]
    where
        - [gradient flow] = -D(x) grad(U)
        - [diffusion geometry] = div(D(x))
        - [flux term] = (f(x)P(x) - div(D(x) P(x)))/P(x).

    Parameters
    ----------
    drift
        Drift vector field evaluated on a d-D grid (d x n1 x n2 x ... x nd
        array)
    diffusion
        Diagonal terms of diffusion matrix evaluated on a d-D grid (dimensions d
        x n1 x n2 x ... nd)
    x
        List of arrays [x1,x2,..xd] that correspond to the grid points in each
        dimension.
    additive_noise
        If True, assume additive noise (default), else multiplicative noise - if
        additive noise, div(D) = 0, as D is constant
    tol
        Tolerance for numerical stability when computing potential and flux
        terms - values of P(x) less than tol are set to tol to avoid log(0) and
        divide by 0 errors when computing potential and flux terms.

    Returns
    -------
    :
        Generalized potential U(x) = -log P(x) evaluated on a d-D grid.
    :
        Gradient flow term -D(x) grad(U) evaluated on a d-D grid.
    :
        Diffusion geometry term div(D(x)) evaluated on a d-D grid.
    :
        Flux term (f(x)P(x) - div(D(x) P(x)))/P(x) evaluated on a d-D grid.
    """

    d = len(x)
    dx = [x[i][1] - x[i][0] for i in range(d)]
    num_bins = [len(x[i]) for i in range(d)]

    if d == 1:  # expand array dimensions for 1D case
        drift = drift[None, :]
        diffusion = diffusion[None, :]

    # numerical solution of Fokker-Planck equation
    # initialize stationary Fokker-Planck solver
    stationary_fp = SteadyFP(num_bins, dx)

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
    Compute the entropy production rate.

    For a given stationary probability P(x) via the Fokker-Planck equation with
    drift vector field f(x) and diffusion matrix D(x), the entropy production
    rate is given by

        epr = integral{ <f(x), J(x)> } dx

    where `J(x) = f(x)P(x) - div(D(x) P(x))` is the probability flux term. It
    can also be expressed as

        epr = integral{ ||D grad U + F||^2 * P(x) } dx

    where U(x) = -log P(x) is the generalized potential and F is the flux term
    in the gradient/flux decomposition of the drift vector field.

    This method uses Simpson's rule for numerical integration.

    Parameters
    ----------
    p
        Stationary probability density evaluated on a d-D grid (n1 x n2 x ... x
        nd array)
    drift
        Drift vector field evaluated on a d-D grid (d x n1 x n2 x ... x nd
        array)
    diffusion
        Diagonal terms of diffusion matrix evaluated on a d-D grid (dimensions d
        x n1 x n2 x ... nd)
    x
        List of arrays [x1,x2,..xd] that correspond to the grid points in each
        dimension.
    additive_noise
        If True, assume additive noise (default), else multiplicative noise - if
        additive noise, div(D) = 0, as D is constant.
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
