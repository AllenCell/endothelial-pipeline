from collections.abc import Callable

import numpy as np
import pysindy as ps

import cellsmap.analyses.utils.numerics.fp_solvers as fps


def vector_field_function(sindy_model: ps.SINDy) -> Callable:
    """
    Turn fit regression model (SINDy object) into vector-valued
    function f that can be evaluated at a point x as f(x) using
    the model's built-in `predict' function. This function serves
    to wrap the model's predict function into a callable function
    that can be used to evaluate the model at a single point x.

    The returned function allows for control parameters
    as an optional argument.

    This function is used when the dimension
    of the state space is greater than 1.
    """

    def f(x, u=None):
        # if x is a single point, convert to 2D array
        # input shape has to be (n_samples, n_features),
        # where n_features = dimension of state space
        if len(x.shape) == 1:
            x_in = x[None, :]
        else:
            x_in = x.copy()

        # optionally pass control parameter u to the model
        if u is None:
            f_out = sindy_model.predict(x_in)  # predict = evaluate the model at x
        else:
            f_out = sindy_model.predict(
                x_in, u=u
            )  # predict = evaluate the model at x with control parameter u

        # if the output is a single point, convert to 1D array
        if f_out.shape[0] == 1:
            f_out = f_out[0]

        # SINDy model outputs its own class
        # of arrays (AxesArray), so convert to numpy array
        return np.asarray(f_out)

    return f  # return the the callable function f


def mesh_grid_function(f: Callable, ndim: int = 2) -> Callable:
    """
    Turn vector-valued function f(x,u) into function f_mesh(x,u)
    that can be evaluated appropriately on a mesh grid (i.e.,
    np.meshgrid(*arrays) object), where x is the state variable
    and u is the (optional) control parameter.

    If the dimension of the state variable is greater than 2,
    the dimension must be passed in explicitly as ndim.
    It is assumed that the state variable x and the vector
    field f(x) are of the same dimension.

    The returned function allows for control parameters
    as an optional argument.
    """

    def f_mesh(mesh_grid, u=None):
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack(
            [mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1
        )

        # Evaluate the function on all grid points
        v_flat = np.apply_along_axis(f, 1, grid_points, u)

        # Reshape the result to match the original grid shape
        v = v_flat.reshape(*mesh_shape, ndim)

        return np.asarray(v)  # return the result as a numpy array (not AxesArray)

    return f_mesh  # return the the callable function f_mesh


def vector_field_component(f: Callable, i: int) -> Callable:
    """
    Return the scalar valued function corresponding to
    the i-th component (indexing starting at 0)
    of the vector-valued function f(x,u), where x is
    the state variable and u is the (optional) control parameter.

    The returned function allows for control parameters
    as an optional argument.
    """

    def f_i(x, u=None):
        # check input type - this is done because this
        # component function is meant to be passed into the pplane
        # script, which alternatively passes in a meshgrid,
        # an array of points, or single points as tuples
        if isinstance(x, tuple) or isinstance(x, list):
            # if passing in a meshgrid, make sure
            # to evaluate the function on the meshgrid
            if len(x[0].shape) == 2:
                f_mesh = mesh_grid_function(f)
                # transpose output so that the components are the first axis
                f_out = f_mesh(x, u).T
            else:  # if single point in ND, convert tuple to array
                # transpose output so that the components are the first axis
                f_out = f(np.array(x).reshape(-1, len(x)), u).T
        # if passing in an array of points,
        # evaluate the function on the array of points
        else:
            # transpose output so that the components are the first axis
            f_out = f(x, u).T
        # get the i-th component of the vector field,
        # transpose to get correct shape (n_samples,1)
        return f_out[i].T

    return f_i  # return the the callable function f_i


def get_normalization_constant(p_fit: np.ndarray, dx: list) -> float:
    """
    Get normalization constant for stationary probability
    distribution p_fit. The normalization constant is the
    integral of the probability distribution over the state space.

    Inputs:
    - p_fit: np.ndarray, stationary probability
        distribution of the fit SDE model
        - shape N[1] x N[2] x ... x N[ndim]
    - dx: list, bin width in each dimension

    Outputs:
    - c: float, normalization constant
    """
    ndim = len(dx)  # number of dimensions

    # copy p_fit to avoid modifying the original array
    c = p_fit.copy()
    for i in range(ndim):
        # integrate over axis=0 as we marginalize over each dimension
        c = np.trapz(c, dx=dx[i], axis=0)

    return c


def get_stationary_probability(
    drift_vals: np.ndarray, diff_vals: np.ndarray, bins: list, tol: float = 1e-10
) -> np.ndarray:
    """
    Get stationary probability distribution of fit SDE (Langevin) model
    with drift function f and diffusion D by solving the
    stationary Fokker-Planck equation. The drift and diffusion functions
    can be scalar-valued (ndim == 1) or vector-valued (ndim > 1).

    This function calls the PDE solver SteadyFP implemented in the
    `cellsmap.analyses.utils.numerics.fp_solvers' module.

    Inputs:
    - drift_vals: np.ndarray, values of the drift function
        evaluated at the bin centers
        - if the drift function is scalar-valued, f_vals is a 1D array
        - if the drift function is vector-valued, f_vals is an
            (ndim+1)D array with shape (ndim, N_x, N_y, ...)
    - diff_vals: np.ndarray, values of the diffusion function
        evaluated at the bin centers
        - if the diffusion function is scalar-valued, D_vals is a 1D array
        - if the diffusion function is vector-valued, D_vals is an
            (ndim+1)D array with shape (ndim, N_x, N_y, ...)
    - bins: list of arrays defining bin edges for each dimension
        of the state variable
    - tol: float, tolerance for small values in the stationary
        probability distribution (default is 1e-10)
        - if the probability distribution is less than tol, it is set to tol

    Outputs:
    - p_fit: np.ndarray, stationary probability
        distribution of the fit SDE model
    """

    ndim = len(bins)
    # bin width in each dimension
    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]
    # bin centers in each dimension
    num_bins = [len(bins[i]) - 1 for i in range(ndim)]

    # initialize SteadyFP object
    fp = fps.SteadyFP(num_bins, dx)

    # solve stationary Fokker-Planck equation
    p_fit = fp.solve(drift_vals, diff_vals)

    # set small values to a small number to avoid numerical issues
    p_fit[p_fit < tol] = tol
    # integrate to get normalization constant
    c = get_normalization_constant(p_fit, dx)
    # normalize probability distribution
    p_fit = p_fit / c

    return p_fit
