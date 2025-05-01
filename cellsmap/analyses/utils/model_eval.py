from typing import Callable

import fipy
import numpy as np
import pysindy as ps

import cellsmap.analyses.utils.numerics.fp_solvers as fps


def vector_field_function(model: ps.SINDy) -> Callable:
    """
    Turn fit regression model (SINDy object) into vector-valued function f that
    can be evaluated at a point x as f(x) using the model's built-in
    `predict' function. This function serves to wrap the model's predict function
    into a callable function that can be used to evaluate the model at a single point x.

    The returned function allows for control parameters as an optional argument.

    This function is used when the dimension of the state space is greater than 1.
    """

    def f(x, u=None):
        # if x is a single point, convert to 2D array
        # input shape has to be (n_samples, n_features), where n_features = dimension of state space
        if len(x.shape) == 1:
            x_in = x[None, :]
        else:
            x_in = x.copy()

        # optionally pass control parameter u to the model
        if u is None:
            f_out = model.predict(x_in)  # predict = evaluate the model at x
        else:
            f_out = model.predict(
                x_in, u=u
            )  # predict = evaluate the model at x with control parameter u

        # if the output is a single point, convert to 1D array
        if f_out.shape[0] == 1:
            f_out = f_out[0]

        # SINDy model outputs its own class of arrays (AxesArray), so convert to numpy array
        return np.asarray(f_out)

    return f  # return the the callable function f


def mesh_grid_function(f: Callable, ndim: int = 2) -> Callable:
    """
    Turn vector-valued function f(x,u) into function f_mesh(x,u) that can be evaluated
    appropriately on a mesh grid (i.e., np.meshgrid(*arrays) object), where x is the state variable
    and u is the (optional) control parameter.

    If the dimension of the state variable is greater than 2, the dimension must be passed in explicitly as ndim.
    It is assumed that the state variable x and the vector field f(x) are of the same dimension.

    The returned function allows for control parameters as an optional argument.
    """

    def f_mesh(mesh_grid, u=None):
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack(
            [mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1
        )

        # Evaluate the function on all grid points
        V_flat = np.apply_along_axis(f, 1, grid_points, u)

        # Reshape the result to match the original grid shape
        V = V_flat.reshape(*mesh_shape, ndim)

        return np.asarray(V)  # return the result as a numpy array (not AxesArray)

    return f_mesh  # return the the callable function f_mesh


def vector_field_component(f: Callable, i: int) -> Callable:
    """
    Returns the scalar valued function corresponding to the i-th component (indexing starting at 0)
    of the vector-valued function f(x,u), where x is the state variable
    and u is the (optional) control parameter.

    The returned function allows for control parameters as an optional argument.
    """

    def f_i(x, u=None):
        # check input type - this is done because this component function is meant to be passed into the pplane
        # script, which alternatively passes in a meshgrid, an array of points, or single points as tuples
        if isinstance(x, tuple) or isinstance(x, list):
            if (
                len(x[0].shape) == 2
            ):  # if passing in a meshgrid, make sure to evaluate the function on the meshgrid
                f_mesh = mesh_grid_function(f)
                f_out = f_mesh(
                    x, u
                ).T  # transpose output so that the components are the first axis
            else:  # if single point in ND, convert tuple to array
                f_out = f(
                    np.array(x).reshape(-1, len(x)), u
                ).T  # transpose output so that the components are the first axis
        else:  # if passing in an array of points, evaluate the function on the array of points
            f_out = f(
                x, u
            ).T  # transpose output so that the components are the first axis
        return f_out[
            i
        ].T  # get the i-th component of the vector field, transpose to get correct shape (n_samples,1)

    return f_i  # return the the callable function f_i


def get_normalization_constant(p_fit: np.ndarray, dx: list) -> float:
    """
    Get normalization constant for stationary probability distribution p_fit.
    The normalization constant is the integral of the probability distribution over the state space.

    Inputs:
    - p_fit: np.ndarray, stationary probability distribution of the fit SDE model
        - shape N[1] x N[2] x ... x N[ndim]
    - dx: list, bin width in each dimension

    Outputs:
    - C: float, normalization constant
    """
    ndim = len(dx)  # number of dimensions

    C = p_fit.copy()  # copy p_fit to avoid modifying the original array
    for i in range(ndim):
        # integrate over each dimension
        C = np.trapz(
            C, dx=dx[i], axis=0
        )  # integrate over the i-th dimension, axis=0 as we marginalize over each dimension

    return C


def get_stationary_probability(
    f_vals: np.ndarray, D_vals: np.ndarray, bins: list, tol: float = 1e-10
) -> np.ndarray:
    """
    Get stationary probability distribution of fit SDE (Langevin) model with drift function f and diffusion D by solving the
    stationary Fokker-Planck equation. The drift and diffusion functions can be scalar-valued (ndim == 1) or vector-valued (ndim > 1).

    This function calls the PDE solver SteadyFP implemented in the `numerics.fp_solvers' module.

    Inputs:
    - f_vals: np.ndarray, values of the drift function evaluated at the bin centers
        - if the drift function is scalar-valued, f_vals is a 1D array
        - if the drift function is vector-valued, f_vals is an (ndim+1)D array with shape (ndim, N_x, N_y, ...)
    - D_vals: np.ndarray, values of the diffusion function evaluated at the bin centers
        - if the diffusion function is scalar-valued, D_vals is a 1D array
        - if the diffusion function is vector-valued, D_vals is an (ndim+1)D array with shape (ndim, N_x, N_y, ...)
    - bins: list of arrays defining bin edges for each dimension of the state variable
    - tol: float, tolerance for small values in the stationary probability distribution (default is 1e-10)
        - if the probability distribution is less than tol, it is set to tol

    Outputs:
    - p_fit: np.ndarray, stationary probability distribution of the fit SDE model
    """

    ndim = len(bins)
    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]  # bin width in each dimension
    Nbins = [len(bins[i]) - 1 for i in range(ndim)]  # number of bins in each dimension

    # initialize SteadyFP object
    if ndim == 1:
        # if 1D, inputs Nbins (number of bins) and dx (bin width) to SteadyFP must be scalars
        fp = fps.SteadyFP(Nbins[0], dx[0])
    else:
        # if 2D, inputs Nbins (number of bins) and dx (bin width) to SteadyFP must be lists (one for each dimension)
        fp = fps.SteadyFP(Nbins, dx)

    p_fit = fp.solve(f_vals, D_vals)  # solve stationary Fokker-Planck equation

    p_fit[p_fit < tol] = (
        tol  # set small values to a small number to avoid numerical issues
    )
    C = get_normalization_constant(p_fit, dx)  # integrate to get normalization constant
    p_fit = p_fit / C  # normalize probability distribution

    return p_fit


def get_stationary_probability_fipy(
    f: Callable, D: Callable, bins: list, u: float, tol: float = 1e-10
) -> np.ndarray:
    """
    Get stationary probability distribution of fit SDE (Langevin) model with drift function f and diffusion D by solving the
    stationary Fokker-Planck equation.

    This function is currently only implemented to accept two-dimensional vector valued drift and diffusion functions.

    This function calls the finite volume PDE solver FiPy to solve the stationary Fokker-Planck equation.

    Inputs:
    - f: Callable, drift function of the SDE model
    - D: Callable, diffusion function of the SDE model
    - bins: list of lists of bin edges for each dimension of the state variable
    - u: float, control parameter (shear stress)
    - tol: float, tolerance for small values in the stationary probability distribution (default is 1e-10)
        - if the probability distribution is less than tol, it is set to tol

    Outputs:
    - p_fit: np.ndarray, stationary probability distribution of the fit SDE model
    """

    Nbins = [len(bins[i]) - 1 for i in range(2)]  # number of bins in each dimension
    dx = [bins[i][1] - bins[i][0] for i in range(2)]  # bin width in each dimension
    bin_min = [bins[i][0] for i in range(2)]  # minimum value of bin in each dimension

    # get drift and diffusion components, used to define terms in the Fokker-Planck equation
    f1 = vector_field_component(f, 0)
    f2 = vector_field_component(f, 1)
    D1 = vector_field_component(D, 0)
    D2 = vector_field_component(D, 1)

    # create fipy mesh
    mesh = fipy.Grid2D(dx=dx[0], dy=dx[1], nx=Nbins[0], ny=Nbins[1])
    x, y = mesh.cellCenters  # get cell centers
    x_ = (
        x.reshape((Nbins[1], Nbins[0])) + bin_min[0]
    )  # reshape to 2D array, shift by bin_min (fipy mesh is defined with origin at (0,0))
    y_ = (
        y.reshape((Nbins[1], Nbins[0])) + bin_min[1]
    )  # reshape to 2D array, shift by bin_min (fipy mesh is defined with origin at (0,0))
    f1_vals = f1([x_, y_], u)[None, :].reshape(
        1, Nbins[1], Nbins[0]
    )  # evaluate drift function at cell centers
    f2_vals = f2([x_, y_], u)[None, :].reshape(
        1, Nbins[1], Nbins[0]
    )  # evaluate drift function at cell centers
    f_vals = np.array(
        np.concatenate([f1_vals, f2_vals])
    )  # concatenate drift components
    D1_vals = D1([x_, y_], u)[None, :].reshape(
        1, Nbins[1], Nbins[0]
    )  # evaluate diffusion function at cell centers
    D2_vals = D2([x_, y_], u)[None, :].reshape(
        1, Nbins[1], Nbins[0]
    )  # evaluate diffusion function at cell centers
    D_vals = np.array(
        np.concatenate([D1_vals, D2_vals])
    )  # concatenate diffusion components

    # get Div(D)
    divD = np.zeros_like(f_vals)
    for i in range(D_vals.shape[0]):
        divD[i] = np.gradient(D_vals[i], dx[i], axis=i, edge_order=2)

    # reshape arrays for fipy
    f_vals = f_vals.reshape(2, -1)
    D_vals = D_vals.reshape(2, -1)
    divD = divD.reshape(2, -1)

    # create fipy variables for initializing the PDE
    p = fipy.CellVariable(
        mesh=mesh, name=r"$P$", value=1 / (Nbins[0] * Nbins[1])
    )  # this is the probability distribution, variable to solve for

    # psi = f(x) - div (D(x))
    psi = fipy.CellVariable(mesh=mesh, value=[f_vals[0] - divD[0], f_vals[1] - divD[1]])
    D = fipy.CellVariable(mesh=mesh, value=[D_vals[0], D_vals[1]])

    # define the Fokker-Planck equation via FiPy's ConvectionTerm and DiffusionTerm
    eq = fipy.ConvectionTerm(coeff=psi, var=p) == fipy.DiffusionTerm(coeff=D, var=p)
    keep_solving = True
    while keep_solving:
        res = eq.sweep(var=p)  # sweep to solve the PDE
        if res < 1e-6:  # if residual is small, stop solving, else, sweep again
            keep_solving = False

    p_fit = p.value.reshape(
        Nbins[1], Nbins[0]
    ).T  # transpose to get expected shape for downstream visualization
    p_fit[p_fit < tol] = (
        tol  # set small values to a small number to avoid numerical issues
    )
    C = get_normalization_constant(p_fit, dx)  # integrate to get normalization constant
    p_fit = p_fit / C  # normalize probability distribution

    return p_fit
