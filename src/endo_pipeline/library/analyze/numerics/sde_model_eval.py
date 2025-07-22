from collections.abc import Callable

import numpy as np
import pysindy as ps


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

    def f(x: np.ndarray, u: float | None = None) -> np.ndarray:
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

    def f_mesh(mesh_grid: tuple[np.ndarray], u: float | None = None) -> np.ndarray:
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack([mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1)

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

    def f_i(x: np.ndarray, u: float | None = None) -> np.ndarray:
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
