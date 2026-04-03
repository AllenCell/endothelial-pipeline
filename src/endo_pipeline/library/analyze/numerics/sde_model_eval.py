from collections.abc import Callable

import numpy as np


def mesh_grid_function(f: Callable, ndim: int = 2) -> Callable:
    """
    Turn a vector-valued function into function that can be evaluated
    appropriately on a mesh grid.

    It is assumed that the state variable x and the vector field f(x) are of the
    same dimension (i.e., f: R^d -> R^d).

    The returned function can be evaluated on a mesh grid, and allows for
    control parameters as an optional argument.

    Parameters
    ----------
    f
        Vector-valued function, where the output is a vector of the same
        dimension as the input state variable.
    ndim
        Dimension of the state variable and vector field (default is 2).
    """

    def f_mesh(mesh_grid: tuple[np.ndarray], u: float | None = None) -> np.ndarray:
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack([mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1)

        # Evaluate the function on all grid points
        v_flat = np.apply_along_axis(f, 1, grid_points, u)

        # Reshape the result to match the original grid shape
        v = v_flat.reshape(*mesh_shape, ndim)

        return np.asarray(v)

    return f_mesh


def vector_field_component(f: Callable, i: int) -> Callable:
    """
    Get a single component of a vector-valued function as a callable function.

    The returned function allows for control parameters as an optional argument.

    Parameters
    ----------
    f
        Vector-valued function, where the output is a vector of the same
        dimension as the input state variable.
    i
        Index of the component to extract (0-based index).
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
