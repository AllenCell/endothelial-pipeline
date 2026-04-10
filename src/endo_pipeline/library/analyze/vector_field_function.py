"""Methods related to turning vector field data into callable functions."""

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Literal, overload

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata

logger = logging.getLogger(__name__)


def _fill_nan_for_vtk(data: np.ndarray, method: str = "nearest") -> np.ndarray:
    """
    Replace NaN values in an array for export as a .vtk file.

    This method uses scipy.interpolate.griddata to fill in NaN values in the
    input array, which is necessary for saving as .vtk files for visualization
    in ParaView, as .vtk files cannot contain NaN values.


    Parameters
    ----------
    data
        Input `numpy` array with NaN values to be imputed.
    method
        Interpolation method to use for imputation.

    Returns
    -------
    :
        A copy of the input array with NaNs imputed.

    """
    if method != "nearest":
        logger.warning(
            "Using extrapolation method other than 'nearest' for vtk file construction results in "
            "significant memory usage and computation time. Consider using 'nearest' method. "
        )

    # Create a copy to avoid modifying the original data
    arr = data.copy()

    # Identify NaN locations
    nan_mask = np.isnan(arr)

    # If there are no NaNs, return the original array
    if not np.any(nan_mask):
        return arr

    # Else, proceed with interpolation
    valid_mask = ~nan_mask
    valid_coords = np.array(np.where(valid_mask)).T
    valid_values = arr[valid_mask]

    nan_coords = np.array(np.where(nan_mask)).T

    # Use griddata to perform interpolation and fill NaN values
    imputed_values = griddata(
        points=valid_coords, values=valid_values, xi=nan_coords, method=method
    )

    # Assign the interpolated values back into the NaN locations
    arr[nan_mask] = imputed_values

    return arr


def compute_extrapolated_vector_field(
    kmcs: np.ndarray,
    grid_coordinates: list[np.ndarray],
    method: str = "linear",
    for_vtk_files: bool = False,
) -> dict:
    """
    Extrapolate a 3D vector field from Kramers-Moyal estimates over a specified
    grid.

    **Method inputs**

    The input ``kmcs`` are the Kramers-Moyal coefficients (drift or diffusion
    estimates) over a 3D mesh grid obtained from feature data. Where there are
    no data points, these estimates are `NaN`. This function extrapolates these
    estimates to the entire grid using nearest-neighbor or linear interpolation.

    The array ``kmcs`` should have shape (num_x, num_y, num_z, 3), where num_x,
    num_y, and num_z are the number of points in each dimension of the 3D
    meshgrid defined by ``grid_coordinates``.

    **Method output**

    The output is a dictionary with two keys: - "vectors": tuple of 3D arrays
    (f1,f2,f3) with the vector values in each dimension - "grid": tuple of 3D
    arrays (xgrid, ygrid, zgrid) with the meshgrid points in each dimension

    **Extrapolation for vtk files vs other uses**

    If ``for_vtk_files`` is `True`, the extrapolation is done using a wrapper
    method for filling NaNs via `scipy.interpolate.griddata`, which ensures no
    NaNs remain in the output. This is important for saving the vector field as
    .vtk files for visualization in ParaView, but is slower for large grids.

    If ``for_vtk_files`` is `False`, the extrapolation is done using
    `scipy.interpolate.RegularGridInterpolator`, which is faster for large
    grids, but can produce NaNs outside the convex hull of the data points
    unless they are filled in with a numerical value (here, zero) first when
    creating the interpolator. This will result in NaNs being converted to zeros
    outside the convex hull, which is suitable for all other uses of the vector
    field except vtk files.

    Parameters
    ----------
    kmcs
        Array of drift or diffusion estimates over a three dimensional grid.
    grid_coordinates
        List of 1D numpy arrays with the grid points in each dimension
    method
        Method to use for extrapolating the vector field where there are NaNs.
    for_vtk_files
        Whether the output is intended for saving as .vtk files.

    Returns
    -------
    :
        Dictionary with extrapolated vector field and corresponding grid.

    """
    filled_kmcs = kmcs.copy()
    n_components = filled_kmcs.shape[-1]
    x, y, z = np.meshgrid(*grid_coordinates, indexing="ij")

    for i in range(n_components):
        component = filled_kmcs[..., i]
        nan_mask = np.isnan(component)
        if np.any(nan_mask):
            if for_vtk_files:
                component = _fill_nan_for_vtk(component, method=method)
            else:
                interpolator = RegularGridInterpolator(
                    grid_coordinates,
                    np.where(
                        nan_mask, 0, component
                    ),  # fill NaNs with zeros to avoid producing NaNs outside convex hull
                    method=method,
                    bounds_error=False,
                    fill_value=None,  # extrapolate outside convex hull
                )
                nan_points = np.array([x[nan_mask], y[nan_mask], z[nan_mask]]).T
                component[nan_mask] = interpolator(nan_points)
            filled_kmcs[..., i] = component

    vectors = tuple(filled_kmcs[..., i] for i in range(n_components))
    return {"vectors": vectors, "grid": (x, y, z)}


@overload
def get_callable_vector_field(
    vector_field_dict: dict, for_solve_ivp: Literal[True], method: str = "linear"
) -> Callable[[float, np.ndarray], np.ndarray]: ...


@overload
def get_callable_vector_field(
    vector_field_dict: dict, for_solve_ivp: Literal[False], method: str = "linear"
) -> Callable[[np.ndarray], np.ndarray]: ...


def get_callable_vector_field(
    vector_field_dict: dict, for_solve_ivp: bool = True, method: str = "linear"
) -> Callable[[float, np.ndarray], np.ndarray] | Callable[[np.ndarray], np.ndarray]:
    """Get a callable vector field from a numpy array via linear interpolation.

    The input is a dictionary with the vector field values on a mesh grid, and this function
    creates a callable function that can be used to evaluate the vector field at any point
    using linear interpolation of the values on the mesh grid.

    **Method inputs**

    The input ``vector_field_dict`` is a dictionary with two keys:

    - "vectors": tuple of 3D arrays (V1,V2,V3) with the vector values in each dimension
    - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each dimension

    The boolean ``for_solve_ivp`` specifies whether to return a callable function
    formatted for use with ``scipy.integrate.solve_ivp`` (if ``True``) or a general callable
    function (if ``False``). The difference is that the function for solve_ivp takes in time
    as the first variable and the point in state space as the second variable, while the general
    callable is only a function of the point in state space.

    Parameters
    ----------
    vector_field_dict
        Dictionary with arrays defining a vector field evaluated on a mesh grid.
    for_solve_ivp
        Return a function formatted for use with scipy.integrate.solve_ivp if True.
    method
        Interpolation method to use ('linear', 'nearest', etc.).

    Returns
    -------
    :
        Callable function representing the vector field.

    """
    grid = vector_field_dict["grid"]  # tuple of 3D arrays (xgrid, ygrid, zgrid)

    # Extract 1D axes from meshgrid
    x = np.unique(grid[0])
    y = np.unique(grid[1])
    z = np.unique(grid[2])
    axes = (x, y, z)

    # Stack vector components into shape (nx, ny, nz, 3)
    vec_field_grid = np.stack(vector_field_dict["vectors"], axis=-1)
    # Create interpolators for each component
    # fill_value set to None for extrapolation
    interpolators = [
        RegularGridInterpolator(
            axes, vec_field_grid[..., i], method=method, bounds_error=False, fill_value=None
        )
        for i in range(3)
    ]

    def vf_general(point: np.ndarray) -> np.ndarray:
        # point: shape (3,) or (N, 3)
        point = np.atleast_2d(point)
        return np.stack([interp(point) for interp in interpolators], axis=-1).squeeze()

    def vf_solve_ivp(t: float, y: np.ndarray) -> np.ndarray:
        # y: shape (3,)
        return vf_general(y)

    return vf_solve_ivp if for_solve_ivp else vf_general


def solve_ddff_ode(
    flow_field_dict: dict,
    init: np.ndarray,
    t_span: tuple[float, float],
    num_t: int = 1750,
    time_limit: float | None = None,
) -> np.ndarray:
    """Solve an autonomous ODE using ``scipy.integrate.solve_ivp``.

    **Method inputs**

    The input ``flow_field_dict`` is a dictionary with two keys:

    - "vectors": tuple of 3D arrays (f1,f2,f3) with the vector values in each dimension
    - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each dimension

    The supplied initial condition ``init`` should be a numpy array of shape (3,).

    The time span ``t_span`` should be a list of two floats specifying the start and end times
    for the ODE solver. The input ``num_t`` specifies the number of time points to evaluate
    the solution at between the start and end times.

    **Method output**

    The output is the solution of the ODE with the given initial condition, which is a numpy
    array of shape (num_t, 3).

    Parameters
    ----------
    flow_field_dict
        Dictionary with arrays defining a vector field evaluated on a mesh grid.
    init
        Initial condition for the trajectory.
    t_span
        Time span for the ODE solver as (t0, tf).
    num_t
        Number of time points to evaluate the solution at.
    time_limit
        Maximum allowed time in seconds for the ODE solver to run. If the elapsed time
        exceeds this limit, the integration will be stopped. If None, no time limit
        is enforced.

    Returns
    -------
    :
        Solution trajectory in 3D state space for the given initial condition and time span.

    """

    if time_limit is None:
        time_limit = np.inf

    start_time = perf_counter()

    def time_limit_exceeded(t, y):
        # check if the elapsed time has exceeded the time limit
        if perf_counter() - start_time > time_limit:
            return 0  # solve_ivp "events" arg will trigger on a 0 crossing
        return 1  # integration will continue if not 0 and does not cross 0

    time_limit_exceeded.terminal = True  # stop the solver when this event is triggered

    # turn flow field into callable function (works via interpolation)
    my_flow = get_callable_vector_field(flow_field_dict, for_solve_ivp=True)
    # timepoints at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], num_t)
    # solve the IVP
    sol = solve_ivp(my_flow, t_span, init, t_eval=t_eval, events=time_limit_exceeded)

    if sol.status == 1:
        logger.warning(
            "Time limit exceeded during ODE integration after %.2f seconds",
            perf_counter() - start_time,
        )
        return np.full(shape=(num_t, init.shape[0]), fill_value=np.nan)
    elif sol.status < 0:
        logger.error("ODE solver failed with status %d", sol.status)
        return np.full(shape=(num_t, init.shape[0]), fill_value=np.nan)

    return sol.y.T  # get trajectory, shape (num_T, 3) (3D trajectory in state space)
