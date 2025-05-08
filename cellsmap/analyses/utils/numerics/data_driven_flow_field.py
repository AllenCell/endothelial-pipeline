from collections.abc import Callable

import numpy as np
from scipy import interpolate as spinterp
from scipy.integrate import solve_ivp


def set_3d_bounds_from_data(
    x: np.array, y: np.array, z: np.array, excluded_fraction: float = 0.1
) -> list[np.ndarray]:
    """
    Set the bounds for the 3D flow field based on the data
    (leaving out a fraction of the data
    based on the excluded_fraction parameter).

    Inputs:
    - x, y, z: 1D arrays of the data points in the three dimensions
    - excluded_fraction: fraction of data to exclude from
        the bounds (default is 0.1)

    Outputs:
    - bounds: list of 3D arrays with the upper and
        lower bounds for each dimension
    """
    bounds = []
    for var in [x, y, z]:
        bounds.append(np.percentile(var, [excluded_fraction, 100 - excluded_fraction]))
    return bounds


def compute_extrapolated_vector_field(
    kmcs: np.ndarray, grid_centers: list[np.ndarray], interpolator: str = "nearest"
) -> dict:
    """
    Get an extrapolated 3D vector field via estimates of
    that vector field (i.e., first or second Kramers-Moyal
    coefficients: drift or diffusion) from the data
    for a given experimental condition.

    Inputs:
    - kmcs: 3D array of drift or diffusion estimates
        - shape num_bins_x, num_bins_y, num_bins_z, 3)
        - Computed via kernel-based method in
        `cellsmap.analyses.utils.regression_helper.get_kramers_moyal`
    - grid_centers: 1D numpy arrays with the grid points in
        each dimension (centers of bins of x,y,z space)
    - interpolator (optional, default="nearest"): interpolation method
        by which to infer flow field at points where data is
        scarce (drift_kmcs = np.nan)
        - options are "nearest" (nearest neighbors)
            and "linear" (linear interpolation)

    Outputs:
    - vector_field_dict: dictionary with the following keys:
        - "vectors": tuple of 3D arrays (F1,F2,F3) with
            the vector values in each dimension
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with
            the grid points in each dimension
    """

    ndim = len(grid_centers)  # number of dimensions

    # generate a mesh grid of points in the state space
    grid = np.meshgrid(*grid_centers, indexing="ij")  # make meshgrid

    # get the vector field components from
    # the Kramers-Moyal coefficients
    vector_field = [kmcs[..., i] for i in range(ndim)]

    # where KMCs have been masked to nan, extrapolate
    # via nearest neighbors.
    # use spinterp.interpn with method='nearest'
    # Find the indices of valid (non-NaN) points
    valid_mask = ~np.isnan(vector_field[0])
    # Get the coordinates of valid points
    valid_points = np.array(np.nonzero(valid_mask)).T
    # Get the values of valid points
    valid_values = [vector_field[i][valid_mask] for i in range(ndim)]

    # Create interpolators for each component of the vector field
    if interpolator == "nearest":  # nearest neighbor
        interpolator_func = [
            spinterp.NearestNDInterpolator(valid_points, valid_values[i])
            for i in range(ndim)
        ]
    elif interpolator == "linear":  # linear interpolation
        interpolator_func = [
            spinterp.LinearNDInterpolator(valid_points, valid_values[i])
            for i in range(ndim)
        ]
    else:
        raise ValueError(
            f"Interpolator {interpolator} not recognized. Use 'nearest' or 'linear'."
        )

    # Find the indices of all points (including NaN points)
    all_points = (
        np.array(np.indices(vector_field[0].shape))
        .reshape(len(vector_field[0].shape), -1)
        .T
    )

    # Interpolate the NaN points
    vec_interpolated = [interpolator_func[i](all_points) for i in range(ndim)]

    # reshape back to original grid shape
    vec_interpolated = [vec.reshape(vector_field[0].shape) for vec in vec_interpolated]

    # Create a dictionary to store the vector field and grid
    vector_field_dict = {"vectors": vec_interpolated, "grid": grid}

    return vector_field_dict


def get_callable_vector_field(
    vector_field_dict: dict, for_solve_ivp: bool = True
) -> Callable:
    """
    Get a callable vector field via linear interpolation
    on computed values of the vector field on the grid.

    Inputs:
    - vector_field_dict: dictionary with the following keys:
        - "vectors": tuple of 3D arrays (V1,V2,V3) with the
            vector values in each dimension
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)
            with the grid points in each dimension
    - for_solve_ivp (optional, default=True): whether to
        return a callable function that takes in time and
        point in state space and returns the flow field at
        that point (for use with scipy.integrate.solve_ivp)
        - if False, returns a callable function that takes
        in a point in state space and returns the flow field
        at that point (for general use)

    Outputs:
    - vec_func: callable function that takes in a time and a
        point in state space and returns the flow field at that point
        - value of the vector field at the given point is interpolated from
        the given values of V1,V2,V3 on the fixed input grid
        (on which the flow field was numerically estimated from the data)
    """

    ndim = len(vector_field_dict["grid"])  # number of dimensions

    # get the interpolator for f_KM
    vec_field_grid = np.stack(
        vector_field_dict["vectors"], axis=-1
    )  # shape (num_bins_x, num_bins_y, ... , ndim)
    xyz_grid = np.moveaxis(np.array(vector_field_dict["grid"]), 0, -1).reshape(
        (-1, ndim)
    )
    vec_field_interp = spinterp.LinearNDInterpolator(
        xyz_grid, vec_field_grid.reshape((-1, ndim))
    )  # interpolator for f_KM

    if for_solve_ivp:
        # define a callable function to pass
        # into the ODE solver
        # for scipy.integrate.solve_ivp,
        # need time as first argument
        # and x as second argument even if
        # the system is time-independent
        def vec_func(t, x):
            # get interpolated value
            vec_interp_val = vec_field_interp(x)
            # return dx/dt = f(x)
            return vec_interp_val

    else:

        def vec_func(x):
            # get interpolated value
            vec_interp_val = vec_field_interp(x)
            # return dx/dt = f(x)
            return vec_interp_val

    return vec_func


def solve_ddff_ode(
    flow_field_dict: dict, init: np.ndarray, t_span: list[float], num_t: int = 1750
) -> np.ndarray:
    """
    Solve the ODE dx/dt = f(x) using scipy.integrate.solve_ivp.

    Inputs:
    - flow_field_dict: dictionary with the following keys:
        - "vectors": tuple of 3D arrays (1, f2, f3) with the
            velocities in each dimension
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the
            grid points in each dimension
    - init: initial condition for the trajectory (shape (3,))
    - t_span: time span for the ODE solver (list of two floats)
    - num_T: number of time points to evaluate the solution
        (default is 1750)

    Outputs:
    - sol: solution of the ODE with the given
        initial condition (shape (num_t, 3))

    """
    # turn flow field into callable function (works via interpolation)
    my_flow = get_callable_vector_field(flow_field_dict)
    # timepoints at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], num_t)
    # solve the IVP
    sol = solve_ivp(my_flow, t_span, init, t_eval=t_eval)
    return sol.y.T  # get trajectory, shape (num_T, 3) (3D trajectory in state space)


def interpolate_on_curve(traj: np.ndarray, n_points: int = 5) -> np.ndarray:
    """
    Interpolate a trajectory in ND space to get
    n_points evenly spaced points along the trajectory.

    Inputs:
    - traj: trajectory in ND space
        - shape (num_t, num_dimensions)
    - n_points: number of points to interpolate to (default is 5)

    Outputs:
    - interpolated_points: interpolated points along the trajectory
        - shape (n_points, num_dimensions),
        - equally spaced by arc length
    """
    ndim = traj.shape[1]  # number of dimensions

    # compute cumulative distance of
    # each point from the first point
    # along the trajectory
    distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc_length = np.cumsum(np.concatenate(([0], distances)))

    # interpolate to by these distances to
    #  get n_points evenly spaced points
    arc_length_new = np.linspace(0, arc_length[-1], n_points)

    # initialize array interpolated points
    interpolated_points = np.zeros((n_points, 3))
    for i in range(ndim):  # loop over dimensions
        interpolated_points[:, i] = np.interp(arc_length_new, arc_length, traj[:, i])

    return interpolated_points


def convert_coordinates_from_volume_to_pc(
    xvol: np.array, grid_spacing: float, origin: float
) -> np.array:
    """
    Convert coordinates from 3D volume space to 3D PC space
    (for saving as .vtk to view in ParaView).
    """
    xpc = origin + xvol * grid_spacing
    return xpc
