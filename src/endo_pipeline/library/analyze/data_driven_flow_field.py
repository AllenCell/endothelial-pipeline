import logging
from collections.abc import Callable
from time import time
from typing import Literal, overload

import numpy as np
import pandas as pd
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.stats import gaussian_kde

from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.numerics.binning import circpercentile
from endo_pipeline.library.visualize.diffae_features.pplane import (
    get_fpt_type,
    get_fpts,
    get_stability_label_from_fpt_type,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import BIN_LIMITS_THETA_RESCALED
from endo_pipeline.settings.flow_field_3d import SAMPLER_RANDOM_SEED
from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLUMN_NAME

logger = logging.getLogger(__name__)


def sample_from_density(
    data: np.ndarray, n_samples: int, random_seed: int = SAMPLER_RANDOM_SEED
) -> np.ndarray:
    """
    Sample points from the density of a given dataset using KDE and rejection sampling.

    Parameters
    ----------
    data
        Input data of shape (N, D).
    n_samples
        Number of samples to draw.
    random_seed
        Random seed for reproducibility.

    Returns
    -------
    :
        Sampled points of shape (n_samples, D).
    """
    rng = np.random.default_rng(seed=random_seed)
    kde = gaussian_kde(data.T)
    n_dims = data.shape[1]
    samples: list[np.ndarray] = []
    # Estimate bounds for rejection sampling
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    # Estimate maximum density for rejection
    test_points = rng.uniform(mins, maxs, size=(10000, n_dims))
    max_density = kde(test_points.T).max()
    while len(samples) < n_samples:
        candidate = rng.uniform(mins, maxs)
        density = kde(candidate)
        if rng.uniform(0, max_density) < density:
            samples.append(candidate)
    return np.array(samples)


def _compute_percentile_values(
    data: pd.DataFrame,
    column_names: list[str],
    q: float,
    polar_angle_range: tuple[float, float] = BIN_LIMITS_THETA_RESCALED,
) -> dict[str, float]:
    """
    Compute the lower and upper percentile bounds for each column in the data.

    Parameters
    ----------
    data
        DataFrame containing the data.
    column_names
        List of column names to compute percentiles for.
    q
        Percentile to compute (e.g. 2 for the 2nd percentile).
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when computing percentiles for circular variables.

    Returns
    -------
    :
        Dictionary mapping column names to their percentile values.
    """
    percentile_values: dict[str, float] = {}
    for column_name in column_names:
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            percentile_value = circpercentile(data[column_name], q=q, polar_range=polar_angle_range)
        else:
            percentile_value = np.percentile(data[column_name], q=q)
        percentile_values[column_name] = percentile_value
    return percentile_values


def is_point_within_percentile_bounds(
    point: np.ndarray | tuple[float, ...],
    column_names: list[str],
    lower_percentile_bounds: dict[str, float],
    upper_percentile_bounds: dict[str, float],
    polar_angle_range: tuple[float, float] = BIN_LIMITS_THETA_RESCALED,
):
    """
    Check if a point is within a specified percentile range along each dimension
    of a dataset, accounting for circular variables.

    **Percentile bound specification**

    The inputs lower_percentile_bounds and upper_percentile_bounds should be
    lists of floats specifying the lower and upper percentiles of the data as
    computed by, e.g., numpy.percentile or the circpercentile function for
    circular variables. That is to say, lower_percentile_bounds[i] should be the
    value of the lower percentile for the data in column column_names[i], not
    the specified percentile (e.g. 2) itself.

    **Handling circular variables**

    For circular variables (e.g. angles), the function checks if the point is
    within the bounds accounting for wraparound. For example, if the lower
    percentile bound is 350 degrees and the upper percentile bound is 10
    degrees, then a point at 355 degrees would be considered within bounds,
    while a point at 20 degrees would not be.

    Furthermore, we do not want to return multiple equivalent points that are
    separated by the wraparound boundary for circular variables. Thus, we also
    specify the polar angle range (e.g. [0, 360] or [-pi, pi]) to ensure that
    the point is only considered within bounds if it is within the bounds in the
    specified polar angle range. For example, if the polar angle range is [0,
    360], then a point at -5 degrees would not be considered "within bounds"
    even if the lower percentile bound is 350 and the upper percentile bound is
    10, degrees.

    Parameters
    ----------
    point
        The point to check.
    column_names
        List of column names corresponding to the dimensions of the point and
        data.
    lower_percentile_bounds
        Dictionary mapping column names to pre-computed lower percentile bounds.
    upper_percentile_bounds
        Dictionary mapping column names to pre-computed upper percentile bounds.
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when checking if the point is within bounds for
        circular variables.

    Returns
    -------
    :
        True if point is within the percentile bounds on all axes, else False.
    """
    if len(point) != len(column_names):
        raise ValueError(
            f"Length of point ({len(point)}) does not match number of column names ({len(column_names)})."
        )

    is_within_bounds = []
    for point_component, column_name in zip(point, column_names, strict=True):
        lower_bound = lower_percentile_bounds[column_name]
        upper_bound = upper_percentile_bounds[column_name]
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            # for circular variables, need to account for bounds wrapping around
            if lower_bound <= upper_bound:
                is_within_bounds.append(
                    (lower_bound <= point_component) & (point_component <= upper_bound)
                )
            else:
                # check if point is within bounds accounting for wraparound
                # and given polar range (e.g. [0, 2pi] or [-pi, pi])
                is_within_bounds.append(
                    (polar_angle_range[1] >= point_component >= lower_bound)
                    | (polar_angle_range[0] <= point_component <= upper_bound)
                )
        else:
            is_within_bounds.append(
                (lower_bound <= point_component) & (point_component <= upper_bound)
            )
    return np.all(is_within_bounds)


def get_fixed_points_within_bounds(
    vector_field_function: Callable[[np.ndarray], np.ndarray],
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    num_inits_for_root_solver: int,
    lower_percentile: float,
    upper_percentile: float,
    polar_angle_range: tuple[float, float],
    stability_label_column_name: str = STABILITY_COLUMN_NAME,
) -> pd.DataFrame:
    """
    Get fixed points of a given estimated vector field with high confidence.

    For a single dataset, this workflow:

    1. Finds fixed points of the vector field by finding roots of the input
       function using multiple initial conditions sampled from the density of
       the given data.
    2. Filters the fixed points to only keep those that are within a specified
       percentile range of the data along each dimension.

    Parameters
    ----------
    vector_field_function
        Callable function that takes in a point in 3D space and outputs a 3D
        vector at that point.
    dataframe
        Dataframe containing the feature data for the dataset, which is used to
        filter the fixed points to only keep those within a certain percentile
        range of the data.
    column_names
        List of column names corresponding to the features used in the analysis,
        in the same order as the columns in feature_data.
    num_inits_for_root_solver
        Number of initial conditions to use for finding fixed points.
    lower_percentile
        Lower percentile for filtering fixed points.
    upper_percentile
        Upper percentile for filtering fixed points.
    polar_angle_range
        The range of the polar angle variable for handling wraparound when
        computing percentiles for circular variables.
    stability_label_column_name
        Column name to use for fixed point stability classification labels in the
        output dataframe.

    Returns
    -------
    :
        Dataframe containing of stable fixed points with high confidence (i.e.,
        points filtered by percentile range).
    """
    check_required_columns_in_dataframe(
        dataframe, [*column_names, Column.DATASET]
    )  # check required columns are in dataframe
    feature_data = dataframe[column_names].to_numpy()  # get feature data as numpy array
    dataset_name = dataframe[Column.DATASET].iloc[0]  # get dataset name from dataframe

    # create Jacobian function for finding stability of fixed points
    vector_field_jacobian = Jacobian(vector_field_function)

    # sample initial conditions for root solver from data density
    sampled_inits_for_root_solver = sample_from_density(feature_data, num_inits_for_root_solver)

    # pass into helper function to get fixed points
    fpts = get_fpts(vector_field_function, sampled_inits_for_root_solver)

    # filter fixed points to only keep ones within a given range of percentiles
    # of data (e.g., 2 to 98) to get high confidence fixed points that are
    # within the region of state space supported by the data
    lower_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=lower_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Lower percentile bounds for filtering fixed points: [ %s ]", lower_percentile_bounds
    )
    upper_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=upper_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Upper percentile bounds for filtering fixed points: [ %s ]", upper_percentile_bounds
    )
    fpts_high_confidence_list = []
    for fpt in fpts:
        within_percentile = is_point_within_percentile_bounds(
            fpt, column_names, lower_percentile_bounds, upper_percentile_bounds, polar_angle_range
        )
        if within_percentile:
            # get stability/type of the fixed point
            fpt_type = get_fpt_type(vector_field_jacobian(fpt))
            logger.debug("[ %s ] at [ (%.2f, %.2f, %.2f) ]", fpt_type, fpt[0], fpt[1], fpt[2])
            fpt_stability_label = get_stability_label_from_fpt_type(fpt_type)
            fpts_high_confidence_list.append(
                pd.DataFrame(
                    {
                        Column.DATASET: [dataset_name],
                        stability_label_column_name: [fpt_stability_label],
                        column_names[0]: [fpt[0]],
                        column_names[1]: [fpt[1]],
                        column_names[2]: [fpt[2]],
                    }
                )
            )

    # check if any fixed points with high confidence were found, and if not, log
    # a warning and return an empty dataframe with the correct columns
    if len(fpts_high_confidence_list) == 0:
        logger.warning(
            "No fixed points with high confidence found for dataset [ %s ]."
            "Consider adjusting percentile thresholds or number of initial conditions for root solver.",
            dataset_name,
        )
        return pd.DataFrame(columns=[Column.DATASET, stability_label_column_name, *column_names])

    # else, concatenate the list of dataframes for each fixed point into a
    # single dataframe and return it
    fpts_high_confidence = pd.concat(fpts_high_confidence_list, ignore_index=True)
    return fpts_high_confidence


def fill_nan_for_vtk(data: np.ndarray, method: str = "nearest") -> np.ndarray:
    """
    Replaces NaN values in a multi-dimensional NumPy array using scipy.interpolate.griddata.

    Parameters
    ----------
    data
        Input NumPy array with NaN values to be imputed.

    Returns
    -------
    :
        A copy of the input array with NaNs imputed.
    """
    # Create a copy to avoid modifying the original data
    arr = data.copy()

    # Identify NaN locations
    nan_mask = np.isnan(arr)

    # If there are no NaNs, return the original array
    if not np.any(nan_mask):
        return arr

    # Else, proceed with interpolation
    # Get a mask for valid (non-NaN) points
    valid_mask = ~nan_mask

    # Get coordinates and values for interpolation source
    valid_coords = np.array(np.where(valid_mask)).T
    valid_values = arr[valid_mask]

    # Get coordinates for interpolation query (NaN points)
    nan_coords = np.array(np.where(nan_mask)).T

    # Use griddata to perform extrapolation
    imputed_values = griddata(
        points=valid_coords, values=valid_values, xi=nan_coords, method=method
    )

    # Assign the extrapolated values back into the NaN locations
    arr[nan_mask] = imputed_values

    return arr


def compute_extrapolated_vector_field(
    kmcs: np.ndarray,
    grid_coordinates: list[np.ndarray],
    method: str = "linear",
    for_vtk_files: bool = False,
) -> dict:
    """
    Extrapolate a 3D vector field from Kramers-Moyal estimates over a specified grid.

    **Method inputs**

    The input ``kmcs`` are the Kramers-Moyal coefficients (drift or diffusion estimates)
    over a 3D mesh grid obtained from feature data. Where there are no data points, these
    estimates are `NaN`. This function extrapolates these estimates to the entire grid
    using nearest-neighbor or linear interpolation.

    The array ``kmcs`` should have shape (num_x, num_y, num_z, 3), where
    num_x, num_y, and num_z are the number of points in each dimension
    of the 3D meshgrid defined by ``grid_coordinates``.

    **Method output**

    The output is a dictionary with two keys:
    - "vectors": tuple of 3D arrays (f1,f2,f3) with the vector values in each dimension
    - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the meshgrid points in each dimension

    **Extrapolation for vtk files vs other uses**

    If ``for_vtk_files`` is `True`, the extrapolation is done using a wrapper method for
    filling NaNs via `scipy.interpolate.griddata`, which ensures no NaNs remain in the output.
    This is important for saving the vector field as .vtk files for visualization in ParaView,
    but is slower for large grids.

    If ``for_vtk_files`` is `False`, the extrapolation is done using `scipy.interpolate.RegularGridInterpolator`,
    which is faster for large grids, but can produce NaNs outside the convex hull of the data points unless
    they are filled in with a numerical value (here, zero) first when creating the interpolator. This will result
    in NaNs being converted to zeros outside the convex hull, which is suitable for all other uses of the vector field
    except vtk files.

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
    """

    filled_kmcs = kmcs.copy()
    n_components = filled_kmcs.shape[-1]
    x, y, z = np.meshgrid(*grid_coordinates, indexing="ij")

    for i in range(n_components):
        component = filled_kmcs[..., i]
        nan_mask = np.isnan(component)
        if np.any(nan_mask):
            if for_vtk_files:
                if method != "nearest":
                    logger.warning(
                        "Using extrapolation method other than 'nearest' for vtk file construction results in "
                        "significant memory usage and computation time. Consider using 'nearest' method. "
                    )
                logger.debug("Starting extrapolation for vtk files.")
                tic = time()
                component = fill_nan_for_vtk(component, method=method)
                toc = time()
                logger.debug(f"Finished extrapolation for vtk files in {toc - tic:.2f} seconds.")
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
    """
    Get a callable vector field from a numpy array via linear interpolation.

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
) -> np.ndarray:
    """
    Solve an autonomous ODE using ``scipy.integrate.solve_ivp``.

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

    Returns
    -------
    :
        Solution trajectory in 3D state space for the given initial condition and time span.
    """
    # turn flow field into callable function (works via interpolation)
    my_flow = get_callable_vector_field(flow_field_dict, for_solve_ivp=True)
    # timepoints at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], num_t)
    # solve the IVP
    sol = solve_ivp(my_flow, t_span, init, t_eval=t_eval)
    return sol.y.T  # get trajectory, shape (num_T, 3) (3D trajectory in state space)


def interpolate_on_curve(traj: np.ndarray, n_points: int = 5) -> np.ndarray:
    """
    Obtain points along a curve equally spaced by arc length.

    Parameters
    ----------
    traj
        Curve in n-dimensional space, shape (num_t, num_dimensions).
    n_points
        Number of equally spaced points to interpolate along the curve.

    Returns
    -------
    :
        Interpolated points along the curve, shape (n_points, num_dimensions).
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


def get_drift_flow_field_as_dict(
    flow_field_dataframe: pd.DataFrame, column_names: list[str | Column.DiffAEData]
) -> dict[str, tuple[np.ndarray]]:
    """Convert a drift flow field dataframe into a dictionary suitable for visualization / analysis.

    Parameters
    ----------
    flow_field_dataframe
        Dataframe containing the flow field data with columns corresponding to the coordinates and drift values.
    column_names
        List of column names corresponding to the dynamics features to use for constructing the flow field.

    Returns
    -------
    dict[str, tuple[np.ndarray]]
        Dictionary containing the flow field vectors and the corresponding grid points.
    """
    # restructure the flow field dataframe into a flow field dictionary
    ndim = len(column_names)

    grid_points_1d = [np.sort(flow_field_dataframe[col].unique()) for col in column_names]
    grid_shape = tuple(len(points) for points in grid_points_1d)
    grid = np.meshgrid(*grid_points_1d, indexing="ij")

    # unpack drift values from dataframe and reshape to grid shape for flow
    # field visualization and ODE solving
    flow_field_column_names = [f"{name}_drift" for name in column_names]
    drift_values = (
        flow_field_dataframe[flow_field_column_names].to_numpy().reshape(*grid_shape, ndim)
    )

    # build flow field dict for downstream functions that expect the flow
    # field in this format
    drift_vector_field = tuple(drift_values[..., i] for i in range(ndim))
    flow_field_dict = {"vectors": tuple(drift_vector_field), "grid": grid}

    return flow_field_dict
