"""Methods related to flow field estimation and analysis."""

import logging
from collections.abc import Callable
from time import perf_counter, time
from typing import Literal, overload

import numpy as np
import pandas as pd
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.stats import gaussian_kde

from endo_pipeline.io.input import load_dataframe
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
    get_kramers_moyal_coeffs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import circpercentile, get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.visualize.diffae_features.pplane import (
    get_fpt_type,
    get_fpts,
    get_stability_label_from_fpt_type,
)
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest_utils import get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_THETA_RESCALED,
    HISTOGRAM_THRESHOLD_FOR_MASKING,
)
from endo_pipeline.settings.flow_field_3d import (
    LOWER_PERCENTILE_FOR_STABLE_FP,
    NUM_INIT_SAMPLES,
    PAD_BINS_FLOAT,
    SAMPLER_RANDOM_SEED,
    UPPER_PERCENTILE_FOR_STABLE_FP,
)
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_DRIFT,
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    STABILITY_COLUMN_NAME,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)


def sample_from_density(
    data: np.ndarray, n_samples: int, random_seed: int = SAMPLER_RANDOM_SEED
) -> np.ndarray:
    """Sample points from the density of a given dataset using KDE and rejection sampling.

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
    """Compute the lower and upper percentile bounds for each column in the data.

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
    """Check if a point is within a specified percentile range in each variable.

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
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """Get fixed points of a given estimated vector field with high confidence.

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
    metadata_dict
        Optional dictionary of metadata to include as columns in the output dataframe.

    Returns
    -------
    :
        Dataframe containing of stable fixed points with high confidence (i.e.,
        points filtered by percentile range).

    """
    check_required_columns_in_dataframe(dataframe, column_names)
    feature_data = dataframe[column_names].to_numpy()

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
    upper_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=upper_percentile, polar_angle_range=polar_angle_range
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
            "No fixed points with high confidence found. Consider adjusting percentile"
            " thresholds or number of initial conditions for root solver."
        )
        fpts_high_confidence = pd.DataFrame(columns=[stability_label_column_name, *column_names])
    # else, concatenate the list of dataframes for each fixed point into a
    # single dataframe and return it
    else:
        fpts_high_confidence = pd.concat(fpts_high_confidence_list, ignore_index=True)

    # add provided metadata columns to the dataframe (e.g. dataset name, shear stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            fpts_high_confidence[key] = metadata_dict[key]

    return fpts_high_confidence


def fill_nan_for_vtk(data: np.ndarray, method: str = "nearest") -> np.ndarray:
    """Replace NaN values in a multi-dimensional `numpy` array using `scipy.interpolate.griddata`.

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
    """Extrapolate a 3D vector field from Kramers-Moyal estimates over a specified grid.

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


def interpolate_on_curve(traj: np.ndarray, n_points: int = 5) -> np.ndarray:
    """Obtain points along a curve equally spaced by arc length.

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


def compute_drift_vector_field(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    time_step: float,
) -> np.ndarray:
    """
    Compute the drift coefficient vector field along specified features for a
    single flow condition.

    **Kernel specification**

    The input ``kernel`` can be a single kernel function that is applied to all
    dimensions, or a list of kernel functions for each dimension (in which case
    the product kernel is used).

    In general, the kernel is specified as a ``KramersMoyalKernel`` dataclass,
    which has attributes for the kernel name, bandwidth, and period (if
    applicable). If a list of kernels is provided, each kernel in the list
    should be a ``KramersMoyalKernel`` dataclass corresponding to each
    dimension.

    Parameters
    ----------
    dataframe
        Dataframe containing the feature data (time series trajectories) for a
        single flow condition.
    column_names
        Feature column names to use for computing the drift coefficients.
    bins
        List of arrays specifying the bin edges for each dimension to use for
        estimating the drift coefficients.
    kernels
        Kramers-Moyal kernel or list of Kramers-Moyal kernels in each dimension
        to use for estimating the drift coefficients.

    Returns
    -------
    :
        Array containing the drift coefficients for each point in the input
        dataframe.

    """

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(dataframe, column_names)

    # get drift estimates in units hours^-1 for each bin in 3D space
    # (Kramers-Moyal coefficient estimation)
    drift_coeffs = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=time_step, kernel=kernel
    )[0]

    return drift_coeffs


def mask_drift_coeffs_by_data_density(
    drift_coeffs: np.ndarray,
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    histogram_bins: list[np.ndarray],
    histogram_kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    probability_threshold: float = HISTOGRAM_THRESHOLD_FOR_MASKING,
) -> np.ndarray:
    """
    Mask drift coefficients in regions of low data density.

    This method uses a kernel density estimate of the data density based on a
    histogram of the data in the specified feature space. The drift coefficients
    are set to NaN in regions where the estimated data density is below the
    specified threshold.

    Parameters
    ----------
    drift_coeffs
        Array containing the drift coefficients for each point in the feature
        grid.
    dataframe
        Dataframe containing the feature data for a single dataset and flow
        condition.
    column_names
        Feature column names corresponding to the dimensions of the feature
        space.
    histogram_bins
        List of arrays specifying the bin edges for each dimension to use for
        estimating the data density.
    histogram_kernel
        Kramers-Moyal kernel or list of Kramers-Moyal kernels in each dimension
        to use for estimating the data density.
    probability_threshold
        Threshold for the estimated data density below which to set drift
        coefficients to NaN.

    Returns
    -------
    :
        Array of the same shape as the input drift_coeffs, but with coefficients
        set to NaN in regions where the estimated data density is below the
        specified threshold.

    """
    hist = np.histogramdd(dataframe[column_names].to_numpy(), bins=histogram_bins)
    hist_kde = get_kernel_density_estimate_from_histogram(
        hist[None, ...],
        bins=histogram_bins,
        kernel=histogram_kernel,
    )
    low_probability_mask = hist_kde < probability_threshold
    drift_coeffs[low_probability_mask] = np.nan

    return drift_coeffs


def create_drift_vector_field_df(
    drift_coeffs: np.ndarray,
    column_names: list[str | Column.DiffAEData],
    feature_grid: tuple[np.ndarray],
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """
    Create dataframe containing the estimated drift vector field for a single
    flow condition.

    The output dataframe will have columns for the grid points in each of the
    three dimensions, the corresponding drift coefficients, and additional
    metadata such as dataset name and shear stress.

    Parameters
    ----------
    drift_coeffs
        Array containing the drift coefficients for each point in the feature
        grid.
    column_names
        List of column names corresponding to the features used for computing
        the drift coefficients.
    feature_grid
        Tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each
        dimension corresponding to the drift coefficients.
    metadata_dict
        Optional, dictionary containing metadata to include in the output dataframe (e.g.
        dataset name, shear stress
    """

    # build dataframe with columns for grid points in each of the three
    # dimensions and the corresponding drift coefficients
    drift_column_names: list[str] = [f"{name}_drift" for name in column_names]
    vector_field_df = pd.DataFrame(columns=[Column.DATASET, *drift_column_names, *column_names])

    # make tuple for indexing the drift coefficients and feature grid
    index_tuple = tuple(range(len(column_names)))
    for index, column_name, drift_column_name in zip(
        index_tuple, column_names, drift_column_names, strict=True
    ):
        vector_field_df[column_name] = feature_grid[index].flatten()
        vector_field_df[drift_column_name] = drift_coeffs[..., index].flatten()

    # add specified metadata columns to the dataframe (e.g. dataset name, shear
    # stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            vector_field_df[key] = metadata_dict[key]

    return vector_field_df


def get_drift_estimates_and_fixed_points(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bin_widths: list[float],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    time_step: float,
    metadata_dict: dict[str, str | float] | None = None,
    pad_bins_float: float = PAD_BINS_FLOAT,
    polar_angle_range: tuple[float, float] = BIN_LIMITS_THETA_RESCALED,
    num_inits_for_root_solver: int = NUM_INIT_SAMPLES,
    lower_percentile_for_stable_fp: float = LOWER_PERCENTILE_FOR_STABLE_FP,
    upper_percentile_for_stable_fp: float = UPPER_PERCENTILE_FOR_STABLE_FP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get bins for flow field estimation based on the trajectories, to be
    # used for kernel-convolution-based estimation of the Kramers-Moyal
    # coefficients. The bins are determined by the specified bin widths and
    # the range of the data.
    bins, centers = get_bins(
        bin_widths,
        data=dataframe[column_names].to_numpy(),
        pad=pad_bins_float,
    )
    feature_grid = np.meshgrid(*centers, indexing="ij")

    # estimate the drift coefficients at each bin/grid point in 3D space
    # using a kernel-convolution-based method for estimating
    # Kramers-Moyal coefficients from time series data.
    drift_coeffs = compute_drift_vector_field(
        dataframe,
        column_names,
        bins=bins,
        kernel=kernel,
        time_step=time_step,
    )

    # Compile estimated drift coefficients and corresponding grid points
    # into a dataframe for this dataset, to be saved and tracked.
    vector_field_df = create_drift_vector_field_df(
        drift_coeffs=drift_coeffs,
        column_names=column_names,
        feature_grid=feature_grid,
        metadata_dict=metadata_dict,
    )

    # Extrapolate the drift to get a flow field over the entire 3D space
    # as specified by the input bins and centers, and use it to get a
    # callable function for the flow field that can be used for root
    # finding to identify fixed points.
    extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
        drift_coeffs, centers, method="linear", for_vtk_files=False
    )
    drift_function = get_callable_vector_field(
        extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
    )
    fixed_points_dataframe = get_fixed_points_within_bounds(
        vector_field_function=drift_function,
        dataframe=dataframe,
        column_names=column_names,
        polar_angle_range=polar_angle_range,
        num_inits_for_root_solver=num_inits_for_root_solver,
        lower_percentile=lower_percentile_for_stable_fp,
        upper_percentile=upper_percentile_for_stable_fp,
        metadata_dict=metadata_dict,
    )

    return vector_field_df, fixed_points_dataframe


def get_drift_df(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """Get the drift dataframe of a data-driven flow field for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset to get the drift dataframe for.
    model_manifest_name
        Name of the model manifest to use for locating the drift dataframe.
    run_name
        Name of the model run to use for locating the drift dataframe.

    Returns
    -------
    pd.DataFrame
        Drift dataframe for the given dataset.
    """

    base_name = f"{model_manifest_name}_{run_name}_grid"
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{base_name}"
    drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)

    if dataset_name not in drift_dataframe_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in drift dataframe manifest [ %s ]!",
            dataset_name,
            drift_dataframe_manifest_name,
        )
        return pd.DataFrame()

    logger.info("Getting drift dataframe for grid-based crops...")

    drift_dataframe_location = get_dataframe_location_for_dataset(
        drift_dataframe_manifest, dataset_name
    )
    drift_df = load_dataframe(drift_dataframe_location, delay=False)

    return drift_df


def get_drift_values_and_grid_from_drift_df(
    flow_field_dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the reshaped drift values and the corresponding grid points from a flow field dataframe.

    Parameters
    ----------
    flow_field_dataframe
        Dataframe containing the flow field data with columns corresponding to the coordinates and drift values.
    column_names
        List of column names corresponding to the dynamics features to use for constructing the flow field.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the drift values reshaped to the grid shape and the 1D grid points for each dimension.
    """

    # restructure the drift dataframe into a flow field dictionary
    ndim = len(column_names)
    drift_column_names = [f"{name}_drift" for name in column_names]

    grid_points_1d = [
        np.sort(flow_field_dataframe[column_name].unique()) for column_name in column_names
    ]
    grid_shape = tuple(len(points) for points in grid_points_1d)

    # unpack drift values from dataframe and reshape to grid shape for flow
    # field visualization and ODE solving
    drift_values = flow_field_dataframe[drift_column_names].to_numpy().reshape(*grid_shape, ndim)

    return drift_values, grid_points_1d


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
    drift_values, grid_points_1d = get_drift_values_and_grid_from_drift_df(
        flow_field_dataframe, column_names
    )

    # reshape the 1D grid points into a an ND grid
    grid = np.meshgrid(*grid_points_1d, indexing="ij")

    # build flow field dict for downstream functions that expect the flow
    # field in this format
    ndim = len(column_names)
    drift_vector_field = tuple(drift_values[..., i] for i in range(ndim))

    flow_field_dict = {"vectors": tuple(drift_vector_field), "grid": tuple(grid)}

    return flow_field_dict


def get_fixed_points_df(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """Get the fixed points dataframe for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset to retrieve fixed points for.
    model_manifest_name
        Name of the model manifest to use for locating the fixed points dataframe.
    run_name
        Name of the model run to use for locating the fixed points dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the fixed points for the specified dataset.
    """

    base_name = f"{model_manifest_name}_{run_name}_grid"
    fixed_points_df_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_df_manifest = load_dataframe_manifest(fixed_points_df_manifest_name)

    if dataset_name not in fixed_points_df_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in fixed points dataframe manifest [ %s ]!",
            dataset_name,
            fixed_points_df_manifest_name,
        )
        return pd.DataFrame()

    # load fixed point dataframe and check that required columns are present
    fixed_points_df_location = get_dataframe_location_for_dataset(
        fixed_points_df_manifest, dataset_name
    )
    fixed_points_df = load_dataframe(fixed_points_df_location, delay=False)

    return fixed_points_df
