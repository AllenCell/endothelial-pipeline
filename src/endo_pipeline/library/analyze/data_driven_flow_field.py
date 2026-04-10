"""Methods related to flow field estimation and analysis."""

import logging
from collections.abc import Callable
from time import perf_counter, time
from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata

from endo_pipeline.io.input import load_dataframe
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import get_fixed_points_within_bounds
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest_utils import get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_DRIFT,
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)


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
        vector_field_function=drift_function, dataframe=dataframe, column_names=column_names
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
