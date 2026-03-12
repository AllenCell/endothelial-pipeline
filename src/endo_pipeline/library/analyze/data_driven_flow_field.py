import logging
import re
from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import numpy as np
import pandas as pd
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import circpercentile
from endo_pipeline.library.visualize.diffae_features.flow_field_viz import flow_field_viz_main
from endo_pipeline.library.visualize.diffae_features.pplane import find_fpt_type, get_fps
from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.dynamics_workflows import BIN_LIMITS_THETA_RESCALED
from endo_pipeline.settings.flow_field_3d import SAMPLER_RANDOM_SEED

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
        if column_name == ColumnName.POLAR_ANGLE:
            percentile_value = circpercentile(data[column_name], q=q, polar_range=polar_angle_range)
        else:
            percentile_value = np.percentile(data[column_name], q=q)
        percentile_values[column_name] = percentile_value
    return percentile_values


def _is_point_within_percentile_bounds(
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

    **Handling circular variables*

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
    num_variables = len(column_names)
    is_within_bounds = np.zeros(num_variables, dtype=bool)
    for i, column_name in enumerate(column_names):
        lower_bound = lower_percentile_bounds[column_name]
        upper_bound = upper_percentile_bounds[column_name]
        if column_name == ColumnName.POLAR_ANGLE:
            # for circular variables, need to account for bounds wrapping around
            if lower_bound <= upper_bound:
                is_within_bounds[i] = (lower_bound <= point[i]) & (point[i] <= upper_bound)
            else:
                # check if point is within bounds accounting for wraparound
                # and given polar range (e.g. [0, 2pi] or [-pi, pi])
                is_within_bounds[i] = (polar_angle_range[1] >= point[i] >= lower_bound) | (
                    polar_angle_range[0] <= point[i] <= upper_bound
                )
        else:
            is_within_bounds[i] = (lower_bound <= point[i]) & (point[i] <= upper_bound)
    return np.all(is_within_bounds)


def ddff_model_analysis(
    dataset_name: str,
    dataframe_manifest: DataframeManifest,
    crop_pattern: Literal["grid", "tracked"],
    pca: PCA,
    kernel: KramersMoyalKernel,
    dt: float,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    time_span: tuple[float, float],
    init_for_traj: np.ndarray,
    num_inits_for_root_solver: int,
    plot_bounds: list[tuple[float, float]],
    plot_stack: bool,
    compute_vtk_files: bool,
    fig_savedir: Path,
    vtk_savedir: Path,
    column_names: list[str],
    lower_percentile: float,
    upper_percentile: float,
    polar_angle_range: tuple[float, float],
) -> list[np.ndarray]:
    """
    Get 3d flow field (drift coefficient) from principal component features from
    a given dataset.

    For a single dataset, this workflow:

    1. Loads the dataframe for the given dataset and gets the top 3 PCs.
    2. Computes the drift and diffusion coefficients (first and second
       Kramers-Moyal coefficients)
        using a kernel-based method.
    3. Extrapolates the drift and diffusion coefficients to get a flow field
       over the entire 3D
        space as specified by the input bins and centers.
    4. Saves out these vector fields as .npy files and as .vtk files for
       visualization.
    5. Solves the ODE dx/dt = f(x) using scipy.integrate.solve_ivp, where f(x)
       is the flow field
        (drift coefficient) and x is the 3D state space.
    6. Visualizes the flow field and the trajectory using the main function in
       flow_field_viz.py.

    Parameters
    ----------
    dataset_name
        Name of dataset for which to compute the flow field.
    dataframe_manifest
        Dataframe manifest with the dataframe locations for each dataset.
    pca
        PCA model to use for transforming the data (projecting onto the top 3
        PCs).
    kernel
        Kernel to use for Kramers-Moyal coefficient estimation.
    dt
        Time step between frames.
    bins
        List of the bin edges for histogramming along each dimension in the 3D
        state space.
    centers
        List of centers of the bins in each dimension.
    time_span
        Time span for the ODE solver.
    init_for_traj
        Initial condition for the trajectory.
    num_inits_for_root_solver
        Number of initial conditions to use for finding fixed points.
    plot_bounds
        Bounds for plotting the flow field.
    plot_stack
        Whether to plot the flow field as a stack of 2D slices in each
        dimension.
    compute_vtk_files
        Whether to compute and save .vtk files for the flow field and diffusion
        field.
    fig_savedir
        Directory to save figures.
    vtk_savedir
        Directory to save .vtk files.
    output_savedir
        Directory to save output files (.npy files with drift field and
        diffusion field).
    column_names
        List of column names corresponding to features to use for the analysis
        (e.g. the top 3 PCs).
    lower_percentile
        Lower percentile for filtering fixed points.
    upper_percentile
        Upper percentile for filtering fixed points.

    Returns
    -------
    :
        List of stable fixed points with high confidence (filtered by percentile
        range).
    """
    # load dataframe and get top 3 PCs
    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
        crop_pattern=crop_pattern,
    )

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(df, column_names)
    # get drift estimates
    # (Kramers-Moyal coefficients)
    drift_km, _ = get_kramers_moyal_coeffs(traj_list, d_traj_list, bins=bins, dt=dt, kernel=kernel)

    # compute flow field on the grid defined by centers
    ndim = len(centers)  # number of dimensions
    # generate a mesh grid of points in the state space
    grid = np.meshgrid(*centers, indexing="ij")  # make meshgrid

    # get the vector field components from
    # the Kramers-Moyal coefficients
    drift_vector_field = [drift_km[..., i] for i in range(ndim)]
    flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

    # if compute vtk files, extrapolate and save out the flow field as vtk
    if compute_vtk_files:
        extrapolated_flow_field_dict_vtk = compute_extrapolated_vector_field(
            drift_km, centers, method="nearest", for_vtk_files=True
        )
        # save out the flow field as vtk image data
        volume_extent = {
            "xmin": bins[0][0],
            "xmax": bins[0][-1],
            "ymin": bins[1][0],
            "ymax": bins[1][-1],
            "zmin": bins[2][0],
            "zmax": bins[2][-1],
        }
        save_vector_field_as_vtk(
            extrapolated_flow_field_dict_vtk,
            vtk_savedir / f"flow_field_{dataset_name}.vtk",
            volume_extent,
        )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by init solve IVP, get back trajectory
    extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
        drift_km, centers, method="linear", for_vtk_files=False
    )
    traj = solve_ddff_ode(extrapolated_flow_field_dict_reg, init_for_traj, time_span)

    # get callable drift function and its Jacobian
    drift_function = get_callable_vector_field(
        extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
    )
    drift_function_jacobian = Jacobian(drift_function)

    # sample initial conditions for root solver from data density
    feature_data = df[column_names]  # get feature data as numpy array
    sampled_inits_for_root_solver = sample_from_density(
        feature_data.to_numpy(), num_inits_for_root_solver
    )

    # pass into helper function to get fixed points
    fpts = get_fps(drift_function, sampled_inits_for_root_solver)

    # filter fixed points to only keep stable ones within a given range of
    # percentiles of data (e.g., 2 to 98) to get high confidence fixed points
    # that are within the region of state space supported by the data
    lower_percentile_bounds = _compute_percentile_values(
        feature_data, column_names, q=lower_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Lower percentile bounds for filtering fixed points: [ %s ]", lower_percentile_bounds
    )
    upper_percentile_bounds = _compute_percentile_values(
        feature_data, column_names, q=upper_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Upper percentile bounds for filtering fixed points: [ %s ]", upper_percentile_bounds
    )
    stable_fpts_high_confidence = []
    for fpt in fpts:
        within_percentile = _is_point_within_percentile_bounds(
            fpt, column_names, lower_percentile_bounds, upper_percentile_bounds, polar_angle_range
        )
        if within_percentile:
            # get stability and type of the fixed point
            fpt_type = find_fpt_type(drift_function_jacobian(fpt))
            # stability of the fixed point is the
            # first word in the fpt_type string
            # if verbose, print the point and its stability
            logger.debug("[ %s ] at [ (%.2f, %.2f, %.2f) ]", fpt_type, fpt[0], fpt[1], fpt[2])
            # if "Stable" or "stable" in the fpt_type, save the point
            if re.search(r"stable", fpt_type, re.IGNORECASE) and not re.search(
                r"unstable", fpt_type, re.IGNORECASE
            ):
                stable_fpts_high_confidence.append(fpt)

    # subfolder for each dataset
    fig_savedir_dataset = fig_savedir / dataset_name
    fig_savedir_dataset.mkdir(parents=True, exist_ok=True)

    # call main visualization function
    flow_field_viz_main(
        flow_field_dict,
        df,
        column_names,
        traj,
        stable_fpts_high_confidence,
        plot_bounds,
        plot_stack,
        fig_savedir_dataset,
    )

    return stable_fpts_high_confidence


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


def get_callable_vector_field(
    vector_field_dict: dict, for_solve_ivp: bool = True, method: str = "linear"
) -> Callable:
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
