import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data, get_bins
from endo_pipeline.library.visualize.diffae_features import flow_field_viz, pplane, vtk_io
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.flow_field_3d import SAMPLER_RANDOM_SEED, TRAJECTORY_DICT_FILE_NAME

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
    np.ndarray
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


def _is_point_within_percentile(point, data, lower=5, upper=95):
    """
    Check if a point is within the given percentile range of the data along each axis.

    Parameters
    ----------
    point
        The point to check.
    data
        The data to compute percentiles from.
    lower
        Lower percentile.
    upper
        Upper percentile.

    Returns
    -------
    :
        True if point is within the percentile bounds on all axes, else False.
    """
    lower_bounds = np.percentile(data, lower, axis=0)
    upper_bounds = np.percentile(data, upper, axis=0)
    point = np.asarray(point)
    return np.all((point >= lower_bounds) & (point <= upper_bounds))


def _ddff_model_analysis(
    dataset_name: str,
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    kernel_params: dict,
    dt: float,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    time_span: list,
    init_for_traj: np.ndarray,
    num_inits_for_root_solver: int,
    plot_bounds: list[np.ndarray],
    plot_stack: bool,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
    pc_column_names: list[str] = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """
    Get 3d flow field (drift coefficient) from principal component features from a given dataset.

    For a single dataset, this workflow:

    1. Loads the dataframe for the given dataset and gets the top 3 PCs.
    2. Computes the drift and diffusion coefficients (first and second Kramers-Moyal coefficients)
        using a kernel-based method.
    3. Extrapolates the drift and diffusion coefficients to get a flow field over the entire 3D
        space as specified by the input bins and centers.
    4. Saves out these vector fields as .npy files and as .vtk files for visualization.
    5. Solves the ODE dx/dt = f(x) using scipy.integrate.solve_ivp, where f(x) is the flow field
        (drift coefficient) and x is the 3D state space.
    6. Visualizes the flow field and the trajectory using the main function in flow_field_viz.py.

    **Method output**

    The output is a dictionary with two keys:
    - "trajectory": numpy array of shape (num_t, 3) with the trajectory in 3D state space
    - "stable_fixed_points": list of stable fixed points found in the flow field within the
        specified percentile bounds.

    Parameters
    ----------
    dataset_name
        Name of dataset for which to compute the flow field.
    dataframe_manifest
        Dataframe manifest with the dataframe locations for each dataset.
    pca
        PCA model to use for transforming the data (projecting onto the top 3 PCs).
    kernel_params
        Parameters for the kernel-based estimation of Kramers-Moyal coefficients.
    dt
        Time step between frames.
    bins
        List of the bin edges for histogramming along each dimension in the 3D state space.
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
        Whether to plot the flow field as a stack of 2D slices in each dimension.
    fig_savedir
        Directory to save figures.
    vtk_savedir
        Directory to save .vtk files.
    output_savedir
        Directory to save output files (.npy files with drift field and diffusion field).
    pc_column_names
        List of column names for the principal components to use.
    lower_percentile
        Lower percentile for filtering fixed points.
    upper_percentile
        Upper percentile for filtering fixed points.
    """
    # load dataframe and get top 3 PCs
    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca,
        include_cell_piling=False,
        include_not_steady_state=False,
    )

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(df, pc_column_names)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute flow field on the grid defined by centers
    ndim = len(centers)  # number of dimensions
    # generate a mesh grid of points in the state space
    grid = np.meshgrid(*centers, indexing="ij")  # make meshgrid

    # get the vector field components from
    # the Kramers-Moyal coefficients
    drift_vector_field = [drift_km[..., i] for i in range(ndim)]
    flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

    # save flow field dictionary as npy
    np.save(
        output_savedir / f"flow_field_dict_{dataset_name}.npy",
        flow_field_dict,  # type: ignore
        allow_pickle=True,
    )

    # get callable version of the flow field
    # first, extrapolate to fill in NaNs
    extrapolated_flow_field_dict = compute_extrapolated_vector_field(
        drift_km, centers, method="linear"
    )
    # save out the flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        extrapolated_flow_field_dict, vtk_savedir / f"flow_field_{dataset_name}.vtk"
    )

    # compute diffusion field on the grid defined by centers
    # (diagonal diffusion tensor represented as 3D vector field)
    diffusion_vector_field = [diff_km[..., i] for i in range(ndim)]
    diffusion_field_dict = {"vectors": diffusion_vector_field, "grid": grid}

    # save diffusion field dictionary as npy
    np.save(
        output_savedir / f"diffusion_field_dict_{dataset_name}.npy",
        diffusion_field_dict,  # type: ignore
        allow_pickle=True,
    )
    # save diffusion field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        diffusion_field_dict, vtk_savedir / f"diffusion_field_{dataset_name}.vtk"
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by init solve IVP, get back trajectory
    traj = solve_ddff_ode(extrapolated_flow_field_dict, init_for_traj, time_span)

    # sample initial conditions from data density
    pc_data = df[pc_column_names].values  # get PC data as numpy array
    sampled_inits_for_root_solver = sample_from_density(pc_data, num_inits_for_root_solver)

    # get callable drift function and its Jacobian
    drift_function = get_callable_vector_field(
        extrapolated_flow_field_dict, for_solve_ivp=False, method="cubic"
    )
    drift_function_jacobian = Jacobian(drift_function)

    # pass into helper function to get fixed points
    fpts = pplane.get_fps(drift_function, sampled_inits_for_root_solver)

    # filter fixed points to only keep stable ones within 2nd-98th percentiles of data
    stable_fpts_high_confidence = []
    for fpt in fpts:
        within_percentile = _is_point_within_percentile(
            fpt, pc_data, lower_percentile, upper_percentile
        )
        if within_percentile:
            # get stability and type of the fixed point
            fpt_type = pplane.find_fpt_type(drift_function_jacobian(fpt))
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

    flow_field_viz.flow_field_viz_main(
        flow_field_dict,
        df,
        traj,
        stable_fpts_high_confidence,
        plot_bounds,
        plot_stack,
        fig_savedir_dataset,
    )

    output_dict = {
        "trajectory": traj,
        "stable_fixed_points": stable_fpts_high_confidence,
    }

    return output_dict


def get_and_analyze_ddff(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    kernel_params: dict,
    dt: float,
    time_span: list,
    init_for_traj: np.ndarray,
    num_inits_for_root_solver: int,
    num_bins: tuple[int, int, int],
    plot_stack: bool,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
    use_common_axis_limits: bool = False,
) -> None:
    """
    Visualize data-driven flow field (DDFF) for a list of datasets.

    **Method output**

    This function saves out the trajectories for each dataset in a single dictionary, where the
    keys are dataset descriptions (based on shear stress conditions) and the values are the
    trajectories in 3D state space.

    It also saves out figures and other intermediate files via it's calls to other functions.
    See the docstring for ``_ddff_model_analysis`` for more details.

    Parameters
    ----------
    dataset_names
        List of dataset names to analyze.
    dataframe_manifest
        Dataframe manifest with the dataframe locations for each dataset.
    pca
        PCA model to use for transforming the data (projecting onto the top 3 PCs).
    kernel_params
        Parameters for the kernel-based estimation of Kramers-Moyal coefficients.
    dt
        Time step between frames.
    time_span
        Time span for the ODE solver.
    init_for_traj
        Initial condition for the trajectory.
    num_inits_for_root_solver
        Number of initial conditions to use for finding fixed points.
    num_bins
        Number of bins for histogramming along each dimension in the 3D state space.
    plot_stack
        Whether to plot the flow field as a stack of 2D slices in each dimension.
    fig_savedir
        Directory to save figures.
    vtk_savedir
        Directory to save .vtk files.
    output_savedir
        Directory to save other output files.
    use_common_axis_limits
        Whether to use common axis limits for all datasets when plotting.
    """
    if use_common_axis_limits:
        # get common bounds for all datasets
        bounds_for_plots = get_3d_bounds_from_data(dataset_names, dataframe_manifest, pca)
    else:
        # get bounds for each dataset separately
        bounds_for_plots = None

    # initialize dict to save trajectories for crop reconstruction
    # and dict to store stable fixed points (visualized together later)
    traj_dict = {}
    stable_fixed_points_dict = {}
    for dataset_name in dataset_names:
        # get bins for KMCs
        bounds_for_km = get_3d_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=True,
        )
        bins, centers = get_bins(num_bins, bin_limits=bounds_for_km)
        output_dict = _ddff_model_analysis(
            dataset_name,
            dataframe_manifest,
            pca,
            kernel_params,
            dt,
            bins,
            centers,
            time_span,
            init_for_traj,
            num_inits_for_root_solver,
            bounds_for_plots if use_common_axis_limits else bounds_for_km,
            plot_stack,
            fig_savedir,
            vtk_savedir,
            output_savedir,
        )

        # save out trajectory for reconstruction using dataset descriptions
        traj_dict[dataset_name] = output_dict["trajectory"]
        stable_fixed_points_dict[dataset_name] = output_dict["stable_fixed_points"]

    np.save(output_savedir / TRAJECTORY_DICT_FILE_NAME, traj_dict, allow_pickle=True)  # type: ignore

    # generate plot of stable fixed points from different datasets overlaid on top of each other
    # (for comparison of stable fixed points across conditions)
    if bounds_for_plots is None:
        # get common bounds if not already computed
        bounds_for_plots = get_3d_bounds_from_data(dataset_names, dataframe_manifest, pca)
    flow_field_viz.plot_stable_fixed_points_together(
        stable_fixed_points_dict, bounds_for_plots, fig_savedir
    )


def compute_extrapolated_vector_field(
    kmcs: np.ndarray,
    grid_coordinates: list[np.ndarray],
    method: str = "linear",
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

    Parameters
    ----------
    kmcs
        Array of drift or diffusion estimates over a three dimensional grid.
    grid_coordinates
        List of 1D numpy arrays with the grid points in each dimension
    method
        Method to use for extrapolating the vector field where there are NaNs.
    """

    filled_kmcs = kmcs.copy()
    n_components = filled_kmcs.shape[-1]
    x, y, z = np.meshgrid(*grid_coordinates, indexing="ij")

    for i in range(n_components):
        component = filled_kmcs[..., i]
        nan_mask = np.isnan(component)
        if np.any(nan_mask):
            # Prepare points and values for interpolation
            # fill_value set to None for extrapolation
            interpolator = RegularGridInterpolator(
                grid_coordinates,
                np.where(nan_mask, 0, component),  # fill NaNs with zeros for shape
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
    t_span: list[int] | list[float],
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
        Time span for the ODE solver as [t0, tf].
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


def convert_coordinates_from_pc_to_latent(coords: np.ndarray, reducer: PCA) -> list[list]:
    """
    Convert coordinates in PCA-based feature space
    to latent space using the PCA model.
    """
    latent = reducer.inverse_transform(coords)
    latent.shape[0]
    # turn coordinate array into list of lists
    latent_coords = [coord.tolist() for coord in latent]

    return latent_coords


def convert_coordinates_from_volume_to_pc(
    xvol: np.ndarray, grid_spacing: float, origin: float
) -> np.ndarray:
    """
    Convert coordinates from 3D volume space to 3D PC space
    (for saving as .vtk to view in ParaView).
    """
    xpc = origin + xvol * grid_spacing
    return xpc
