import logging
from collections.abc import Callable
from pathlib import Path
from time import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, griddata
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
    get_dataset_descriptions,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data, get_bins
from endo_pipeline.library.visualize.diffae_features import flow_field_viz, vtk_io
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.flow_field_3d import TRAJECTORY_DICT_FILE_NAME

logger = logging.getLogger(__name__)


def _ddff_model_analysis(
    dataset_name: str,
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    kernel_params: dict,
    dt: float,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    time_span: list,
    init: np.ndarray,
    plot_bounds: list[np.ndarray],
    plot_stack: bool,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
    pc_column_names: list[str] = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
) -> np.ndarray | list[np.ndarray]:
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
    init
        Initial condition for the trajectory.
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

    Returns
    -------
    :
        Trajectory in 3D state space for the given initial condition and time span
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
    vtk_io.save_vector_field_as_vtk(
        extrapolated_flow_field_dict, vtk_savedir / f"flow_field_{dataset_name}.vtk", volume_extent
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
        diffusion_field_dict, vtk_savedir / f"diffusion_field_{dataset_name}.vtk", volume_extent
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by init
    # solve IVP, get back trajectory
    traj = solve_ddff_ode(extrapolated_flow_field_dict, init, time_span)

    flow_field_viz.flow_field_viz_main(
        flow_field_dict, df, traj, plot_bounds, plot_stack, fig_savedir
    )

    return traj


def get_and_analyze_ddff(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    kernel_params: dict,
    dt: float,
    time_span: list,
    init: np.ndarray,
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
    init
        Initial condition for the trajectory.
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

    # get experimental condition
    # descriptions of each dataset
    condition_dict = get_dataset_descriptions(dataset_names, simple=True)

    # initialize dict to save trajectories
    # used for crop reconstruction
    traj_dict = {}
    for dataset_name in dataset_names:
        # get bins for KMCs
        bounds_for_km = get_3d_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=True,
        )
        bins, centers = get_bins(num_bins, bin_limits=bounds_for_km)
        traj = _ddff_model_analysis(
            dataset_name,
            dataframe_manifest,
            pca,
            kernel_params,
            dt,
            bins,
            centers,
            time_span,
            init,
            bounds_for_plots if use_common_axis_limits else bounds_for_km,
            plot_stack,
            fig_savedir,
            vtk_savedir,
            output_savedir,
        )

        # save out using dataset descriptions
        condition = condition_dict[dataset_name]
        traj_dict[condition] = traj

    np.save(output_savedir / TRAJECTORY_DICT_FILE_NAME, traj_dict, allow_pickle=True)  # type: ignore

    # generate plot of stable fixed points from different datasets
    # overlaid on top of each other
    # (for comparison of stable fixed points across conditions)
    flow_field_viz.plot_stable_fixed_points_together(dataset_names, fig_savedir, output_savedir)


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


def convert_coordinates_from_volume_to_pc(
    xvol: np.ndarray, grid_spacing: float, origin: float
) -> np.ndarray:
    """
    Convert coordinates from 3D volume space to 3D PC space
    (for saving as .vtk to view in ParaView).
    """
    xpc = origin + xvol * grid_spacing
    return xpc
