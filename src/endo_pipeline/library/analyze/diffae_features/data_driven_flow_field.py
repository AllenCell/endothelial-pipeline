import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import interpolate as spinterp
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe import (
    get_dataframe_for_dynamics_workflows,
    get_dataset_descriptions,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data, get_bins
from endo_pipeline.library.visualize.diffae_features import flow_field_viz, vtk_io
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, NUM_PCS_TO_ANALYZE

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
    df = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca)

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(df, pc_column_names)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = compute_extrapolated_vector_field(
        drift_km, centers, extrapolation_method="nearest"
    )
    # save flow field dictionary as npy
    np.save(
        output_savedir / f"flow_field_dict_{dataset_name}.npy",
        flow_field_dict,  # type: ignore
        allow_pickle=True,
    )
    # save flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(flow_field_dict, vtk_savedir / f"flow_field_{dataset_name}.vtk")

    # compute interpolated diffusion field
    # (diagonal diffusion tensor represented as 3D vector field)
    diffusion_field_dict = compute_extrapolated_vector_field(
        diff_km, centers, extrapolation_method="nearest"
    )
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
    # with initial conditions given by init
    # solve IVP, get back trajectory
    traj = solve_ddff_ode(flow_field_dict, init, time_span)

    # call main flow field viz function (makes and saves plots)
    flow_field_viz.flow_field_viz_main(flow_field_dict, df, traj, fig_savedir)

    # hack-y work around for intermediate shear stress
    # simulate second trajectory to get second stable point
    if dataset_name == "20250319_20X":
        init = np.array([1.1, 0.0, -0.2])
        time_span = [0, 5000]
        traj_2 = solve_ddff_ode(flow_field_dict, init, time_span)
        traj_list = [traj, traj_2]  # return both trajectories
        return traj_list
    else:
        return traj


def get_and_analyze_ddff(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    kernel_params: dict,
    dt: float,
    time_span: list,
    init: np.ndarray,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
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
    fig_savedir
        Directory to save figures.
    vtk_savedir
        Directory to save .vtk files.
    output_savedir
        Directory to save other output files.
    """
    # get bins for KMCs
    bounds = get_3d_bounds_from_data(dataset_names, dataframe_manifest, pca)
    num_bins = [50, 50, 50]
    bins, centers = get_bins(num_bins, bin_limits=bounds)

    # get experimental condition
    # descriptions of each dataset
    condition_dict = get_dataset_descriptions(dataset_names, simple=True)

    # initialize dict to save trajectories
    # used for crop reconstruction
    traj_dict = {}
    for dataset_name in dataset_names:
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
            fig_savedir,
            vtk_savedir,
            output_savedir,
        )

        # save out using dataset descriptions
        condition = condition_dict[dataset_name]
        traj_dict[condition] = traj

    np.save(output_savedir / "traj_dict", traj_dict, allow_pickle=True)  # type: ignore

    # generate plot of stable fixed points from different datasets
    # overlaid on top of each other
    # (for comparison of stable fixed points across conditions)
    flow_field_viz.plot_stable_fixed_points_together(dataset_names, fig_savedir, output_savedir)


def compute_extrapolated_vector_field(
    kmcs: np.ndarray,
    grid_centers: list[np.ndarray],
    extrapolation_method: Literal["nearest", "linear"] = "nearest",
) -> dict:
    """
    Extrapolate a 3D vector field from Kramers-Moyal estimates over a specified grid.

    **Method inputs**

    The input ``kmcs`` are the Kramers-Moyal coefficients (drift or diffusion estimates)
    over a 3D mesh grid obtained from feature data. Where there are no data points, these
    estimates are `NaN`. This function extrapolates these estimates to the entire grid
    using nearest-neighbor or linear interpolation.

    The array ``kmcs`` should have shape (num_bins_x, num_bins_y, num_bins_z, 3), where
    num_bins_x, num_bins_y, and num_bins_z are the number of bins in each dimension
    of the 3D meshgrid defined by ``grid_centers``.

    **Method output**

    The output is a dictionary with two keys:
    - "vectors": tuple of 3D arrays (f1,f2,f3) with the vector values in each dimension
    - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each dimension

    Parameters
    ----------
    kmcs
        Array of drift or diffusion estimates over a three dimensional grid.
    grid_centers
        List of 1D numpy arrays with the grid points in each dimension
    extrapolation_method
        Method to use for extrapolating the vector field where there are NaNs.
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
    if extrapolation_method not in ["nearest", "linear"]:
        logger.error(
            "Extrapolation method [ %s ] not recognized. Use 'nearest' or 'linear'.",
            extrapolation_method,
        )
        raise ValueError(f"Extrapolation method [ {extrapolation_method} ] not recognized.")
    if extrapolation_method == "nearest":  # nearest neighbor
        interpolator_func = [
            spinterp.NearestNDInterpolator(valid_points, valid_values[i]) for i in range(ndim)
        ]
    else:  # linear interpolation
        interpolator_func = [
            spinterp.LinearNDInterpolator(valid_points, valid_values[i]) for i in range(ndim)
        ]

    # Find the indices of all points (including NaN points)
    all_points = (
        np.array(np.indices(vector_field[0].shape)).reshape(len(vector_field[0].shape), -1).T
    )

    # Interpolate the NaN points
    vec_interpolated = [interpolator_func[i](all_points) for i in range(ndim)]

    # reshape back to original grid shape
    vec_interpolated = [vec.reshape(vector_field[0].shape) for vec in vec_interpolated]

    # Create a dictionary to store the vector field and grid
    vector_field_dict = {"vectors": vec_interpolated, "grid": grid}

    return vector_field_dict


def get_callable_vector_field(vector_field_dict: dict, for_solve_ivp: bool = True) -> Callable:
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

    Returns
    -------
    :
        Callable function representing the vector field.
    """

    ndim = len(vector_field_dict["grid"])  # number of dimensions

    # get the interpolator for f_KM
    vec_field_grid = np.stack(
        vector_field_dict["vectors"], axis=-1
    )  # shape (num_bins_x, num_bins_y, ... , ndim)
    xyz_grid = np.moveaxis(np.array(vector_field_dict["grid"]), 0, -1).reshape((-1, ndim))
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
        def vec_func_ivp(t: Any, x: np.ndarray) -> np.ndarray:
            # get interpolated value
            vec_interp_val = vec_field_interp(x)
            # return dx/dt = f(x)
            return vec_interp_val

        return vec_func_ivp
    else:

        def vec_func(x: np.ndarray) -> np.ndarray:
            # get interpolated value
            vec_interp_val = vec_field_interp(x)
            # return dx/dt = f(x)
            return vec_interp_val

        return vec_func


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
    my_flow = get_callable_vector_field(flow_field_dict)
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
