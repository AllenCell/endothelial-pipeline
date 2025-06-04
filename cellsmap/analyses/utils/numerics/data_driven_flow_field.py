from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import interpolate as spinterp
from scipy.integrate import solve_ivp
from sklearn.pipeline import Pipeline

import cellsmap.analyses.utils.io.vtk_io as vtk_io
import cellsmap.analyses.utils.regression_helper as rh
import cellsmap.analyses.utils.viz.flow_field_viz as ffv
import cellsmap.util.manifest_io as manifest_io
import cellsmap.util.manifest_preprocessing.diffae_feature_preprocessing as diffae_preproc


def set_3d_bounds_from_data(
    list_of_datasets: list[str],
    pca: Pipeline,
    col_names: Literal["pc", "feat"] = "pc",
) -> list[np.ndarray]:

    num_dims = 3
    bounds = [[100, -100], [100, -100], [100, -100]]

    for name in list_of_datasets:
        df = diffae_preproc.get_manifest_for_dynamics_workflows(name, pca)
        # get column names for features
        feat_cols = manifest_io.get_feature_cols(df)
        match col_names:
            case "pc":
                # get the PCs
                x_proj = pca.transform(df[feat_cols].values)
                # add PCs to dataframe
                num_pcs = x_proj.shape[1]
                pc_cols: list = []
                for pc in range(num_pcs):
                    pc_col_name = f"pc{pc+1}"
                    pc_cols.append(pc_col_name)
                cols = pc_cols
            case "feat":
                cols = feat_cols
        for j in range(num_dims):
            bounds[j][0] = min(bounds[j][0], df[cols[j]].min())
            bounds[j][1] = max(bounds[j][1], df[cols[j]].max())

    bounds = [np.array(b) for b in bounds]

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


def convert_coordinates_from_pc_to_latent(coords, reducer):
    """
    Convert coordinates in PCA-based feature space
    to latent space using the PCA model.
    """
    coords = np.array(coords)
    latent = reducer.inverse_transform(coords)
    num_coords = latent.shape[0]
    # turn coordinate array into list of lists
    latent_coords = [l.tolist() for l in latent]

    return latent_coords


def convert_coordinates_from_volume_to_pc(
    xvol: np.array, grid_spacing: float, origin: float
) -> np.array:
    """
    Convert coordinates from 3D volume space to 3D PC space
    (for saving as .vtk to view in ParaView).
    """
    xpc = origin + xvol * grid_spacing
    return xpc


def get_and_viz_ddff(
    name: str,
    pca: Pipeline,
    kernel_params: dict,
    dt: float,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    time_span: list,
    init: np.ndarray,
    fig_savedir: str,
    vtk_savedir: str,
    output_savedir: str,
) -> np.ndarray | list[np.ndarray]:

    # load dataframe and get top 3 PCs
    df = diffae_preproc.get_manifest_for_dynamics_workflows(name, pca)
    feat_cols = manifest_io.get_feature_cols(df)[:3]

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = rh.get_traj_and_diff(df, feat_cols)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = rh.get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = compute_extrapolated_vector_field(
        drift_km, centers, interpolator="nearest"
    )
    # save flow field dictionary as npy
    np.save(
        output_savedir + f"flow_field_dict_{name}.npy",
        flow_field_dict,
        allow_pickle=True,
    )
    # save flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        flow_field_dict, vtk_savedir + f"flow_field_{name}.vtk"
    )

    # compute interpolated diffusion field
    # (diagonal diffusion tensor represented as 3D vector field)
    diffusion_field_dict = compute_extrapolated_vector_field(
        diff_km, centers, interpolator="nearest"
    )
    # save diffusion field dictionary as npy
    np.save(
        output_savedir + f"diffusion_field_dict_{name}.npy",
        diffusion_field_dict,
        allow_pickle=True,
    )
    # save diffusion field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        diffusion_field_dict, vtk_savedir + f"diffusion_field_{name}.vtk"
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by init
    # solve IVP, get back trajectory
    traj = solve_ddff_ode(flow_field_dict, init, time_span)

    # call main flow field viz function (makes and saves plots)
    ffv.flow_field_viz_main(flow_field_dict, df, traj, fig_savedir)

    # hack-y work around for intermediate shear stress
    # simulate second trajectory to get second stable point
    if name == "20250319_20X" or name == "20250326_20X":
        init = np.array([1.1, 0.0, -0.2])
        time_span = [0, 5000]
        traj_2 = solve_ddff_ode(flow_field_dict, init, time_span)
        traj = [traj, traj_2]  # return both trajectories

    return traj


def ddff_main(
    list_of_datasets: list[str],
    pca: Pipeline,
    kernel_params: dict,
    dt: float,
    time_span: list,
    init: np.ndarray,
    fig_savedir: str,
    vtk_savedir: str,
    output_savedir: str,
) -> None:
    # get bins for KMCs
    bounds = set_3d_bounds_from_data(list_of_datasets, pca, col_names="feat")
    num_bins = [50, 50, 50]
    bins, centers = rh.get_bins(num_bins, bin_limits=bounds)

    # get experimental condition
    # descriptions of each dataset
    condition_dict = diffae_preproc.get_dataset_descriptions(
        list_of_datasets, simple=True
    )

    # initialize dict to save trajectories
    # used for crop reconstruction
    traj_dict = {}
    for name in list_of_datasets:
        traj = get_and_viz_ddff(
            name,
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
        condition = condition_dict[name]
        traj_dict[condition] = traj

    np.save(output_savedir + "traj_dict", traj_dict, allow_pickle=True)

    # generate plot of stable fixed points
    # for low, high, and 12dyn datasets
    ffv.plot_stable_fixed_points_together(fig_savedir, output_savedir)
