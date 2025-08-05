import logging
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import color_palette

from src.endo_pipeline.configs import (
    get_model_manifest,
    load_dataset_collection_config,
    load_model_config,
)
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import load_dataframe
from src.endo_pipeline.library.analyze.diffae_features import (
    compute_extrapolated_vector_field,
    solve_ddff_ode,
)
from src.endo_pipeline.library.analyze.diffae_manifest import (
    add_description_column,
    get_manifest_for_dynamics_workflows,
    get_traj_and_diff,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.kramersmoyal.kramers_moyal import get_kramers_moyal
from src.endo_pipeline.library.analyze.numerics.binning import get_3d_bounds_from_data, get_bins
from src.endo_pipeline.library.analyze.optical_flow_calculator import (
    one_direction_vector_field_example,
)
from src.endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from src.endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

logger = logging.getLogger(__name__)


def add_normalized_time(
    df_all_positions: pd.DataFrame,
    time_col: str = "time_hours",
) -> pd.DataFrame:
    """
    Add a column to the dataframe with normalized time values
    between 0 and 1 for each track_id in each position.

    Parameters
    ----------
    df_all_positions
        DataFrame containing all positions and tracks.
    time_col
        The name of the column containing time values.

    Returns
    -------
    :
        DataFrame with an additional column
        "normalized_time" containing the normalized time values between 0 and 1.
    """
    for _, df_pos in df_all_positions.groupby("position_as_str"):
        for _, df_track in df_pos.groupby("track_id"):

            time_values = df_track[time_col].values.astype(np.float64)
            sorted_inds = np.argsort(time_values)
            time_values = time_values[sorted_inds]
            df_track = df_track.iloc[sorted_inds]

            start_time = np.min(time_values)
            end_time = np.max(time_values)

            normalized_time_values = np.divide(
                time_values - start_time,
                end_time - start_time,
                out=np.zeros_like(time_values, dtype=np.float64),
                where=(end_time - start_time) != 0,
            )

            normalized_time_values = np.clip(normalized_time_values, 0, 1)

            df_all_positions.loc[
                df_track.index,
                "normalized_time",
            ] = normalized_time_values

    return df_all_positions


def get_coarse_grained_trajectory_heatmap_data(
    df_all_positions: pd.DataFrame,
    bounds: np.ndarray | List,
    num_bins: List[int] = [150, 150, 150],
    pc_cols: List[str] = ["pc1", "pc2", "pc3"],
    feature_to_use: str = "normalized_time",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a coarse-grained trajectory heatmap data from the DataFrame.

    Parameters
    ----------
    df_all_positions
        DataFrame containing tracks for one microscope position.
    bounds
        Bounds for the heatmap in each dimension.
        Should be a list of tuples or a 2D numpy array with shape (ndim, 2),
        where ndim is the number of dimensions.
    num_bins
        Number of bins for each dimension in the heatmap.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the heatmap data and the bin counts.
    """
    if feature_to_use not in df_all_positions.columns:
        raise ValueError(f"Feature '{feature_to_use}' not found in DataFrame columns.")

    bin_data = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)
    ndim = len(pc_cols)
    bins_array = np.array(
        [np.linspace(bounds[i][0], bounds[i][1], num_bins[i]) for i in range(ndim)]
    ).T
    for _, df_one_position in df_all_positions.groupby("position_as_str"):
        for _, df_track in df_one_position.groupby("track_id"):
            trajectory = df_track[pc_cols].values
            feature_values = df_track[feature_to_use].values
            bin_indices = np.zeros((trajectory.shape[0], ndim), dtype=int)
            for dim in range(len(pc_cols)):
                # get the bin index in which each timepoint lies
                bin_indices[:, dim] = np.digitize(trajectory[:, dim], bins_array[:, dim]) - 1
                # clip the bin indices to be within the valid range
                bin_indices[:, dim] = np.clip(bin_indices[:, dim], 0, num_bins[dim] - 1)
            # increment the bin data and count
            for i in range(trajectory.shape[0]):
                bin_data[tuple(bin_indices[i])] += feature_values[i]
                bin_counts[tuple(bin_indices[i])] += 1

    return bin_data, bin_counts


def merge_diffae_feats_liveseg_feats_tables(
    diffae_tracking_df: pd.DataFrame,
    live_seg_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the DiffAE tracking data with the live segmentation features data.

    Parameters:
        diffae_tracking_df (pd.DataFrame): DataFrame containing DiffAE tracking data.
        live_seg_feats_df (pd.DataFrame): DataFrame containing live segmentation features data.

    Returns:
        pd.DataFrame: Merged DataFrame with DiffAE and live segmentation features.
    """
    dataset_name = sequence_to_scalar(diffae_tracking_df["dataset"])
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    diffae_tracking_df["is_unique"] = diffae_tracking_df.groupby(
        ["dataset", "position", "frame_number", "track_id"]
    )["frame_number"].transform(lambda t: t.nunique() == t.size)
    diffae_tracking_df = diffae_tracking_df[diffae_tracking_df["is_unique"]]

    # give the crop_index column the same value as the track_ids
    diffae_tracking_df["crop_index"] = (
        diffae_tracking_df.groupby(["position", "track_id"], as_index=False).ngroup().astype(int)
    )
    diffae_tracking_df = add_description_column(
        diffae_tracking_df, dataset_name, simple=True
    )  # add description column (e.g., 48hr_High)
    diffae_tracking_df["track_id"] = diffae_tracking_df["track_id"].astype(int)
    diffae_tracking_df.rename(columns={"position": "position_as_str"}, inplace=True)

    logging.debug("processing the live segmentation features data...")
    live_seg_feats_df["position_as_str"] = live_seg_feats_df["position"].transform(
        lambda x: "P" + str(x)
    )
    live_seg_feats_df["track_id"] = live_seg_feats_df["track_id"].astype(int)

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merged_feats_df = pd.merge(
        left=live_seg_feats_df,
        right=diffae_tracking_df,
        how="left",
        left_on=["dataset_name", "position_as_str", "image_index", "track_id"],
        right_on=["dataset", "position_as_str", "frame_number", "track_id"],
        validate="one_to_one",
    )

    return merged_feats_df


def get_diffae_feats_liveseg_feats_merged_table(
    dataset_name: str, filtered: bool = False
) -> pd.DataFrame:

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_track_manifest = load_dataframe_manifest("diffae_tracking_integration")
    diffae_track_location = get_dataframe_location_for_dataset(diffae_track_manifest, dataset_name)
    diffae_tracking_df = load_dataframe(diffae_track_location)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_manifest = load_dataframe_manifest("live_merged_seg_features")
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df = load_dataframe(live_seg_location)

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    if filtered:
        # filter the merged table
        merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

        # remove any rows that were not evaluated by the model and thus have no mlflow_id
        merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

    return merged_feats_df


def get_traj_and_flowfield(
    df: pd.DataFrame,
    bounds: list,
    load_precomputed_trajectories: Path | None,
) -> tuple[np.ndarray, dict]:

    # load default config, get kernel params
    dynamics_config = load_dynamics_config("default")
    kernel_params = dynamics_config["kramers_moyal"]["kernel_params"]

    # get time between frames in minutes
    dt = dynamics_config["dt"]

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = [0.0, 2880.0]

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array([-0.1, -0.7, -0.1])

    num_bins = [50, 50, 50]
    bins, centers = get_bins(num_bins, bin_limits=bounds)

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = [f"pc{pc+1}" for pc in range(3)]

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(df, cols)

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = compute_extrapolated_vector_field(drift_km, centers, interpolator="nearest")

    if load_precomputed_trajectories is not None:
        logger.debug("Loading precomputed trajectories...")
        traj = np.load(load_precomputed_trajectories)
    else:
        # solve IVP, get back trajectory
        logger.debug("Trying to solve ODE...")
        traj = solve_ddff_ode(flow_field_dict, init, time_span)
        logger.debug("ODE solved.")

    return traj, flow_field_dict


def get_gridcrop_and_cellcentric_trajectories_and_flow_fields(
    dataset_name: str,
    merged_feats_df: pd.DataFrame,
    diffae_grid_crops: pd.DataFrame,
    bounds: list[float],
    trajectory_dir: Path,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    Get the trajectories and flow fields for the grid-based and cell-centric crops.
    This function is called after loading and preprocessing the manifests.
    The function looks for precomputed trajectories in trajectory_dir and loads them
    from there if found. If not found then they will be computed and saved to that location.
    The names of the files that it looks for are:
    - {dataset_name}_traj_grids.npy for grid-based crops
    - {dataset_name}_traj_tracks.npy for cell-centric crops
    """
    logger.info("Getting trajectories and flow fields for grid-based and cell-centric crops...")

    # try to load the grid crop-based  data for the cell-centric
    #  crops or, if needed, compute and save them
    precomputed_trajectories_path = trajectory_dir / f"{dataset_name}_traj_grids.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for grid-based crops...")
    # This takes about 2 minutes to compute if not loading precomputed
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        df=diffae_grid_crops,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the grid-based crops...")
        np.save(precomputed_trajectories_path, traj_grids)

    # try to load the trajectory data for the cell-centric crops or,
    # if needed, compute and save them
    precomputed_trajectories_path = trajectory_dir / f"{dataset_name}_traj_tracks.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for tracks-based crops...")
    # This takes about 5 minutes to compute if not loading precomputed
    traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(
        df=merged_feats_df,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the track-based crops...")
        np.save(precomputed_trajectories_path, traj_tracks)

    return traj_grids, flow_field_dict_grids, traj_tracks, flow_field_dict_tracks


def get_vector_vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle_rad


def get_vector_vector_angle_fast(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot_prod = np.einsum("ij,ij->i", v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    angle_rad = np.arccos(dot_prod / (norm1 * norm2))
    return angle_rad


def get_approx_vec_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes])))
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    v1_grids_approx = v1_grids[slice_indexes][min_idx]
    v2_grids_approx = v2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(v1_grids_approx.tolist(), v2_grids_approx.tolist())))


def get_approx_point_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes])))
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    g1_grids_approx = g1_grids[slice_indexes][min_idx]
    g2_grids_approx = g2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(g1_grids_approx.tolist(), g2_grids_approx.tolist())))


def get_vector_angles_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the angles of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )
    ang_full = get_vector_vector_angle_fast(vecs_grids, vecs_tracks)
    ang_arr = ang_full.reshape((50, 50, 50))
    angles = ang_arr[slice_indexes].reshape(my_shape)
    return angles


def get_vector_dot_products_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the dot products of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )
    dot_prod_full = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    dot_prod_arr = dot_prod_full.reshape((50, 50, 50))
    dot_prod = dot_prod_arr[slice_indexes].reshape(my_shape)
    return dot_prod


def make_angular_deviation_test(out_dir: Path) -> None:
    test_flow_field = one_direction_vector_field_example()

    test_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ]
    )

    test_points = np.array(
        [
            [-8.0, -4.0],
            [-6.0, -3.0],
            [-4.0, -2.0],
            [-2.0, -1.0],
            [2.0, 1.0],
            [4.0, 2.0],
            [6.0, 3.0],
            [8.0, 4.0],
        ]
    )

    slice_indexes = np.where(np.ones_like(test_flow_field[0][1]))
    test_flow_field_points = get_approx_point_from_grid(
        test_points,
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        slice_indexes,
    )

    test_flow_field_vectors = get_approx_vec_from_grid(
        test_vectors,
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        slice_indexes,
    )

    test_angular_deviation = get_vector_vector_angle_fast(test_flow_field_vectors, test_vectors)
    test_angular_deviation_deg = np.rad2deg(test_angular_deviation)

    cmap = color_palette("dark:red", as_cmap=True)
    angle_deg_to_color = lambda a: cmap(np.abs(a) / 180.0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.quiver(
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color="lightgrey",
    )
    ax.quiver(
        test_flow_field_points[:, 0],
        test_flow_field_points[:, 1],
        test_flow_field_vectors[:, 0],
        test_flow_field_vectors[:, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color="grey",
    )
    ax.quiver(
        test_points[:, 0],
        test_points[:, 1],
        test_vectors[:, 0],
        test_vectors[:, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color=angle_deg_to_color(test_angular_deviation_deg),
    )
    ax.set_xlim(-9, 9)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("Angular deviation from\nflow field test")
    fig.savefig(
        out_dir / "get_angular_deviation_deg_test.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return


def get_preprocessed_manifests_and_km_bounds(
    dataset_name: str,
    datasets_for_bounds: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load and process the DiffAE and live segmentation feature manifests for a given dataset.
    If no `datasets_for_bounds` are provided, it uses the reference datasets plus dataset_name
    to compute the bounds for the PCA projection. In my experience using only the dataset_name
    for the bounds has sometimes caused the solver to hang, perhaps due to overly restrictive bounds.
    """
    logger.info(f"Loading and processing manifests for dataset: {dataset_name}")

    # load the tables
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name, filtered=True)

    # fit the PCA (uses the reference datasets)
    pca = fit_pca()

    # read in the grid crop-based diffae features
    model_name = sequence_to_scalar(merged_feats_df["model_name"])
    model_config = load_model_config(model_name)
    model_manifest = get_model_manifest(dataset_name, model_config)  # type: ignore[arg-type]
    diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

    # add the PC columns to the track-based DiffAE table
    # (the grid-based DiffAE table already has them, but
    # but I believe that the columns are named "feat_0",
    # "feat_1", etc. when they should be named "pc1",
    # "pc2", etc.)
    merged_feats_df = project_manifest_to_pcs(merged_feats_df, pca)

    # use the full set of datasets to be analyzed for the bounds
    if datasets_for_bounds is None:
        datasets_for_bounds = load_dataset_collection_config("pca_reference").datasets + [
            dataset_name
        ]

    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in datasets_for_bounds  # type: ignore[arg-type]
    ]
    bounds = get_3d_bounds_from_data(model_manifest_list, pca)

    # lastly, add a normalized version of the "time_hours" column
    merged_feats_df = add_normalized_time(merged_feats_df)

    return merged_feats_df, diffae_grid_crops, bounds
