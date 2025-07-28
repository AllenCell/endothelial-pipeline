import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.endo_pipeline.configs import (
    get_model_manifest,
    load_dataset_collection_config,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import configure_logging, get_output_path, load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.analyze.optical_flow_calculator import (
    one_direction_vector_field_example,
)
from src.endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import plot_one_slice_quiver
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import (
    get_valid_slice_indexes,
    grid_vs_track_vec_angle_hist2d,
    grid_vs_track_vec_dot_prod_hist2d,
    plot_grid_vs_tracks_flow_field,
)
from src.endo_pipeline.library.visualize.seg_features.general_standard_plots import hist_2D_of_feats

logger = logging.getLogger(__name__)


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
    # diffae_tracking_df["crop_index"] = diffae_tracking_df["track_id"]
    diffae_tracking_df["crop_index"] = (
        diffae_tracking_df.groupby(["position", "track_id"], as_index=False).ngroup().astype(int)
    )
    diffae_tracking_df = diffae_preproc.add_description_column(
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


def get_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> pd.DataFrame:

    logging.debug(f"Loading dataset config file for dataset: {dataset_name}...")
    dataset_config = load_dataset_config(dataset_name)

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_fms_id = dataset_config.diffae_tracking_integration_fmsid
    if diffae_fms_id is None:
        logging.warning(
            f"No DiffAE track integration FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    diffae_tracking_df = load_dataframe_from_fms(diffae_fms_id)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_fmsid = dataset_config.live_merged_seg_features_manifest_fmsid
    if live_seg_fmsid is None:
        logging.warning(
            f"No live segmentation features FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    live_seg_feats_df = load_dataframe_from_fms(live_seg_fmsid)  # this takes a minute

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    return merged_feats_df


def get_traj_and_flowfield(
    df: pd.DataFrame,
    bounds: Pipeline,
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
    bins, centers = rh.get_bins(num_bins, bin_limits=bounds)

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = [f"pc{pc+1}" for pc in range(3)]

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = rh.get_traj_and_diff(df, cols)

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = rh.get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = ddff.compute_extrapolated_vector_field(
        drift_km, centers, interpolator="nearest"
    )

    if load_precomputed_trajectories is not None:
        logger.debug("Loading precomputed trajectories...")
        traj = np.load(load_precomputed_trajectories)
    else:
        # solve IVP, get back trajectory
        logger.debug("Trying to solve ODE...")
        traj = ddff.solve_ddff_ode(flow_field_dict, init, time_span)
        logger.debug("ODE solved.")

    return traj, flow_field_dict


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


def plot_pc_integrated_track(
    dataset_name: str,
    position_name: str,
    track_id: int,
    df: pd.DataFrame,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    out_subdir: Path,
) -> None:

    out_subdir_integrated_tracks = out_subdir / "integrated_tracks"
    out_subdir_integrated_tracks.mkdir(parents=True, exist_ok=True)

    out_subdir_integrated_tracks_hued = out_subdir / "integrated_tracks_hued"
    out_subdir_integrated_tracks_hued.mkdir(parents=True, exist_ok=True)

    # plot a single track integrated into the flow field
    # shown as dots connected by arrows to give an idea
    # of the direction of motion of the cell through the
    # flow field
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="blue",
    )
    ax.quiver(
        df["pc1"].iloc[:-1],
        df["pc2"].iloc[:-1],
        df["dpc1"].iloc[1:],
        df["dpc2"].iloc[1:],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.004,
    )
    sns.scatterplot(
        data=df.query("time_hours == time_hours.min()"),
        x="pc1",
        y="pc2",
        marker="o",
        color="red",
        alpha=0.7,
        lw=0,
        ax=ax,
        s=50,
        legend=False,
    )
    sns.scatterplot(
        data=df.query("time_hours == time_hours.max()"),
        x="pc1",
        y="pc2",
        marker="x",
        color="red",
        alpha=0.7,
        lw=2,
        ax=ax,
        s=50,
        legend=False,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    fig.savefig(
        out_subdir_integrated_tracks
        / f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    cmap = sns.color_palette("dark:red", as_cmap=True)
    angle_deg_to_color = lambda a: cmap(np.abs(a) / 180.0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="blue",
    )
    ax.quiver(
        df["pc1"].iloc[:-1],
        df["pc2"].iloc[:-1],
        df["dpc1"].iloc[1:],
        df["dpc2"].iloc[1:],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color=angle_deg_to_color(df["track_angular_deviation_deg"].iloc[1:]),
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    fig.savefig(
        out_subdir_integrated_tracks_hued
        / f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field_hued.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # # force garbage collection to keep memory
    # # free when creating plots from a loop
    # gc.collect()

    return


def get_vector_angles_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    my_shape: list[int],
) -> np.ndarray:
    """Get the angles of the vectors as a grid."""
    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )
    test_ang = get_vector_vector_angle_fast(vecs_grids, vecs_tracks)
    test_ang_arr = test_ang.reshape((50, 50, 50))
    angles = test_ang_arr[slice_indexes].reshape(my_shape)
    return angles


def get_vector_dot_products_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    my_shape: list[int],
) -> np.ndarray:
    """Get the dot products of the vectors as a grid."""
    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )
    test_dot_prod = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    test_dot_prod_arr = test_dot_prod.reshape((50, 50, 50))
    dot_prod = test_dot_prod_arr[slice_indexes].reshape(my_shape)
    return dot_prod


# ## NOTE CODE FOR DEV ONLY
# dataset_name = "20241120_20X"

# live_seg_feats_df = pd.read_csv(
#     r"C:\Users\serge.parent\Documents\projects\cellsmap\results\2025-07-16\make_seg_feats_manifest\segmentation_features_manifests\20241120_20X_live_segmentation_features.tsv",
#     sep="\t",
# )
# diffae_tracking_df = pd.read_parquet(
#     r"C:\Users\serge.parent\Documents\projects\cellsmap\results\models\diffae_04_10\20241120_20X\predict_20241120_20X_diffae_04_10_tracked_crop_features.parquet"
# )

# merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)
# ## NOTE END OF DEV CODE


## NOTE THIS NEEDS TO BE BROKEN UP INTO SMALLER AND MORE
## INFORMATIVE FUNCTIONS
def process_dataset(dataset_name: str, make_integrated_plots: bool = True) -> None:
    logger.info(f"Processing dataset: {dataset_name}")

    out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)
    configure_logging(out_subdir, logger, verbose=True)

    # load the tables
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name)

    # filter the merged table
    merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

    # remove any rows that were not evaluated by the model and thus have no mlflow_id
    merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

    # keep only the columns that are needed for the analysis to reduce memory usage
    cols_to_keep = [
        "dataset_name",
        "position",
        "position_as_str",
        "track_id",
        "label",
        "crop_index",
        "mlflow_id",
        "model_name",
        "image_index",
        "frame_number",
        "time_hours",
        "time_minutes",
        "track_duration",
    ] + [col for col in merged_feats_df.columns if "feat" in col]

    merged_feats_df = merged_feats_df[cols_to_keep]

    # fit the PCA (uses the reference datasets)
    pca = fit_pca()

    # read in the grid crop-based diffae features
    model_name = merged_feats_df["model_name"].unique()[0]
    model_config = load_model_config(model_name)
    model_manifest = get_model_manifest(dataset_name, model_config)
    diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

    # add the PC columns to the track-based DiffAE table
    # (the grid-based DiffAE table already has them, but
    # but I believe that the columns are named "feat_0",
    # "feat_1", etc. when they should be named "pc1",
    # "pc2", etc.)
    merged_feats_df = project_manifest_to_pcs(merged_feats_df, pca)

    # use the full set of datasets to be analyzed for the bounds
    datasets_for_bounds = [
        "20241120_20X",
        "20250409_20X",
        "20241217_20X",
        "20250428_20X",
        "20250319_20X",
        "20250326_20X",
    ]

    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in datasets_for_bounds
    ]
    bounds = ddff.set_3d_bounds_from_data(model_manifest_list, pca)

    # This takes about 1 minute to run
    precomputed_trajectories_path = out_subdir / f"{dataset_name}_traj_grids.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for grid-based crops...")
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        df=diffae_grid_crops,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the grid-based crops...")
        np.save(precomputed_trajectories_path, traj_grids)

    # This takes about 5 minutes to run
    precomputed_trajectories_path = out_subdir / f"{dataset_name}_traj_tracks.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for tracks-based crops...")
    traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(
        df=merged_feats_df,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the track-based crops...")
        np.save(precomputed_trajectories_path, traj_tracks)

    yvalids_grids, zvalids_grids = get_valid_slice_indexes(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )

    slice_indexes = (zvalids_grids, yvalids_grids)[0]
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    v1_grids, v2_grids, v3_grids = flow_field_dict_grids["vectors"]
    g1_grids, g2_grids, g3_grids = flow_field_dict_grids["grid"]
    v1_tracks, v2_tracks, v3_tracks = flow_field_dict_tracks["vectors"]
    g1_tracks, g2_tracks, g3_tracks = flow_field_dict_tracks["grid"]

    angles = get_vector_angles_as_grid(
        v1_grids,
        v2_grids,
        v3_grids,
        v1_tracks,
        v2_tracks,
        v3_tracks,
        slice_indexes,
        my_shape,
    )

    dot_prod = get_vector_dot_products_as_grid(
        v1_grids,
        v2_grids,
        v3_grids,
        v1_tracks,
        v2_tracks,
        v3_tracks,
        slice_indexes,
        my_shape,
    )

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the full resolution:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_full_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,
        ds=1,
        scale=60,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the standard/default resolution:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_partial_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,
    )
    ax.scatter(
        traj_grids[-1, 0],
        traj_grids[-1, 1],
        s=250,
        color="cyan",
        marker="*",
        lw=1,
        edgecolor="darkblue",
        zorder=10,
    )
    ax.scatter(
        traj_tracks[-1, 0],
        traj_tracks[-1, 1],
        s=250,
        color="yellow",
        marker="*",
        lw=1,
        edgecolor="darkred",
        zorder=10,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the angular deviation between the grid and cell-centric crop-based
    # flow field vectors:
    out_path = out_subdir / f"{dataset_name}_vecvec_angles.png"
    grid_vs_track_vec_angle_hist2d(angles, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Plot the dot product between the grid and cell-centric crop-based
    out_path = out_subdir / f"{dataset_name}_vecvec_dot_products.png"
    grid_vs_track_vec_dot_prod_hist2d(dot_prod, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Plot flow fields overlaid on the PC1 vs PC2
    # histograms to get an idea of where the flow
    # fields have the most data to work with
    out_path = out_subdir / f"{dataset_name}_grid_crops_pc1_pc2_hist2d.png"
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=diffae_grid_crops,
        x="pc1",
        y="pc2",
        bins=50,
        cmap="Blues",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out_path = out_subdir / f"{dataset_name}_tracked_crops_pc1_pc2_hist2d.png"
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=merged_feats_df,
        x="pc1",
        y="pc2",
        bins=50,
        cmap="Reds",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_tracks, v2_tracks),
        grid=(g1_tracks, g2_tracks),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Compare the angles between grid crop PC vectors
    # and the PC vectors of a single track
    merged_feats_df["dpc1"] = merged_feats_df.groupby("crop_index")["pc1"].diff()
    merged_feats_df["dpc2"] = merged_feats_df.groupby("crop_index")["pc2"].diff()
    merged_feats_df["dt"] = merged_feats_df.groupby("crop_index")["time_minutes"].diff()

    get_approx_grid_bin = lambda pc1_pc2_arr: get_approx_point_from_grid(
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,
    )
    get_approx_grid_bin_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_bin(df.to_numpy()), index=df.index
    )

    get_approx_grid_vec = lambda pc1_pc2_arr: get_approx_vec_from_grid(
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,
    )
    get_approx_grid_vec_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_vec(df.to_numpy()), index=df.index
    )

    merged_feats_df[["approx_bin_pc1", "approx_bin_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_bin_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )
    merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_vec_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )

    merged_feats_df["track_angle_deviation_rad"] = get_vector_vector_angle_fast(
        merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]].values,
        merged_feats_df[["dpc1", "dpc2"]].values,
    )

    merged_feats_df["track_angular_deviation_deg"] = merged_feats_df[
        "track_angle_deviation_rad"
    ].transform(np.rad2deg)

    merged_feats_df["pc1_pc2_vec_mag"] = np.linalg.norm(
        merged_feats_df[["dpc1", "dpc2"]].values, axis=1
    )

    test = (
        merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"])[
            ["track_angular_deviation_deg", "pc1_pc2_vec_mag"]
        ]
        .agg("mean")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=test, x="track_angular_deviation_deg", binwidth=1, ax=ax)
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Counts")
    fig.savefig(
        out_subdir / f"{dataset_name}_angular_deviation_histogram.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=test, x="track_angular_deviation_deg", y="pc1_pc2_vec_mag", binwidth=(1, None), ax=ax)  # type: ignore[arg-type]
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Track PC1-PC2\nvector magnitude")
    fig.savefig(
        out_subdir / f"{dataset_name}_angular_deviation_vs_mag_histogram.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if make_integrated_plots:

        merged_feats_df = merged_feats_df.query("track_duration > 120")
        groups = merged_feats_df.query("track_duration > 120").groupby(
            ["dataset_name", "position_as_str", "crop_index"]
        )
        # group_names = list(merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"]).groups.keys())
        # group_names = [nm for nm, df in merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"])]

        i = 0
        for nm, df in tqdm(groups, desc=dataset_name):
            # for nm in tqdm(group_names, desc=dataset_name):
            # df = merged_feats_df[(merged_feats_df[["dataset_name", "position_as_str", "crop_index"]] == nm).all(axis=1)]  # type: ignore
            ds_nm, pos, tid = nm
            plot_pc_integrated_track(
                dataset_name=str(ds_nm),
                position_name=str(pos),
                track_id=tid,
                df=df,
                v1_grids=v1_grids,
                v2_grids=v2_grids,
                g1_grids=g1_grids,
                g2_grids=g2_grids,
                slice_indexes=slice_indexes,
                out_subdir=out_subdir,
            )
            i += 1
            if i % 100 == 0:
                # force garbage collection to keep memory free when
                # creating plots from a loop every 100th iteration
                gc.collect()

    # force garbage collection to keep memory free
    # when this dataset is done being processed
    gc.collect()
    return


dataset_name_list = load_dataset_collection_config("pca_reference").datasets[::-1]
# dataset_name_list = ["20241217_20X"]

for dataset_name in dataset_name_list:
    logger.info(f"Processing {dataset_name}...")
    process_dataset(dataset_name, make_integrated_plots=True)

# create a test flow field and test set of vectors
# to check that the angular deviation calculation
# works as expected
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


cmap = sns.color_palette("dark:red", as_cmap=True)
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
    get_output_path(Path(__file__).stem) / "get_angular_deviation_deg_test.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)
