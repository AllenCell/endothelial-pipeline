import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import load_dataset_config
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.analyze.optical_flow_calculator import (
    one_direction_vector_field_example,
)
from src.endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar

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


def get_diffae_feats_liveseg_feats_merged_table(
    dataset_name: str, filtered: bool = False
) -> pd.DataFrame:

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

    if filtered:
        # filter the merged table
        merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

        # remove any rows that were not evaluated by the model and thus have no mlflow_id
        merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

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
) -> np.ndarray:
    """Get the dot products of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )
    test_dot_prod = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    test_dot_prod_arr = test_dot_prod.reshape((50, 50, 50))
    dot_prod = test_dot_prod_arr[slice_indexes].reshape(my_shape)
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
        out_dir / "get_angular_deviation_deg_test.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return
