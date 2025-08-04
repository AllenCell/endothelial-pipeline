import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.endo_pipeline.configs import load_dataset_config
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import load_dataframe, load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import (
    compute_extrapolated_vector_field,
    solve_ddff_ode,
)
from src.endo_pipeline.library.analyze.diffae_manifest import (
    add_description_column,
    get_traj_and_diff,
)
from src.endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from src.endo_pipeline.library.analyze.numerics import get_bins
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
    diffae_tracking_df["crop_index"] = diffae_tracking_df.groupby(
        ["position", "track_id"], as_index=False
    ).ngroup()
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


def get_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> pd.DataFrame:

    logging.debug(f"Loading dataset config file for dataset: {dataset_name}...")
    dataset_config = load_dataset_config(dataset_name)

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_track_manifest = load_dataframe_manifest("diffae_tracking_integration")
    diffae_track_location = get_dataframe_location_for_dataset(diffae_track_manifest, dataset_name)
    diffae_tracking_df = load_dataframe(diffae_track_location)

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
    bounds: List,
) -> tuple[np.ndarray, dict]:

    # load default config, get kernel params
    dynamics_config = load_dynamics_config("default")
    kernel_params = dynamics_config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
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

    # solve IVP, get back trajectory
    print("Trying to solve ODE...")
    traj = solve_ddff_ode(flow_field_dict, init, time_span)
    print("ODE solved.")

    return traj, flow_field_dict
