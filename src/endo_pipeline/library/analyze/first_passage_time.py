"""Methods for calculating and analyzing first passage time"""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from odrpack import odr_fit
from scipy.stats import pearsonr

from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_track_length,
    filter_dataframe_to_binned_value,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_track_duration_to_dataframe,
)
from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest import DataframeManifest
from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.first_passage_time import (
    FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
)
from endo_pipeline.settings.literal_types import PatchTypeLiteral
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def load_filtered_trajectory_df_for_first_passage_time_workflow(
    dataset_name: str,
    patch_type: PatchTypeLiteral,
    minimum_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
) -> pd.DataFrame:
    """
    Load and filter the trajectory dataframe for first passage time analysis.

    Trajectories are loaded from the appropriate manifest for the given patch
    type, filtered to steady-state timepoints, and then filtered to only include
    tracks that meet the minimum track length requirement.

    Parameters
    ----------
    dataset_name
        Name of the dataset to load trajectories for.
    patch_type
        Whether to load grid-based or cell-centered crops.
    minimum_track_length
        Minimum number of timepoints a track must span to be included in the output.

    Returns
    -------
    :
        DataFrame containing the filtered trajectories with dynamics feature columns
        and track metadata.
    """

    if patch_type == "grid_based":
        dynamics_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    elif patch_type == "cell_centered":
        dynamics_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME)
    else:
        raise ValueError(f"Unsupported patch type: {patch_type}")

    dynamics_loc = get_dataframe_location_for_dataset(dynamics_manifest, dataset_name)
    trajectories_df_delayed = load_dataframe(dynamics_loc, delay=True)
    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        *DYNAMICS_COLUMN_NAMES,
    ]
    trajectories_df = trajectories_df_delayed[columns_to_compute].compute().reset_index()

    # the loaded grid-based dynamics dataframe is disordered by default so
    # sort the grid-based dynamics dataframe by crop index and timepoint
    trajectories_df = trajectories_df.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

    # filter the grid-based dynamics dataframe to only include timepoints from steady state
    dataset_config = load_dataset_config(dataset_name)
    trajectories_df = filter_dataframe_to_steady_state(
        dataframe=trajectories_df, dataset_config=dataset_config
    )

    # add the track durations post-filtering
    trajectories_df = add_track_duration_to_dataframe(
        dataframe=trajectories_df,
        grouping_columns=[Column.CROP_INDEX],
        time_column=Column.TIMEPOINT,
    )

    # filter trajectories to only include long ones
    trajectories_df = filter_dataframe_by_track_length(
        dataframe=trajectories_df, minimum_track_length=minimum_track_length
    )

    return trajectories_df


def add_distance_to_fixed_points_columns(
    trajectory_df: pd.DataFrame,
    fixed_point_df: pd.DataFrame,
    trajectory_columns: list[Column.DiffAEData | str],
    fixed_point_columns: list[Column.DiffAEData | str] | None = None,
    polar_angle_period: float | None = None,
) -> pd.DataFrame:
    """
    Compute distance from each point in the trajectory to the fixed points.

    This distance gets added as a new column to the trajectory dataframe for
    each fixed point, along with the signed difference along each axis
    (e.g. theta, r, rho) from each fixed point.

    Parameters
    ----------
    trajectory_df
        DataFrame containing the trajectory points.
    fixed_point_df
        DataFrame containing the fixed points.
    trajectory_columns
        List of column names in trajectory_df to use for distance computation.
    fixed_point_columns
        List of column names in fixed_point_df to use for distance computation.
        Expected to be in the same order as trajectory_columns.
        If None, the trajectory_columns will be used.
    column_suffix
        Suffix to append to the new distance-from-fixed-point columns.
    polar_angle_period
        The period to use for the polar angle variable when computing differences, if applicable.
        If None, the default POLAR_ANGLE_PERIOD will be used. The other expected
        value for this parameter would be 2 * np.pi.

    Returns
    -------
    :
        DataFrame containing the distances to the nearest fixed point for each trajectory point.
    """

    if fixed_point_columns is None:
        fixed_point_columns = trajectory_columns

    # determine distance from each fixed point over time and add to the dataframe, along
    # with the signed difference along each axis (e.g. theta, r, rho) from each fixed point
    polar_angle_period = POLAR_ANGLE_PERIOD if polar_angle_period is None else polar_angle_period

    for i in fixed_point_df.index:
        fpt = fixed_point_df.loc[i]

        for j, col in enumerate(fixed_point_columns):
            # this lambda function computes the signed difference from the fixed point for a given
            # column, taking into account the periodicity of the polar angle variable if applicable
            diff_func = lambda x, fpt=fpt, col=col: (
                np.mod(x - fpt[col] + polar_angle_period / 2, polar_angle_period)
                - polar_angle_period / 2
                if Column.DiffAEData.POLAR_ANGLE.value in col
                else (x - fpt[col])
            )
            trajectory_df[ColumnTemplate.DISTANCE_FROM_FIXED_POINT_1D_SIGNED_PREFIX % (i, col)] = (
                diff_func(trajectory_df[trajectory_columns[j]])
            )

        dynamics_diff_columns = [
            ColumnTemplate.DISTANCE_FROM_FIXED_POINT_1D_SIGNED_PREFIX % (i, col)
            for col in fixed_point_columns
        ]
        trajectory_df[ColumnTemplate.DISTANCE_FROM_FIXED_POINT % i] = np.linalg.norm(
            trajectory_df[dynamics_diff_columns], axis=1
        )

    return trajectory_df


def load_dataframes_for_first_passage_time_analysis(
    dataset_name: str, minimum_track_length: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process feature dataframes for first passage time analysis and
    visualization.

    Load both the grid-based and cell-centered feature dataframes and filter to
    only include trajectories of at least the specified minimum track length as
    well as the fixed points. Then, add columns for distance from each point in
    the trajectory to the fixed points.

    Parameters
    ----------
    dataset_name
        Name of the dataset to load trajectory data for.
    minimum_track_length
        Minimum track length to filter trajectories.

    Returns
    -------
    :
        Grid-based features, cell-centered features, and fixed point dataframes.
    """

    # load the dynamics features from the grid-based and track-based dataframes
    traj_df_grid = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        patch_type="grid_based",
        minimum_track_length=minimum_track_length,
    )
    traj_df_grid[Column.SegData.TIME_HRS] = traj_df_grid[Column.TIMEPOINT] * TIME_STEP_IN_HOURS

    traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        patch_type="cell_centered",
        minimum_track_length=minimum_track_length,
    )
    traj_df_tracked[Column.SegData.TIME_HRS] = (
        traj_df_tracked[Column.TIMEPOINT] * TIME_STEP_IN_HOURS
    )

    # load the flow field dictionaries and fixed points
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    fp_cluster_mean_cols = [
        ColumnTemplate.BOOTSTRAP_CLUSTER_MEAN % col for col in DYNAMICS_COLUMN_NAMES
    ]

    # add the distances from the fixed points for the grid-based trajectories
    traj_df_grid = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_grid,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        fixed_point_columns=fp_cluster_mean_cols,
    )

    # add the distances from the fixed points for the track-based trajectories
    traj_df_tracked = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_tracked,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        fixed_point_columns=fp_cluster_mean_cols,
    )

    return traj_df_grid, traj_df_tracked, fixed_points_df


def build_first_passage_time_bins(
    traj_df_grid: pd.DataFrame,
    traj_df_tracked: pd.DataFrame,
    bin_size_theta_deg: float | None = None,
    bin_size_radius: float | None = None,
    bin_size_rho: float | None = None,
) -> tuple[list, list, dict, dict]:
    """
    Build bins for first passage time analysis.

    Parameters
    ----------
    traj_df_grid
        Dataframe of grid-based trajectories
    traj_df_tracked
        Dataframe of cell-centered trajectories
    bin_size_theta_deg
        Bin size for polar theta in degrees.
    bin_size_radius
        Bin size for polar r feature.
    bin_size_rho
        Bin size for rho feature.

    Returns
    -------
    :
        List of bin edges and centers and dictionaries of bin sizes and limits.
    """

    bin_sizes = {
        Column.DiffAEData.POLAR_ANGLE: (
            np.deg2rad(bin_size_theta_deg) if bin_size_theta_deg is not None else np.deg2rad(15)
        ),
        Column.DiffAEData.POLAR_RADIUS: (bin_size_radius if bin_size_radius is not None else 0.25),
        Column.DiffAEData.PC3_FLIPPED: bin_size_rho if bin_size_rho is not None else 0.5,
    }

    # get the data limits for each feature to be binned
    bin_limits: dict = {}
    for col in DYNAMICS_COLUMN_NAMES:
        col_min = min(traj_df_grid[col].min(), traj_df_tracked[col].min())
        col_max = max(traj_df_grid[col].max(), traj_df_tracked[col].max())
        bin_limits[col] = (col_min, col_max)

    # adjust the bin_limits if the feature has a defined range (e.g. for angles)
    defined_bin_limits = {
        Column.DiffAEData.POLAR_ANGLE: (0, np.pi),
        Column.DiffAEData.POLAR_RADIUS: (0, None),
        Column.DiffAEData.PC3_FLIPPED: (None, None),
    }
    for col in DYNAMICS_COLUMN_NAMES:
        if col in defined_bin_limits:
            bin_limits[col] = adjust_limits_from_bin_size(
                data_min_max=bin_limits[col],
                defined_min_max=defined_bin_limits[col],
                bin_size=bin_sizes[col],
            )

    bin_widths = tuple(float(bin_sizes[col]) for col in DYNAMICS_COLUMN_NAMES)
    bin_limits_list = [bin_limits[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_edges, bin_centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits_list)

    return bin_edges, bin_centers, bin_sizes, bin_limits


def add_first_passage_time_column(
    fixed_point_index: int,
    trajectory_df: pd.DataFrame,
    threshold: float,
    time_column: str,
) -> pd.DataFrame:
    """
    Add the time of first passage for each track in the trajectory dataframe.

    The first passage time is computed as the first timepoint (specified by
    `Column.TIMEPOINT`) at which distance from the fixed point is less than or
    equal to the given threshold for each track (grouped by
    `Column.CROP_INDEX`).

    Parameters
    ----------
    fixed_point_index
        Index of the fixed point corresponding to the row being used to compute
        first passage time.
    trajectory_df
        DataFrame containing the trajectory points.
    threshold
        Threshold value to determine the first passage.
    time_column
        Column name in trajectory_df corresponding to the time variable (e.g.
        `Column.TIMEPOINT` or `Column.SegData.TIME_HRS`).

    Returns
    -------
    :
        DataFrame containing the first passage time for each track.
    """

    # compute where the trajectory first passes the threshold distance to the fixed point
    column = ColumnTemplate.DISTANCE_FROM_FIXED_POINT % fixed_point_index
    new_column_name = ColumnTemplate.FIRST_PASSAGE_TIME_DISTANCE % fixed_point_index
    trajectory_df[new_column_name] = (
        trajectory_df.groupby(Column.CROP_INDEX)
        .apply(
            lambda grp: pd.DataFrame(
                {new_column_name: grp[time_column][grp[column] <= threshold].min()},
                index=grp.index,
            ),
            include_groups=False,
        )
        .droplevel(0)
    )

    # trim all trajectories to only include timepoints prior to reaching the fixed point
    trajectory_df = trajectory_df[
        trajectory_df.apply(
            lambda row, fp_idx=fixed_point_index, time_column=time_column: row[time_column]
            < row[ColumnTemplate.FIRST_PASSAGE_TIME_DISTANCE % fp_idx],
            axis=1,
        )
    ]

    # compute the time to the first passage time from each timepoint
    trajectory_df[ColumnTemplate.TIME_TO_FIXED_POINT % fixed_point_index] = (
        trajectory_df[ColumnTemplate.FIRST_PASSAGE_TIME_DISTANCE % fixed_point_index]
        - trajectory_df[time_column]
    )

    return trajectory_df


def filter_to_trajectories_reaching_fixed_point(
    traj_df_grid: pd.DataFrame,
    traj_df_tracked: pd.DataFrame,
    fixed_point_index: int,
    fixed_point_radius_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter dataframes to only include trajectories that reach the fixed point.

    Parameters
    ----------
    traj_df_grid
        Dataframe of grid-based trajectories with distance to fixed point
        columns.
    traj_df_tracked
        Dataframe of cell-centered trajectories with distance to fixed point
        columns.
    fixed_point_index
        Index of the fixed point to filter trajectories by.
    fixed_point_radius_threshold
        Distance threshold from the fixed point below which a trajectory
        timepoint is considered to have reached the fixed point.

    Returns
    -------
    :
        Filtered dataframes with only trajectories that reach the fixed point.
    """

    distance_from_fixed_point_column = ColumnTemplate.DISTANCE_FROM_FIXED_POINT % fixed_point_index
    is_at_fixed_point_column = ColumnTemplate.IS_AT_FIXED_POINT % fixed_point_index
    traj_reached_fixed_point_column = (
        ColumnTemplate.TRAJECTORY_REACHED_FIXED_POINT % fixed_point_index
    )

    # Mark each timepoint as "at the fixed point" if it is within the radius
    # threshold, then propagate that flag to all timepoints in the trajectory
    traj_df_grid[is_at_fixed_point_column] = (
        traj_df_grid[distance_from_fixed_point_column] <= fixed_point_radius_threshold
    )
    traj_df_tracked[is_at_fixed_point_column] = (
        traj_df_tracked[distance_from_fixed_point_column] <= fixed_point_radius_threshold
    )

    traj_df_grid[traj_reached_fixed_point_column] = traj_df_grid.groupby(Column.CROP_INDEX)[
        is_at_fixed_point_column
    ].transform(any)
    traj_df_tracked[traj_reached_fixed_point_column] = traj_df_tracked.groupby(Column.CROP_INDEX)[
        is_at_fixed_point_column
    ].transform(any)

    # Filter to only trajectories that eventually reach the fixed point
    traj_df_grid_sub = traj_df_grid[traj_df_grid[traj_reached_fixed_point_column]]
    traj_df_tracked_sub = traj_df_tracked[traj_df_tracked[traj_reached_fixed_point_column]]

    return traj_df_grid_sub, traj_df_tracked_sub


def compute_first_passage_time_statistics_for_one_bin(
    bin_index: int,
    bin_center: Sequence[float],
    bin_edges: list[np.ndarray],
    trajectory_df: pd.DataFrame,
    time_to_first_passage_col_name: str,
    feature_column_names: list[str],
) -> pd.DataFrame:
    """
    Compute summary statistics for the first passage time for all trajectories that fall
    within a single spatial bin.

    Parameters
    ----------
    bin_index
        Integer index identifying this bin, assigned as a column in the returned dataframe.
    bin_center
        Coordinates of the bin centre along each feature dimension.
    bin_edges
        List of arrays, one per feature dimension, specifying the left and right edges of
        this bin.
    trajectory_df
        DataFrame containing the trajectory data with a first-passage-time column.
    time_to_first_passage_col_name
        Name of the column in ``trajectory_df`` that stores the time-to-first-passage value.
    feature_column_names
        List of feature column names used to filter trajectories to this bin.

    Returns
    -------
    :
        Single-row DataFrame containing ``pd.describe``-style summary statistics for the
        first passage times in this bin, with the bin index appended as a column.
    """
    trajectory_df_one_bin = filter_dataframe_to_binned_value(
        dataframe=trajectory_df,
        columns=feature_column_names,
        values=bin_center,
        bin_edges=bin_edges,
    )
    first_passage_time_stats_df = (
        trajectory_df_one_bin[time_to_first_passage_col_name].describe().to_frame().T
    )
    # compute standard error of the mean and add it to the dataframe
    first_passage_time_stats_df["sem"] = first_passage_time_stats_df["std"] / np.sqrt(
        first_passage_time_stats_df["count"]
    )
    # new column names use empty string for the second placeholder because the
    # dataframe will be merged later, which applies the actual value of the
    # placeholder using the pandas dataframe merge 'suffixes' option
    new_col_names = {
        col: ColumnTemplate.FIRST_PASSAGE_TIME_METRIC % (col, "")
        for col in first_passage_time_stats_df.columns
    }
    first_passage_time_stats_df.rename(columns=new_col_names, inplace=True)

    first_passage_time_stats_df = first_passage_time_stats_df.assign(bin_index=bin_index)

    return first_passage_time_stats_df


def compute_first_passage_time_statistics_for_bins(
    bin_centers: list[np.ndarray],
    bin_edges: list[np.ndarray],
    trajectory_df: pd.DataFrame,
    time_to_first_passage_col_name: str,
    feature_column_names: list[str],
) -> pd.DataFrame:
    """
    Compute first passage time summary statistics for every bin in the feature-space grid.

    Parameters
    ----------
    bin_centers
        List of 1-D arrays, one per feature dimension, containing the bin centre coordinates.
    bin_edges
        List of 1-D arrays, one per feature dimension, containing the bin edge coordinates.
    trajectory_df
        DataFrame containing the trajectory data with a first-passage-time column.
    time_to_first_passage_col_name
        Name of the column in ``trajectory_df`` that stores the time-to-first-passage value.
    feature_column_names
        List of feature column names used to assign trajectories to bins.

    Returns
    -------
    :
        DataFrame with one row per bin containing summary statistics for the first passage
        time and the corresponding bin centre and edge coordinates.
    """
    # create a meshgrid of the bin centers and edges for iterating through the bins
    bin_centers_mesh = np.meshgrid(*bin_centers, indexing="ij")
    bin_centers_all = list(zip(*[arr.ravel() for arr in bin_centers_mesh], strict=True))
    bin_indices_nd, _ = list(zip(*np.ndenumerate(bin_centers_mesh[0]), strict=True))

    results = []
    for bin_index, bin_center in enumerate(bin_centers_all):
        # I tried to avoid doing nd indexing because it gets a little hair, but
        # it seems necessary to get the correct bin edges for each bin when
        # filtering the trajectories to each bin
        # the reason we can use + 2 below instead of + 1 is because the bin_edges_mesh
        # includes the right edge of the last bin, so it has one more element than
        # the bin_centers_mesh along each dimension
        bin_index_nd = bin_indices_nd[bin_index]
        bin_e = []
        for dim, idx in enumerate(bin_index_nd):
            bin_e.append(tuple(bin_edges[dim][idx : idx + 2]))
        first_passage_time_stats_df = compute_first_passage_time_statistics_for_one_bin(
            bin_index=bin_index,
            bin_center=bin_center,
            bin_edges=bin_edges,
            trajectory_df=trajectory_df,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=feature_column_names,
        )
        first_passage_time_stats_df[Column.VectorField.BIN_CENTER] = [bin_center]
        first_passage_time_stats_df[Column.VectorField.BIN_EDGES] = [bin_e]

        results.append(first_passage_time_stats_df)

    first_passage_time_stats_df = pd.concat(results, ignore_index=True)

    return first_passage_time_stats_df


def compute_first_passage_time_statistics(
    traj_df_grid_sub: pd.DataFrame,
    traj_df_tracked_sub: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
    bin_edges: list[np.ndarray],
    bin_centers: list[np.ndarray],
) -> pd.DataFrame:
    """
    Compute first passage time statistics per bin.

    Parameters
    ----------
    traj_df_grid_sub
        Dataframe of grid-based trajectories that reach the fixed point, with
        distance to fixed point columns and first passage time column.
    traj_df_tracked_sub
        Dataframe of cell-centered trajectories that reach the fixed point, with
        distance to fixed point columns and first passage time column.
    dataset_name
        Name of the dataset being analyzed, used for labeling the output
        dataframe.
    fixed_point_index
        Index of the fixed point that the trajectories reach, used for labeling
        the first passage time column in the output dataframe.
    bin_edges
        Edges of the bins in feature space to compute statistics for.
    bin_centers
        Centers of the bins in feature space to compute statistics for.

    Returns
    -------
    :
        Dataframe with first passage time statistics.
    """

    # For each bin (across all steady-state timepoints), compute the mean,
    # median, and standard deviation of first-passage times for the trajectories
    time_to_first_passage_col_name = ColumnTemplate.TIME_TO_FIXED_POINT % fixed_point_index

    fpt_stats_df_grid = compute_first_passage_time_statistics_for_bins(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        trajectory_df=traj_df_grid_sub,
        time_to_first_passage_col_name=time_to_first_passage_col_name,
        feature_column_names=list(DYNAMICS_COLUMN_NAMES),
    )
    fpt_stats_df_tracked = compute_first_passage_time_statistics_for_bins(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        trajectory_df=traj_df_tracked_sub,
        time_to_first_passage_col_name=time_to_first_passage_col_name,
        feature_column_names=list(DYNAMICS_COLUMN_NAMES),
    )

    # merge the grid and tracked first passage time stats dataframes
    first_passage_time_stats_df = merge_grid_and_tracked_first_passage_time_statistics_dataframes(
        fpt_stats_df_grid=fpt_stats_df_grid,
        fpt_stats_df_tracked=fpt_stats_df_tracked,
        dataset_name=dataset_name,
        fixed_point_index=fixed_point_index,
    )

    return first_passage_time_stats_df


def compute_first_passage_time_parameter_sweep(
    fixed_point_index: int, trajectory_df: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """
    Run a parameter sweep over first passage time distance thresholds and return aggregated
    summary statistics for each threshold value.

    For each threshold, trajectories that reach within that distance of the fixed point are
    identified, the first passage time is computed, and summary statistics (mean, std, etc.)
    are collected. The fraction of trajectories that approached the fixed point under each
    threshold is also recorded.

    Parameters
    ----------
    fixed_point_index
        Index of the fixed point to compute first passage times for.
    trajectory_df
        DataFrame containing the trajectory data with pre-computed distance-from-fixed-point
        columns.
    thresholds
        Sequence of distance threshold values to sweep over.

    Returns
    -------
    :
        DataFrame with one row per threshold value containing aggregated first passage time
        statistics and the fraction of trajectories that approached the fixed point.
    """
    sweep_results: list = []
    for thresh in thresholds:
        trajectory_df_one_param = trajectory_df.copy()
        trajectory_df_one_param[Column.NUM_TRAJECTORIES_BEFORE_FPT_FILTER] = trajectory_df[
            Column.CROP_INDEX
        ].nunique()
        trajectory_df_one_param = add_first_passage_time_column(
            fixed_point_index=fixed_point_index,
            trajectory_df=trajectory_df_one_param,
            threshold=thresh,
            time_column=Column.SegData.TIME_HRS,
        )
        trajectory_df_one_param[Column.NUM_TRAJECTORIES_AFTER_FPT_FILTER] = trajectory_df_one_param[
            Column.CROP_INDEX
        ].nunique()
        trajectory_df_one_param = trajectory_df_one_param.assign(
            **{Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
        )
        trajectory_df_one_param = trajectory_df_one_param.assign(
            **{Column.VectorField.FPT_DISTANCE_THRESHOLD: thresh}
        )
        sweep_results.append(trajectory_df_one_param)

    fpt_param_sweep_df = pd.concat(sweep_results, ignore_index=True)

    # compute the summary statistics on the first passage time parameter sweep
    first_passage_time_col = ColumnTemplate.TIME_TO_FIXED_POINT % fixed_point_index
    fpt_param_sweep_agg_df = (
        fpt_param_sweep_df.groupby(Column.VectorField.FPT_DISTANCE_THRESHOLD)[
            first_passage_time_col
        ]
        .agg("describe")
        .reset_index(drop=False)
    )

    # also compute the fraction of trajectories that approached the fixed point for each
    # parameter combination to see how the fixed point distance threshold affects the
    # number of trajectories that are considered to have reached the fixed point
    fpt_param_sweep_df[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP] = (
        fpt_param_sweep_df[Column.NUM_TRAJECTORIES_AFTER_FPT_FILTER]
        / fpt_param_sweep_df[Column.NUM_TRAJECTORIES_BEFORE_FPT_FILTER]
    ) * 100

    num_traj_param_sweep_agg = (
        fpt_param_sweep_df.groupby(Column.VectorField.FPT_DISTANCE_THRESHOLD)[
            Column.VectorField.PERCENT_TRAJ_APPROACHED_FP
        ]
        .agg(lambda x: np.unique(x).item())
        .to_frame()
    ).reset_index(drop=False)
    num_traj_param_sweep_mapping = dict(
        zip(
            num_traj_param_sweep_agg[Column.VectorField.FPT_DISTANCE_THRESHOLD],
            num_traj_param_sweep_agg[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP],
            strict=True,
        )
    )
    fpt_param_sweep_agg_df[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP] = fpt_param_sweep_agg_df[
        Column.VectorField.FPT_DISTANCE_THRESHOLD
    ].map(num_traj_param_sweep_mapping)

    return fpt_param_sweep_agg_df


def merge_grid_and_tracked_first_passage_time_statistics_dataframes(
    fpt_stats_df_grid: pd.DataFrame,
    fpt_stats_df_tracked: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
) -> pd.DataFrame:
    """Merges the grid and tracked first passage time stats dataframes on the bin index,
    checking that the bin centers and edges are the same for both dataframes and
    dropping duplicate columns after the merge.

    Parameters
    ----------
    fpt_stats_df_grid
        DataFrame containing the first passage time stats for the grid-based
        trajectories, with columns for the bin index, bin centers, bin edges, and first
        passage time stats.
    fpt_stats_df_tracked
        DataFrame containing the first passage time stats for the track-based
        trajectories, with columns for the bin index, bin centers, bin edges, and first
        passage time stats.
    dataset_name
        Name of the dataset corresponding to the dataframes, used for error messages.
    fixed_point_index
        Index of the fixed point used for first passage time stats, used for error messages.

    Returns
    -------
    :
        A merged DataFrame containing the first passage time stats for both the grid-based
        and track-based trajectories, with duplicate columns for bin centers and edges dropped
        after verifying they are the same for both dataframes.
    """
    # merge the dataframes on the bin index, adding suffixes to duplicate columns

    fpt_stats_df = fpt_stats_df_grid.merge(
        fpt_stats_df_tracked,
        on=[Column.VectorField.BIN_INDEX],
        suffixes=("grid_based", "cell_centered"),
        validate="one_to_one",
    )
    fpt_stats_df = fpt_stats_df.assign(
        **{Column.DATASET: dataset_name, Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
    )

    # check that the bin centers and edges are the same for the grid and tracked dataframes
    bin_centers_close = np.allclose(
        np.array(
            list(zip(*fpt_stats_df[f"{Column.VectorField.BIN_CENTER}_grid_based"], strict=True))
        ),
        np.array(
            list(
                zip(
                    *fpt_stats_df[f"{Column.VectorField.BIN_CENTER}_cell_centered"],
                    strict=True,
                )
            )
        ),
    )
    bin_edges_close = np.allclose(
        np.array(
            list(zip(*fpt_stats_df[f"{Column.VectorField.BIN_EDGES}_grid_based"], strict=True))
        ),
        np.array(
            list(
                zip(
                    *fpt_stats_df[f"{Column.VectorField.BIN_EDGES}_cell_centered"],
                    strict=True,
                )
            )
        ),
    )
    if not bin_centers_close or not bin_edges_close:
        error_message = (
            "Bin centers or edges are not the same for grid and tracked dataframes for "
            f"dataset {dataset_name} and fixed point {fixed_point_index}. This may indicate an issue "
            "with the binning or merging of the dataframes."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # drop the duplicate bin center and edge columns from one of the dataframes
    # since they are the same and rename the columns to remove the suffixes
    fpt_stats_df = fpt_stats_df.drop(
        columns=[
            f"{Column.VectorField.BIN_CENTER}_cell_centered",
            f"{Column.VectorField.BIN_EDGES}_cell_centered",
        ]
    )
    fpt_stats_df = fpt_stats_df.rename(
        columns={
            f"{Column.VectorField.BIN_CENTER}_grid_based": Column.VectorField.BIN_CENTER,
            f"{Column.VectorField.BIN_EDGES}_grid_based": Column.VectorField.BIN_EDGES,
        }
    )

    return fpt_stats_df


def merge_grid_and_tracked_first_passage_time_parameter_sweep_dataframes(
    fpt_param_sweep_df_grid: pd.DataFrame,
    fpt_param_sweep_df_tracked: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
):
    """Merge the grid and tracked first passage time parameter sweep dataframes."""
    param_sweep_df = pd.merge(
        fpt_param_sweep_df_grid,
        fpt_param_sweep_df_tracked,
        on=Column.VectorField.FPT_DISTANCE_THRESHOLD,
        suffixes=("_grid_based", "_cell_centered"),
        how="outer",
        validate="one_to_one",
    )
    param_sweep_df = param_sweep_df.assign(
        **{Column.DATASET: dataset_name, Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
    )

    return param_sweep_df


def compute_first_passage_times_one_dataset(
    dataset_name: str,
    minimum_track_length: int,
    fixed_point_radius_threshold: float | None = None,
    bin_size_theta_deg: float | None = None,
    bin_size_radius: float | None = None,
    bin_size_rho: float | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Compute first passage times to the fixed points for grid-based and track-based trajectories.
    Also runs a parameter sweep over the first passage time threshold and saves the results as well.
    """

    logger = logging.getLogger(__name__)

    fpt_stats_df_list: list = []
    param_sweep_df_list: list = []

    traj_df_grid, traj_df_tracked, fixed_points_df = (
        load_dataframes_for_first_passage_time_analysis(
            dataset_name=dataset_name,
            minimum_track_length=minimum_track_length,
        )
    )

    # filter the fixed points to only the ones with higher confidence
    fixed_points_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
    ]
    fixed_points_df = fixed_points_df[fixed_points_df[Column.FIXED_POINT_STABILITY] == "stable"]

    if fixed_points_df.empty:
        logger.warning(f"No fixed points found for dataset {dataset_name}, skipping dataset.")
        fpt_stats_df_list.append(pd.DataFrame({Column.DATASET: [dataset_name]}))
        param_sweep_df_list.append(pd.DataFrame({Column.DATASET: [dataset_name]}))
        return fpt_stats_df_list, param_sweep_df_list

    # 1. bin (theta, r, rho) feature space define the bin sizes for each feature to be binned
    bin_edges, bin_centers, bin_sizes, bin_limits = build_first_passage_time_bins(
        traj_df_grid=traj_df_grid,
        traj_df_tracked=traj_df_tracked,
        bin_size_theta_deg=bin_size_theta_deg,
        bin_size_radius=bin_size_radius,
        bin_size_rho=bin_size_rho,
    )

    # 2. identify trajectories that pass a fixed point and filter df to only those trajectories
    # find if and when a trajectory reaches a fixed point
    thresholds = np.linspace(0, 1, 41)
    for fp_idx, fp_row in fixed_points_df.iterrows():
        # for now we will only look at first passage times to stable fixed points
        fp_stability = fp_row[Column.FIXED_POINT_STABILITY]
        if fp_stability != "stable":
            logger.info(
                f"Fixed point {fp_idx} in dataset {dataset_name} is not stable (stability = "
                f"{fp_stability}), skipping for first passage time analysis."
            )
            continue

        # if run_FPT_threshold_parameter_sweep:
        # run a parameter sweep of the first passage times using different
        # thresholds for what it means to have "reached" the fixed point
        fpt_param_sweep_df_grid = traj_df_grid.copy()
        fpt_param_sweep_df_tracked = traj_df_tracked.copy()
        fpt_param_sweep_df_grid = compute_first_passage_time_parameter_sweep(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_grid,
            thresholds=thresholds,
        )
        fpt_param_sweep_df_tracked = compute_first_passage_time_parameter_sweep(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_tracked,
            thresholds=thresholds,
        )
        parameter_sweep_df = merge_grid_and_tracked_first_passage_time_parameter_sweep_dataframes(
            fpt_param_sweep_df_grid=fpt_param_sweep_df_grid,
            fpt_param_sweep_df_tracked=fpt_param_sweep_df_tracked,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
        )
        parameter_sweep_df[Column.FIXED_POINT_STABILITY] = fp_stability

        traj_df_grid_sub, traj_df_tracked_sub = filter_to_trajectories_reaching_fixed_point(
            traj_df_grid=traj_df_grid,
            traj_df_tracked=traj_df_tracked,
            fixed_point_index=fp_idx,
            fixed_point_radius_threshold=fixed_point_radius_threshold,
        )

        # compute the timepoint at which each trajectory first reaches a fixed point
        traj_df_grid_sub = add_first_passage_time_column(
            fixed_point_index=fp_idx,
            trajectory_df=traj_df_grid_sub,
            threshold=fixed_point_radius_threshold,
            time_column=Column.SegData.TIME_HRS,
        )
        traj_df_tracked_sub = add_first_passage_time_column(
            fixed_point_index=fp_idx,
            trajectory_df=traj_df_tracked_sub,
            threshold=fixed_point_radius_threshold,
            time_column=Column.SegData.TIME_HRS,
        )

        # 3. for each bin (across all steady-state timepoints), compute stats
        # on first-passage times for the trajectories
        first_passage_time_stats_df = compute_first_passage_time_statistics(
            traj_df_grid_sub=traj_df_grid_sub,
            traj_df_tracked_sub=traj_df_tracked_sub,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )
        first_passage_time_stats_df[Column.FIXED_POINT_STABILITY] = fp_stability
        fp_dynamics_cols = [ColumnTemplate.FIXED_POINT % col for col in DYNAMICS_COLUMN_NAMES]
        first_passage_time_stats_df[fp_dynamics_cols] = fp_row[list(DYNAMICS_COLUMN_NAMES)]

        # add the bin sizes and bin limits to the dataframes for transparency
        for col in DYNAMICS_COLUMN_NAMES:
            first_passage_time_stats_df = first_passage_time_stats_df.assign(
                **{
                    ColumnTemplate.BIN_SIZE % col: bin_sizes[col],
                    ColumnTemplate.BIN_LIMITS
                    % col: [bin_limits[col]]
                    * len(first_passage_time_stats_df),
                }
            )

        fpt_stats_df_list.append(first_passage_time_stats_df)
        param_sweep_df_list.append(parameter_sweep_df)

    return fpt_stats_df_list, param_sweep_df_list


def fit_orthogonal_distance_regression(
    x: Sequence, y: Sequence, weight_x: Sequence | None = None, weight_y: Sequence | None = None
) -> tuple:
    """
    Fit a line to (x, y) data using orthogonal distance regression (ODR).

    Parameters
    ----------
    x
        Sequence of x-axis values.
    y
        Sequence of y-axis values.
    weight_x
        Optional sequence of weights for the x data (e.g. inverse variance).
    weight_y
        Optional sequence of weights for the y data (e.g. inverse variance).

    Returns
    -------
    :
        Tuple of (slope_fit, intercept_fit, slope_stdev, intercept_stdev,
        reduced_chi_squared, line_fit) from the ODR fit.
    """
    # use a line function for the ODR fit
    # p0 is the initial guess for the parameters of the function,
    # in this case the slope and intercept of the line
    # odr_fit requires this initial guess to be one object, which is why we
    # are using p0 instead of passing slope and intercept more explicitly
    line_func = lambda x, p0: p0[0] * x + p0[1]

    # need some initial guesses for the function parameters
    slope_initial_guess = 1
    intercept_initial_guess = 0

    line_fit = odr_fit(
        f=line_func,
        xdata=x,
        ydata=y,
        weight_x=weight_x,
        weight_y=weight_y,
        beta0=(slope_initial_guess, intercept_initial_guess),
        task="explicit-ODR",
    )
    slope_fit = line_fit.beta[0]
    intercept_fit = line_fit.beta[1]
    slope_stdev = line_fit.sd_beta[0]
    intercept_stdev = line_fit.sd_beta[1]
    reduced_chi_squared = line_fit.res_var

    return slope_fit, intercept_fit, slope_stdev, intercept_stdev, reduced_chi_squared, line_fit


def filter_first_passage_time_by_min_num_trajectories(
    fpt_stats_df: pd.DataFrame,
    min_num_traj_per_bin: int,
    metric_for_filter: Literal["mean", "median"],
) -> pd.DataFrame:
    """
    Filter a first passage time stats dataframe to only retain bins that have at least
    ``min_num_traj_per_bin`` trajectories and a non-NaN value for the chosen metric in
    both the grid and tracked columns.

    Parameters
    ----------
    fpt_stats_df
        DataFrame containing first passage time summary statistics per bin, as produced
        by :func:`compute_first_passage_time_stats_for_bins`.
    min_num_traj_per_bin
        Minimum number of trajectories required in a bin for it to be retained.
    metric_for_filter
        Which central-tendency metric to require to be non-NaN: ``"mean"`` or ``"median"``.

    Returns
    -------
    :
        Filtered DataFrame containing only bins that satisfy the trajectory-count and
        non-NaN requirements.
    """
    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_for_filter == "median" else metric_for_filter
    template = ColumnTemplate.FIRST_PASSAGE_TIME_METRIC

    # NaN values are unacceptable for the linear regression
    fpt_stats_df_no_nan = fpt_stats_df.copy().dropna(
        subset=[template % (metric, "grid_based"), template % (metric, "cell_centered")]
    )
    # keep only the bins with the minimum number of tracks per bin in them
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan[template % ("count", "grid_based")] >= min_num_traj_per_bin
    ]
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan[template % ("count", "cell_centered")] >= min_num_traj_per_bin
    ]

    return fpt_stats_df_no_nan


def build_first_passage_time_line_fit_results_dataframe(
    fpt_stats_df_no_nan: pd.DataFrame, metric_to_fit: Literal["mean", "median"] = "mean"
) -> pd.DataFrame:
    """
    Build a dataframe of line-fit results comparing grid-based and track-based first passage
    times using weighted orthogonal distance regression (ODR).
    The weights used are the inverse variances of the first passage times in each bin.

    Parameters
    ----------
    fpt_stats_df_no_nan
        Pre-filtered first passage time stats dataframe (no NaN values in the metric
        columns), as returned by :func:`filter_fpt_stats_df_by_min_num_trajectories`.
    metric_to_fit
        Which central-tendency metric to use as the value to regress: ``"mean"`` or
        ``"median"``.

    Returns
    -------
    :
        DataFrame with one row per (dataset, fixed-point, stability) group containing
        the OLS and ODR slope, intercept, and goodness-of-fit statistics.
    """
    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_to_fit == "median" else metric_to_fit
    template = ColumnTemplate.FIRST_PASSAGE_TIME_METRIC

    # perform a linear regression comparing the grid and tracked metrics for each fixed point
    line_fit_df = (
        fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.FIXED_POINT_STABILITY,
            ]
        )
        .apply(
            lambda df, metric=metric, template=template: pd.Series(
                index=[
                    Column.VectorField.LINEFIT_SLOPE,
                    Column.VectorField.LINEFIT_INTERCEPT_ODR,
                    Column.VectorField.LINEFIT_SLOPE_STDEV_ODR,
                    Column.VectorField.LINEFIT_INTERCEPT_STDEV_ODR,
                    Column.VectorField.LINEFIT_REDUCED_CHI_SQUARED_ODR,
                    Column.VectorField.ODR_RESULT,
                ],
                # use the inverse of the variance of the mean (sampling variance)
                # as the weights for the ODR fit, which is the square of the standard error
                data=fit_orthogonal_distance_regression(
                    x=df[template % (metric, "grid_based")],
                    y=df[template % (metric, "cell_centered")],
                    weight_x=df[template % ("sem", "grid_based")] ** -2,
                    weight_y=df[template % ("sem", "cell_centered")] ** -2,
                ),
            )
        )
        .reset_index()
    )

    # perform a Pearson correlation test comparing the grid and tracked metrics for each fixed point
    pearson_df = (
        fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.FIXED_POINT_STABILITY,
            ]
        ).apply(
            lambda df, metric=metric, template=template: pd.Series(
                index=[Column.VectorField.PEARSON_R, Column.VectorField.PEARSON_P],
                data=pearsonr(
                    x=df[template % (metric, "grid_based")],
                    y=df[template % (metric, "cell_centered")],
                ),
            )
        )
    ).reset_index()

    line_fit_df = line_fit_df.merge(
        pearson_df,
        on=[Column.DATASET, Column.VectorField.FIXED_POINT_INDEX, Column.FIXED_POINT_STABILITY],
        validate="one_to_one",
    )
    return line_fit_df


def load_filtered_first_passage_time_dataframe(
    first_passage_time_manifest: DataframeManifest,
    dataset_names: list[str] | None = None,
    min_num_traj_per_bin: int = FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
    metric_to_fit: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """Load and filter first passage time dataframes for given datasets."""

    # Load the first passage time statistics dataframe. If given, only load
    # the selected datasets. Otherwise, load all datasets.
    if dataset_names is None:
        dfs = [load_dataframe(loc) for loc in first_passage_time_manifest.locations.values()]
    else:
        dfs = [load_dataframe(first_passage_time_manifest.locations[d]) for d in dataset_names]

    fpt_stats_df = pd.concat(dfs)

    # filter out nans and bins with too few trajectories for a certain measure
    # (either mean or median) for the correlation and line fitting steps
    return filter_first_passage_time_by_min_num_trajectories(
        fpt_stats_df=fpt_stats_df,
        min_num_traj_per_bin=min_num_traj_per_bin,
        metric_for_filter=metric_to_fit,
    )
