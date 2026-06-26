"""Methods for calculating and analyzing first passage time"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_track_length,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_track_duration_to_dataframe,
)
from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.track_integration import (
    compute_first_passage_time_parameter_sweep_df,
    compute_first_passage_time_stats_for_bins,
    merge_grid_and_tracked_first_passage_time_parameter_sweep_dfs,
    merge_grid_and_tracked_first_passage_time_stats_dfs,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.literal_types import PatchTypeLiteral
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
)


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
    column_suffix: str = "",
    polar_angle_period: float | None = None,
    time_column: str = Column.TIMEPOINT,
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
    time_column
        Column name in trajectory_df corresponding to the time variable
        (e.g. `Column.TIMEPOINT` or `Column.SegData.TIME_HRS`).

    Returns
    -------
    :
        DataFrame containing the distances to the nearest fixed point for each trajectory point.
    """

    if fixed_point_columns is None:
        fixed_point_columns = trajectory_columns

    if column_suffix and not column_suffix.startswith("_"):
        column_suffix = f"_{column_suffix}"  # make sure the suffix starts with an underscore

    # determine distance from each fixed point over time and add to the dataframe, along
    # with the signed difference along each axis (e.g. theta, r, rho) from each fixed point
    dist_from_fp_col_prefix = Column.VectorField.DISTANCE_FROM_FP_PREFIX
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
            trajectory_df[
                f"{Column.VectorField.DISTANCE_FROM_FP_1D_SIGNED_PREFIX}{i}_{col}{column_suffix}"
            ] = diff_func(trajectory_df[trajectory_columns[j]])

        dynamics_diff_columns = [
            f"{Column.VectorField.DISTANCE_FROM_FP_1D_SIGNED_PREFIX}{i}_{col}{column_suffix}"
            for col in fixed_point_columns
        ]
        trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}"] = np.linalg.norm(
            trajectory_df[dynamics_diff_columns], axis=1
        )

        dd = (
            trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}"]
            .groupby(trajectory_df[Column.CROP_INDEX])
            .diff()
        )
        dt = trajectory_df[time_column].groupby(trajectory_df[Column.CROP_INDEX]).diff()
        trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}_veloc"] = dd / dt

    # determine which fixed point is closest at each timepoint for each track
    dist_from_fp_columns = [
        f"{dist_from_fp_col_prefix}{i}{column_suffix}" for i in fixed_point_df.index
    ]
    trajectory_df[f"closest_fp{column_suffix}"] = (
        trajectory_df[dist_from_fp_columns]
        .idxmin(axis=1, skipna=True)
        .transform(
            lambda s: (
                np.nan if pd.isna(s) else int(s.strip(dist_from_fp_col_prefix).strip(column_suffix))
            )
        )
    )

    # create a dictionary mapping a fixed point index to its stability
    fp_stability_map = dict(
        zip(
            fixed_point_df.index,
            fixed_point_df[Column.FIXED_POINT_STABILITY],
            strict=True,
        )
    )

    # add the stability as a column for the closest fixed point at each timepoint
    trajectory_df[f"closest_fp_stability{column_suffix}"] = trajectory_df[
        f"closest_fp{column_suffix}"
    ].map(fp_stability_map)

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
        time_column=Column.SegData.TIME_HRS,
    )

    # add the distances from the fixed points for the track-based trajectories
    traj_df_tracked = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_tracked,
        fixed_point_df=fixed_points_df,
        trajectory_columns=list(DYNAMICS_COLUMN_NAMES),
        fixed_point_columns=fp_cluster_mean_cols,
        time_column=Column.SegData.TIME_HRS,
    )

    return traj_df_grid, traj_df_tracked, fixed_points_df


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
    column = f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fixed_point_index}"
    new_column_name = f"{Column.VectorField.FIRST_PASSAGE_PREFIX}{column}"
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
            < row[f"{Column.VectorField.FIRST_PASSAGE_DIST_PREFIX}{fp_idx}"],
            axis=1,
        )
    ]

    # compute the time to the first passage time from each timepoint
    trajectory_df[f"{Column.VectorField.TIME_TO_FP_PREFIX}{fixed_point_index}"] = (
        trajectory_df[f"{Column.VectorField.FIRST_PASSAGE_DIST_PREFIX}{fixed_point_index}"]
        - trajectory_df[time_column]
    )

    return trajectory_df


def compute_first_passage_times_one_dataset(
    dataset_name: str,
    minimum_track_length: int,
    fixed_point_radius_threshold: float | None = None,
    bin_size_theta_deg: float | None = None,
    bin_size_radius: float | None = None,
    bin_size_rho: float | None = None,
    collapse_feature: Literal["theta", "radius", "rho"] | None = None,
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

    bin_widths = [bin_sizes[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_limits_list = [bin_limits[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_edges, bin_centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits_list)

    if collapse_feature is not None:
        feature_to_column_map = {
            "theta": Column.DiffAEData.POLAR_ANGLE,
            "radius": Column.DiffAEData.POLAR_RADIUS,
            "rho": Column.DiffAEData.PC3_FLIPPED,
        }
        feature_to_collapse = feature_to_column_map[collapse_feature]
        collapse_index = DYNAMICS_COLUMN_NAMES.index(feature_to_collapse)
        # convert the bin edges into a single bin with only 2 edges
        bin_edges[collapse_index] = np.array(
            [bin_edges[collapse_index].min(), bin_edges[collapse_index].max()]
        )
        # take the midpoint of the bin edges as the bin center for the collapsed feature
        bin_centers[collapse_index] = np.array(
            [(bin_edges[collapse_index][0] + bin_edges[collapse_index][1]) / 2]
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
        fpt_param_sweep_df_grid = compute_first_passage_time_parameter_sweep_df(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_grid,
            thresholds=thresholds,
        )
        fpt_param_sweep_df_tracked = compute_first_passage_time_parameter_sweep_df(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_tracked,
            thresholds=thresholds,
        )
        parameter_sweep_df = merge_grid_and_tracked_first_passage_time_parameter_sweep_dfs(
            fpt_param_sweep_df_grid=fpt_param_sweep_df_grid,
            fpt_param_sweep_df_tracked=fpt_param_sweep_df_tracked,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
        )
        parameter_sweep_df[Column.FIXED_POINT_STABILITY] = fp_stability

        traj_df_grid[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"] = (
            traj_df_grid[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}"]
            <= fixed_point_radius_threshold
        )
        traj_df_tracked[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"] = (
            traj_df_tracked[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}"]
            <= fixed_point_radius_threshold
        )

        traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"] = traj_df_grid.groupby(
            Column.CROP_INDEX
        )[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"].transform(any)
        traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"] = (
            traj_df_tracked.groupby(Column.CROP_INDEX)[
                f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"
            ].transform(any)
        )

        traj_df_grid_sub = traj_df_grid[
            traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"]
        ]
        traj_df_tracked_sub = traj_df_tracked[
            traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"]
        ]

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

        # 3. for each bin (across all steady-state timepoints), compute the mean,
        #    median, and standard deviation of first-passage times for the trajectories
        time_to_first_passage_col_name = f"{Column.VectorField.TIME_TO_FP_PREFIX}{fp_idx}"

        fpt_stats_df_grid = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_grid_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )
        fpt_stats_df_tracked = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_tracked_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )

        # merge the grid and tracked first passage time stats dataframes
        first_passage_time_stats_df = merge_grid_and_tracked_first_passage_time_stats_dfs(
            fpt_stats_df_grid=fpt_stats_df_grid,
            fpt_stats_df_tracked=fpt_stats_df_tracked,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
        )
        first_passage_time_stats_df[Column.FIXED_POINT_STABILITY] = fp_stability
        fp_dynamics_cols = [
            f"{Column.VectorField.FIXED_POINT_PREFIX}{col}" for col in DYNAMICS_COLUMN_NAMES
        ]
        first_passage_time_stats_df[fp_dynamics_cols] = fp_row[list(DYNAMICS_COLUMN_NAMES)]

        # add the bin sizes and bin limits to the dataframes for transparency
        for col in DYNAMICS_COLUMN_NAMES:
            first_passage_time_stats_df = first_passage_time_stats_df.assign(
                **{
                    f"{Column.VectorField.BIN_SIZE_PREFIX}{col}": bin_sizes[col],
                    f"{Column.VectorField.BIN_LIMITS_PREFIX}{col}": [bin_limits[col]]
                    * len(first_passage_time_stats_df),
                }
            )

        fpt_stats_df_list.append(first_passage_time_stats_df)
        param_sweep_df_list.append(parameter_sweep_df)

    return fpt_stats_df_list, param_sweep_df_list
