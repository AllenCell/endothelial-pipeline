import logging
import math
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import dask.array as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from skimage.measure import regionprops
from tqdm import tqdm

from endo_pipeline.configs import get_annotated_timepoints_for_position, load_dataset_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.lib_init_density_vs_flow import vector_mean_angle_and_mag
from endo_pipeline.library.model.eval_model import add_diffae_model_eval_crop_columns
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.image_data import (
    DIMENSION_ORDER,
    IMG_SHAPE_RESOLUTION_0_3i_X,
    IMG_SHAPE_RESOLUTION_0_3i_Y,
)

logger = logging.getLogger(__name__)


def merge_measured_segmentation_features_tables(
    cellprops_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    nucprops_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge dataframes output by the tracking workflow (run_cdh5_tracking.py), the
    segmentation measurement workflow (get_cdh5_measured_features.py), and the
    labelfree nuclei measurement workflow (get_nuclei_measured_features.py).
    """

    # Drop duplicate columns in tracking dataframe before merging with
    # segmentation features dataframe
    tracking_df = tracking_df.drop(
        columns=[
            Column.SegData.CENTROID,
            Column.SegData.AREA_PX_SQ,
            Column.SegData.PERIMETER_PX,
            Column.SegData.ECCENTRICITY,
            Column.SegData.ORIENTATION,
            Column.SegData.CENTROID_X,
            Column.SegData.CENTROID_Y,
            Column.SegDataFilters.IS_EDGE_SEGMENTATION,
        ]
    )

    merge_columns = [Column.DATASET, Column.POSITION, Column.TIMEPOINT]

    big_table = pd.merge(
        left=tracking_df, right=cellprops_df, on=[*merge_columns, Column.SegData.LABEL]
    )
    big_table = pd.merge(
        left=big_table,
        right=nucprops_df,
        left_on=[*merge_columns, Column.SegData.LABEL],
        right_on=[*merge_columns, Column.SegDataWorkflowVerification.CDH5_SEGMENTATION_LABEL],
    )

    # Drop the now redundant CDH5 segmentation label column used for merging
    big_table = big_table.drop(columns=[Column.SegDataWorkflowVerification.CDH5_SEGMENTATION_LABEL])

    return big_table


def write_filter_log_file(
    out_dir: Path,
    datasets_analyzed: Sequence[str],
    num_rows_before_filtering: int,
    num_rows_after_filtering: int,
    num_unique_tracks_before_filtering: int,
    num_unique_tracks_after_filtering: int,
) -> None:
    timestamp = pd.Timestamp.now()
    out_dir_logs = out_dir / f'filter_run_logs/{timestamp.strftime("%Y%m%d_%H%M")}/'
    out_dir_logs.mkdir(parents=True, exist_ok=True)
    with open(
        out_dir_logs / f'{timestamp.strftime("%Y%m%d_%H%M")}_filtered_tracking_results_run_log.txt',
        "w",
    ) as f:
        f.write(
            f"""
                Date run: {timestamp!s}\n
                Datasets analyzed: {datasets_analyzed}\n
                Number of rows before filtering: {num_rows_before_filtering}\n
                Number of rows after filtering: {num_rows_after_filtering}\n
                Number of unique tracks before filtering: {num_unique_tracks_before_filtering}\n
                Number of unique tracks after filtering: {num_unique_tracks_after_filtering}\n"""
        )
    return


def save_filter_validation_plots(
    out_dir: Path,
    big_table_filtered: pd.DataFrame,
    min_track_duration: int,
) -> None:
    for (dataset_nm, position), df in big_table_filtered.groupby([Column.DATASET, Column.POSITION]):
        summary = df.groupby(Column.TIMEPOINT)[
            [
                Column.TIMEPOINT,
                Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
                Column.SegData.NUM_TRACKS_AFTER_FILTERING,
            ]
        ].agg("median")
        timelapse_duration = load_dataset_config(dataset_nm).duration
        # QUESTION: are the number of cell labels after filtering roughly equally distributed over time?
        out_dir_plots = out_dir / "num_tracks_plots" / dataset_nm
        out_dir_plots.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.set_title(f"Dataset {dataset_nm} P{position}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Number of unique tracks")
        sns.lineplot(
            x=Column.TIMEPOINT,
            y=Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
            data=summary,
            ax=ax,
            label="Before filtering",
        )
        sns.lineplot(
            x=Column.TIMEPOINT,
            y=Column.SegData.NUM_TRACKS_AFTER_FILTERING,
            data=summary,
            ax=ax,
            label="After filtering",
        )
        ax.set_ylim(0)
        left_of_boxes = [
            (0, 0),
            (
                timelapse_duration - min_track_duration,
                timelapse_duration - min_track_duration,
            ),
        ]
        right_of_boxes = [
            (min_track_duration, min_track_duration),
            (timelapse_duration, timelapse_duration),
        ]
        top_of_boxes = [ax.get_ylim()] * len(left_of_boxes)
        boxes = zip(top_of_boxes, left_of_boxes, right_of_boxes, strict=True)
        ax.set_xlim(0, timelapse_duration)
        [ax.fill_betweenx(y=y, x1=x1, x2=x2, color="lightgrey") for y, x1, x2 in boxes]
        fig.savefig(
            out_dir_plots / f"{dataset_nm}_P{position}_num_tracks_over_time.png",
            dpi=80,
        )
        plt.close(fig)
    return


def add_filter_columns(
    big_table: pd.DataFrame,
    out_dir: Path | None,
    min_track_duration: int = 24,
    max_area_change: float = 0.1,
    min_num_valid_points_per_track: int = 20,
) -> pd.DataFrame:
    """
    These filter columns are `True` if an entry should be dropped, therefore
    keeping anything where a filter is `False`.
    E.g. to remove entries where cells touch the image border you can take
    `big_table[big_table["is_edge_segmentation"] == False]`
    (or equivalently: `big_table[~big_table["is_edge_segmentation"]]`)
    """

    # get the number of segmentations in total and per timepoint
    num_rows_before_filtering = len(big_table)
    num_unique_tracks_before_filtering = (
        big_table.groupby([Column.DATASET, Column.POSITION])[Column.TRACK_ID].nunique().sum()
    )

    # keep only tracks with duration longer than min_track_duration
    big_table[Column.SegDataFilters.MIN_TRACK_DURATION] = min_track_duration
    big_table[Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION] = (
        big_table[Column.TRACK_LENGTH] > min_track_duration
    )

    # keep only tracks where area_change is not too large
    big_table[Column.SegDataFilters.MAX_SMOOTHED_AREA_NORMALIZED_CHANGE] = max_area_change
    big_table[Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE] = (
        big_table[Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF].abs() < max_area_change
    )

    # is_included is just all the previous filters combined with the
    # filter to exclude segmentations that touch the edges of the image
    big_table[Column.SegDataFilters.IS_INCLUDED] = (
        big_table[Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION]
        & big_table[Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE]
        & ~big_table[Column.SegDataFilters.IS_EDGE_SEGMENTATION]
    )

    # drop because there are insufficient valid timepoints
    big_table[Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK] = big_table.groupby(
        [Column.DATASET, Column.POSITION, Column.TRACK_ID]
    )[Column.SegDataFilters.IS_INCLUDED].transform(sum)
    big_table[Column.SegDataFilters.MIN_NUM_VALID_TIMEPOINTS_PER_TRACK] = (
        min_num_valid_points_per_track
    )
    big_table[Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK] = (
        big_table[Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK]
        > min_num_valid_points_per_track
    )

    # update is_included column with valid_tp_per_track
    big_table[Column.SegDataFilters.IS_INCLUDED] = (
        big_table[Column.SegDataFilters.IS_INCLUDED]
        & big_table[Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK]
    )

    # get the number of unique tracks after filtering in total and per timepoint
    num_rows_after_filtering = np.count_nonzero(big_table[Column.SegDataFilters.IS_INCLUDED])
    num_unique_tracks_after_filtering = (
        big_table[big_table[Column.SegDataFilters.IS_INCLUDED]]
        .groupby([Column.DATASET, Column.POSITION])[Column.TRACK_ID]
        .nunique()
        .sum()
    )
    big_table[Column.SegData.NUM_TRACKS_AFTER_FILTERING] = (
        big_table[big_table[Column.SegDataFilters.IS_INCLUDED]]
        .groupby([Column.DATASET, Column.POSITION, Column.TIMEPOINT])[Column.TRACK_ID]
        .transform(lambda x: x.nunique())
    )

    # save a log file of the filtering that was done if saving the results
    if out_dir:
        # save a log file and create some plots showing number of
        # tracks before and after filtering
        datasets_analyzed = big_table[Column.DATASET].unique().tolist()
        write_filter_log_file(
            out_dir,
            datasets_analyzed,
            num_rows_before_filtering,
            num_rows_after_filtering,
            num_unique_tracks_before_filtering,
            num_unique_tracks_after_filtering,
        )
        # create some validation plots
        save_filter_validation_plots(
            out_dir,
            big_table[big_table[Column.SegDataFilters.IS_INCLUDED]],
            min_track_duration,
        )
    return big_table


def add_cell_piling_and_steady_state_annotation_columns(big_table: pd.DataFrame) -> pd.DataFrame:
    """Adds the annotations about cell piling and steady state that were done by
    hand as columns to the data table.
    """
    # load dataset config and timepoint annotations
    dataset = sequence_to_scalar(big_table[Column.DATASET])
    dataset_config = load_dataset_config(dataset)
    if dataset_config.timepoint_annotations is not None:
        filters_for_dataset = list(dataset_config.timepoint_annotations.keys())
        for filt in filters_for_dataset:
            # add the timepoint annotations as filter columns
            big_table[filt] = (
                big_table.groupby(Column.POSITION, as_index=True)
                .apply(
                    lambda df, filt=filt: (
                        pd.DataFrame(
                            (
                                df[Column.TIMEPOINT].isin(
                                    get_annotated_timepoints_for_position(
                                        dataset_config,
                                        sequence_to_scalar(df[Column.POSITION]),
                                        [filt],
                                    )
                                )
                            ),
                            index=df.index,
                        )
                    )
                )
                .droplevel(0)
            )
    return big_table


def add_track_duration_to_dataframe(
    dataframe: pd.DataFrame, grouping_columns: list[str | Column], time_column: str | Column
) -> pd.DataFrame:
    """Adds a column for the track duration to the dataframe.
    Track duration is calculated as the difference between the maximum and minimum
    timepoints for each track defined by the grouping columns.

    Parameters
    ----------
    dataframe
        The input dataframe containing the tracking data, which must include the columns specified in `grouping_columns` and `time_column`.
    grouping_columns
        The columns to group by when calculating the track duration.
        This is expected to be either
        [Column.DATASET, Column.POSITION, Column.TRACK_ID]
        or
        [Column.CROP_INDEX]
        depending on the purpose of the computed track duration and which dataframe is used.
    time_column
        The column representing the timepoints for each track.
        This is expected to be Column.TIMEPOINT, but is left as a parameter for
        flexibility in case dimensional time is used (e.g. Column.SegData.TIME_HRS).

    Returns
    -------
    pd.DataFrame
         The input dataframe with an additional column for the track duration.
    """
    dataframe[Column.TRACK_LENGTH] = dataframe.groupby(grouping_columns)[time_column].transform(
        lambda t: t.max() - t.min()
    )
    return dataframe


def calculate_derived_data_dynamics_independent(
    big_table: pd.DataFrame, num_processes: int | None = None
) -> pd.DataFrame:
    """
    This function uses the existing columns in the data table to calculate
    other features about the data such as dimensionalizing data and
    converting measurements based on one thing (e.g. alignment) to
    another feature (e.g. nematic order) that is used in other analyses
    to help with interpretability of the data.

    The following things are calculated here:
    - the time in minutes and hours
    - the number of tracks at a given timepoint
    - the orientation of the fitted ellipse in degrees (instead of radians)
    - the nematic order
    - the aspect ratio
    - the velocities of the regions based on centroid displacement
    - the centroid velocity magnitude and angle
    - the number of neighbors touching each region
    """
    dataset_name = sequence_to_scalar(big_table[Column.DATASET])
    data_config = load_dataset_config(dataset_name)

    # add the shear stress regime to the data table
    logger.info("Adding shear stress regime...")
    shear_stress_regime = "_to_".join([shear.value for shear in data_config.shear_stress_regime])
    big_table[Column.SHEAR_STRESS_REGIME] = shear_stress_regime

    shear_stresses = [condition.shear_stress for condition in data_config.flow_conditions]
    big_table[Column.SHEAR_STRESS] = [shear_stresses] * len(big_table)

    # dimensionalize the time column
    logger.info("Adding time intervals per timepoint...")
    if data_config.time_interval_in_minutes is not None:
        dt_in_mins = data_config.time_interval_in_minutes
    else:
        dt_in_mins = np.nan
    big_table[Column.TIME_RESOLUTION_MINUTES] = dt_in_mins

    logger.info("Calculating time in minutes and hours...")
    big_table[Column.SegData.TIME_MINS] = big_table[Column.TIMEPOINT] * dt_in_mins
    big_table[Column.SegData.TIME_HRS] = big_table[Column.SegData.TIME_MINS] / 60

    # add time elapsed since flow onset (in hours)
    flow_start_time_hrs = data_config.flow_conditions[0].start * dt_in_mins / 60.0
    big_table[Column.SegData.TIME_HRS_SINCE_FLOW] = (
        big_table[Column.SegData.TIME_HRS] - flow_start_time_hrs
    )

    # add a column for the number of unique tracks
    # per dataset per position per timepoint
    # (this should be 1 everywhere)
    big_table[Column.SegDataWorkflowVerification.NUM_UNIQUE_TRACKS_PER_TIMEPOINT] = (
        big_table.groupby(
            [Column.DATASET, Column.POSITION, Column.TIMEPOINT, Column.TRACK_ID]
        ).transform("size")
    )

    # add the columns for the fold change in area
    logger.info("Calculating locally-normalized area...")
    sigma = 2.0
    big_table[Column.SegDataWorkflowVerification.SIGMA_FOR_AREA_SMOOTHING] = sigma
    big_table[Column.SegDataWorkflowVerification.SMOOTHED_AREA_NORMALIZED] = big_table.groupby(
        [Column.DATASET, Column.POSITION, Column.TRACK_ID]
    )[Column.SegData.AREA_PX_SQ].transform(
        lambda x: calculate_smoothed_normd_area(x, smoothing_sigma=sigma)
    )
    big_table[Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF] = big_table.groupby(
        [Column.DATASET, Column.POSITION, Column.TRACK_ID]
    )[Column.SegDataWorkflowVerification.SMOOTHED_AREA_NORMALIZED].transform(lambda x: x.diff())

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    logger.info("Adding number of tracks for each timepoint...")
    big_table[Column.SegData.NUM_TRACKS_BEFORE_FILTERING] = big_table.groupby(
        [Column.DATASET, Column.POSITION, Column.TIMEPOINT]
    )[Column.TRACK_ID].transform(lambda x: x.nunique())

    # add the duration of each track
    logger.info("Calculating track durations...")
    big_table = add_track_duration_to_dataframe(
        dataframe=big_table,
        grouping_columns=[Column.DATASET, Column.POSITION, Column.TRACK_ID],
        time_column=Column.TIMEPOINT,
    )

    # add column for orientation in degrees of the
    # ellipse fitted to each segmentation in degrees
    logger.info("Converting orientation to degrees...")
    big_table[Column.SegData.ALIGNMENT] = big_table[Column.SegData.ORIENTATION].transform(
        lambda x: make_orientation_relative_to_flow(x)
    )
    big_table[Column.SegData.ALIGNMENT_DEG] = np.rad2deg(big_table[Column.SegData.ALIGNMENT])

    # add column for the orientation in degrees
    big_table[Column.SegData.ORIENTATION_DEG] = np.rad2deg(big_table[Column.SegData.ORIENTATION])

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    logger.info("Calculating nematic order and aspect ratio...")
    big_table[Column.SegData.NEMATIC_ORDER] = big_table[Column.SegData.ORIENTATION].transform(
        get_nematic_order
    )
    big_table[Column.SegData.ASPECT_RATIO] = big_table[Column.SegData.ECCENTRICITY].transform(
        get_aspect_ratio
    )

    # add pixel sizes
    big_table[Column.PIXEL_SIZE_XY_IN_UM] = data_config.pixel_size_xy_in_um
    big_table[Column.SegData.AREA_UM_SQ] = (
        big_table[Column.SegData.AREA_PX_SQ] * big_table[Column.PIXEL_SIZE_XY_IN_UM] ** 2
    )
    big_table[Column.SegData.PERIMETER_UM] = (
        big_table[Column.SegData.PERIMETER_PX] * big_table[Column.PIXEL_SIZE_XY_IN_UM]
    )

    # compute intensity means and standard deviations for edge and node pixels
    # separately and together
    big_table[Column.SegData.EDGE_FLUOR_MEAN] = big_table[Column.SegData.EDGE_FLUOR].transform(
        lambda x: x.mean()
    )
    big_table[Column.SegData.EDGE_FLUOR_STD] = big_table[Column.SegData.EDGE_FLUOR].transform(
        lambda x: x.std()
    )
    big_table[Column.SegData.NODE_FLUOR_MEAN] = big_table[Column.SegData.NODE_FLUOR].transform(
        lambda x: x.mean()
    )
    big_table[Column.SegData.NODE_FLUOR_STD] = big_table[Column.SegData.NODE_FLUOR].transform(
        lambda x: x.std()
    )
    big_table[Column.SegData.EDGE_AND_NODE_FLUOR_MEAN] = big_table.apply(
        lambda row: np.mean(
            row[Column.SegData.EDGE_FLUOR].tolist() + row[Column.SegData.NODE_FLUOR].tolist()
        ),
        axis=1,
    )
    big_table[Column.SegData.EDGE_AND_NODE_FLUOR_STD] = big_table.apply(
        lambda row: np.std(
            row[Column.SegData.EDGE_FLUOR].tolist() + row[Column.SegData.NODE_FLUOR].tolist()
        ),
        axis=1,
    )

    # add a column for the number of neighbors
    # touching each region that is being tracked
    logger.info("Calculating number of neighbors...")
    big_table[Column.SegData.NUM_NEIGHBORS] = big_table[Column.SegData.NEIGHBOR_LABELS].transform(
        lambda x: len(x)
    )

    # add the image size and channel indices to the data table
    big_table[Column.IMAGE_SIZE_X] = IMG_SHAPE_RESOLUTION_0_3i_X
    big_table[Column.IMAGE_SIZE_Y] = IMG_SHAPE_RESOLUTION_0_3i_Y
    big_table[Column.CDH5_CHANNEL_INDEX_ZARR] = data_config.zarr_channel_indices.channel_488
    big_table[Column.BF_CHANNEL_INDEX_ZARR] = data_config.zarr_channel_indices.brightfield

    # add the number of nuclei that overlap the most with each cell
    # (this can be used as a filter later so we only measure cells
    # with a single clearly distinguishable nuclei)
    big_table[Column.SegDataWorkflowVerification.NUM_NUC_WITH_MOST_OVERLAP] = big_table[
        Column.SegDataWorkflowVerification.NUCLEI_LABELS_IN_CDH5_SEGMENTATION
    ].transform(len)

    # split the centroid column into separate x and y columns
    big_table[[Column.SegData.CENTROID_Y, Column.SegData.CENTROID_X]] = pd.DataFrame(
        big_table[Column.SegData.CENTROID].tolist(), index=big_table.index
    )

    # add the nuclei centroids relative to the cell centroids
    big_table[Column.SegData.NUCLEI_POSITION_X], big_table[Column.SegData.NUCLEI_POSITION_Y] = (
        get_nuclei_rel_to_cell_position(
            big_table[Column.SegData.CENTROID_X],
            big_table[Column.SegData.CENTROID_Y],
            big_table[Column.SegData.NUCLEI_CENTROID_X],
            big_table[Column.SegData.NUCLEI_CENTROID_Y],
        )
    )

    # get the angles and distances of the nuclei relative positions
    big_table[Column.SegData.NUCLEI_POSITION_DISTANCE] = np.linalg.norm(
        [big_table[Column.SegData.NUCLEI_POSITION_X], big_table[Column.SegData.NUCLEI_POSITION_Y]],
        axis=0,
    )
    big_table[Column.SegData.NUCLEI_POSITION_ANGLE] = np.arctan2(
        big_table[Column.SegData.NUCLEI_POSITION_Y], big_table[Column.SegData.NUCLEI_POSITION_X]
    )
    big_table[Column.SegData.NUCLEI_POSITION_ANGLE_DEG] = np.rad2deg(
        big_table[Column.SegData.NUCLEI_POSITION_ANGLE]
    )

    # add the DiffAE crop locations and binning level; these can be used to load
    # a crop from the zarr files and compute the number of nuclei in that crop
    big_table = add_diffae_model_eval_crop_columns(big_table)

    # compute the number of nuclei found in a defined crop size
    # (first take a subset using only the required columns to reduce memory usage)
    required_columns = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.SegData.LABEL,
        Column.SegData.CENTROID_Y,
        Column.SegData.CENTROID_X,
        Column.IMAGE_SIZE_Y,
        Column.IMAGE_SIZE_X,
        Column.SegData.CROP_SIZE,
        Column.SegData.START_Y_RES_0,
        Column.SegData.END_Y_RES_0,
        Column.SegData.START_X_RES_0,
        Column.SegData.END_X_RES_0,
        Column.SegDataFilters.IS_VALID_BBOX,
    ]
    num_nuclei_in_crop_df = add_num_nuclei_in_crop_column(
        big_table[required_columns], use_precomputed=False, max_cores=num_processes
    )
    crops = [Column.DATASET, Column.POSITION, Column.TIMEPOINT, Column.TRACK_ID]

    added_cols = list(set(num_nuclei_in_crop_df.columns) - set(big_table.columns))
    big_table = pd.merge(
        left=big_table,
        right=num_nuclei_in_crop_df[crops + added_cols],
        on=crops,
        how="left",
        validate="one_to_one",
    )
    del num_nuclei_in_crop_df

    # add column for the labels of other cells that are in the crop of each cell
    # so we can calculate mean migration vector of those other cells in the crop
    add_all_labels_in_crop_df = add_all_labels_in_crop_column(
        big_table[required_columns], use_precomputed=False, max_cores=num_processes
    )
    added_cols = list(set(add_all_labels_in_crop_df.columns) - set(big_table.columns))
    big_table = pd.merge(
        left=big_table,
        right=add_all_labels_in_crop_df[crops + added_cols],
        on=crops,
        how="left",
        validate="one_to_one",
    )

    return big_table


def calculate_derived_data_dynamics_dependent(
    big_table: pd.DataFrame,
    compute_per_crop_metrics: bool = False,
) -> pd.DataFrame:
    """
    Calculates dynamics-dependent features and add them to given dataframe.

    Added columns include:

    - centroid_dx_dt: cdh5-based cell segmentation centroid velocity in x (units
      in um/min)
    - centroid_dy_dt: cdh5_based cell segmentation centroid velocity in y (units
      in um/min)
    - centroid_velocity_magnitude: magnitude of the cdh5-based cell segmentation
      centroid velocity
    - centroid_velocity_angle: the angle of the cdh5-based cell segmentation
      centroid velocity (from -pi to pi with 0 being to the right)
    - dalignment_dt_deg_rel_to_flow: the change in alignment angle (in
      degrees/min)
    - num_tracks_at_T: the number of tracks at a given timepoint per dataset per
      position (this number is affected by any filtering that was done to the
      passed in table)

    NOTE: The accuracy of these metrics are affected by how clean the data in
    the table is, therefore it should only be used after filtering out incorrect
    segmentations from the data table.
    """

    # recalculate the centroid speeds of each track after filtering
    logger.info("Calculating centroid positions in microns...")
    big_table[Column.SegData.CENTROID_X_UM] = (
        big_table[Column.SegData.CENTROID_X] * big_table[Column.PIXEL_SIZE_XY_IN_UM]
    )
    big_table[Column.SegData.CENTROID_Y_UM] = (
        big_table[Column.SegData.CENTROID_Y] * big_table[Column.PIXEL_SIZE_XY_IN_UM]
    )

    logger.info("Calculating centroid velocities...")
    big_table[
        [
            Column.SegData.CENTROID_VELOCITY_X_UM_PER_MIN,
            Column.SegData.CENTROID_VELOCITY_Y_UM_PER_MIN,
        ]
    ] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)[
            [
                Column.SegData.CENTROID_X_UM,
                Column.SegData.CENTROID_Y_UM,
                Column.SegData.TIME_MINS,
            ]
        ]
        .apply(
            lambda df: pd.DataFrame(  # type: ignore[arg-type, return-value]
                columns=[
                    Column.SegData.CENTROID_VELOCITY_X_UM_PER_MIN,
                    Column.SegData.CENTROID_VELOCITY_Y_UM_PER_MIN,
                ],
                data=zip(
                    *get_centroid_velocity(
                        df[Column.SegData.CENTROID_X_UM].values,  # type: ignore[arg-type, call-overload, return-value]
                        df[Column.SegData.CENTROID_Y_UM].values,  # type: ignore[arg-type, call-overload, return-value]
                        df[Column.SegData.TIME_MINS].values,  # type: ignore[arg-type, call-overload, return-value]
                    ),
                    strict=True,
                ),  # type: ignore[return-value]
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    logger.info("Calculating centroid velocity magnitude and angle...")
    big_table[Column.SegData.CENTROID_VELOCITY_UM_PER_MIN] = np.linalg.norm(
        [
            big_table[Column.SegData.CENTROID_VELOCITY_X_UM_PER_MIN],
            big_table[Column.SegData.CENTROID_VELOCITY_Y_UM_PER_MIN],
        ],
        axis=0,
    )
    big_table[Column.SegData.CENTROID_VELOCITY_ANGLE] = np.arctan2(
        big_table[Column.SegData.CENTROID_VELOCITY_Y_UM_PER_MIN],
        big_table[Column.SegData.CENTROID_VELOCITY_X_UM_PER_MIN],
    )
    big_table[Column.SegData.CENTROID_VELOCITY_ANGLE_DEG] = np.rad2deg(
        big_table[Column.SegData.CENTROID_VELOCITY_ANGLE]
    )
    big_table[Column.SegData.ALIGNMENT_VELOCITY_DEG] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df[Column.SegData.ALIGNMENT_DEG].diff() / df[Column.SegData.TIME_MINS].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    big_table[Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG] = (
        get_smallest_angle_difference(
            big_table[Column.SegData.NUCLEI_POSITION_ANGLE_DEG],
            big_table[Column.SegData.CENTROID_VELOCITY_ANGLE_DEG],
        )
    )

    big_table[Column.SegData.NUCLEI_POSITION_X_UM] = (
        big_table[Column.SegData.NUCLEI_POSITION_X] * big_table[Column.PIXEL_SIZE_XY_IN_UM]
    )
    big_table[Column.SegData.NUCLEI_POSITION_Y_UM] = (
        big_table[Column.SegData.NUCLEI_POSITION_Y] * big_table[Column.PIXEL_SIZE_XY_IN_UM]
    )
    big_table[Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD] = np.einsum(
        "ij,ij->i",
        big_table[
            [
                Column.SegData.CENTROID_VELOCITY_X_UM_PER_MIN,
                Column.SegData.CENTROID_VELOCITY_Y_UM_PER_MIN,
            ]
        ],
        big_table[[Column.SegData.NUCLEI_POSITION_X_UM, Column.SegData.NUCLEI_POSITION_Y_UM]],
    )

    # add fluorescence intensity dynamics column
    logger.info("Calculating fluorescence intensity dynamics...")
    big_table[Column.SegData.CHANGE_IN_FLUOR_PER_MIN_CELL] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df[Column.SegData.CELL_FLUOR_MEAN].diff() / df[Column.SegData.TIME_MINS].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )
    big_table[Column.SegData.CHANGE_IN_FLUOR_PER_MIN_EDGE] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df[Column.SegData.EDGE_FLUOR_MEAN].diff() / df[Column.SegData.TIME_MINS].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )
    big_table[Column.SegData.CHANGE_IN_FLUOR_PER_MIN_NODE] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df[Column.SegData.NODE_FLUOR_MEAN].diff() / df[Column.SegData.TIME_MINS].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    # add a normalized version of the "time_hours" column
    big_table = add_normalized_time(big_table)

    # add approximate cell density dynamics column
    logger.info("Calculating approximate cell density dynamics...")
    big_table[Column.SegData.CHANGE_IN_NUM_NUCLEI_IN_CROP_PER_MIN] = (
        big_table.groupby([Column.DATASET, Column.POSITION, Column.TRACK_ID], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df[Column.SegData.NUM_NUCLEI_IN_CROP].diff() / df[Column.SegData.TIME_MINS].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    if compute_per_crop_metrics:
        # find the migration vectors for all cells in a crop
        logger.info("Calculating vector mean of migration per crop...")
        big_table = add_vector_mean_of_migration_in_crop_column(
            big_table, velocity_column_name=Column.SegData.CENTROID_VELOCITY_UM_PER_MIN
        )

    return big_table


def get_nematic_order(theta: float) -> float:
    nematic_order_S = np.cos(2 * theta)
    return nematic_order_S


def get_aspect_ratio(eccentricity: float) -> float:
    # The following is a derivation of the aspect ratio
    # from the eccentricity:
    # eccentricity = focal_distance / major_axis
    # focal distance = sqrt(major_axis**2 - minor_axis**2)
    # eccentricity**2 = (major_axis**2 - minor_axis**2) / major_axis**2
    # eccentricity**2 = 1 - (minor_axis / major_axis)**2
    # aspect_ratio = major_axis / minor_axis
    # eccentricity**2 = 1 - (1 / aspect_ratio)**2
    # 1**2 / aspect_ratio**2 = 1 - eccentricity**2
    # 1 / (1 - eccentricity**2) = aspect_ratio**2
    # aspect_ratio = sqrt(1 / (1 - eccentricity**2))
    aspect_ratio = np.sqrt(1 / (1 - eccentricity**2))
    return aspect_ratio


def get_centroid_velocity(
    centroid_xs: float, centroid_ys: float, timepoints: float
) -> tuple[float, float]:
    dx, dy, dt = np.diff([centroid_xs, centroid_ys, timepoints], prepend=np.nan, axis=1)
    dx_dt, dy_dt = dx / dt, dy / dt
    return dx_dt, dy_dt


def calculate_smoothed_normd_area(
    area: pd.Series | np.ndarray,
    smoothing_sigma: float = 2.0,
) -> pd.DataFrame:
    smoothed_area = gaussian_filter1d(area, sigma=smoothing_sigma)
    area_normd = area / smoothed_area
    return area_normd


def make_orientation_relative_to_flow(orientation: float) -> float:
    """
    Restricts the orientation to be between 0 and pi/2 to
    be between 0 and pi/4. This way the orientations can
    be interpreted as being either aligned (i.e. parallel)
    to the flow (0 degrees) or perpendicular to flow (90 degrees).
    Orientation must be in radians.
    """
    # restrict the x and y components of the orientation to be strictly
    # positive so that the angle is between 0 and pi/2
    x_component_rel_flow = np.abs(np.cos(orientation))
    y_component_rel_flow = np.abs(np.sin(orientation))
    # take the arctan of these components to get the angle relative to flow
    return np.arctan2(y_component_rel_flow, x_component_rel_flow)


def get_nuclei_rel_to_cell_position(
    nuclei_centroid_x: float | np.ndarray | pd.Series,
    nuclei_centroid_y: float | np.ndarray | pd.Series,
    cell_centroid_x: float | np.ndarray | pd.Series,
    cell_centroid_y: float | np.ndarray | pd.Series,
) -> tuple[float | np.ndarray | pd.Series, float | np.ndarray | pd.Series]:

    dx = cell_centroid_x - nuclei_centroid_x
    dy = cell_centroid_y - nuclei_centroid_y

    return dx, dy


def get_smallest_angle_difference(
    angles: np.ndarray | pd.Series,
    reference_angles: np.ndarray | pd.Series,
    units: Literal["deg", "rad"] = "deg",
) -> np.ndarray:
    """
    Returns the smallest difference between angles and reference_angles. The
    result is signed, so if the returned angle is positive then the angle is
    counter-clockwise from the reference angle, and if the returned angle is
    negative then the angle is clockwise from the reference angle.

    Parameters
    ----------
    angles
        The angles to compare.
    reference_angles
        The reference angles to compare against.
    units
        The units of the angles. Either "deg" for degrees or "rad" for radians.

    Returns
    -------
    :
        The smallest difference between the angles and the reference angles.
    """

    if units == "deg":
        full_circle = 360.0
    elif units == "rad":
        full_circle = 2 * math.pi
    else:
        raise ValueError("units must be either 'deg' or 'rad'")

    half_circle = full_circle / 2

    def smallest_angle_difference_helper():
        for angle, ref_angle in zip(angles, reference_angles, strict=True):
            diff = (angle - ref_angle) % full_circle  # diff is in [0, full_circle)
            yield diff if diff < half_circle else diff - full_circle

    return np.array(list(smallest_angle_difference_helper()))


def get_nuclei_coords(
    props: regionprops,  # type:ignore
    props_dim_order: str,
    kind: Literal["centroid", "all"] = "centroid",
) -> dict[str, np.ndarray]:
    """
    Get the coordinates of the nuclei in the image.

    Parameters
    ----------
    props : regionprops
        The properties of the labeled nuclei in the image.
    props_dim_order : str
        The dimension order of the properties, e.g. "YX" or "ZYX".
        NOTE: this has only been tested with dim_order = "YX".
    kind : Literal["centroid", "all"]
        The kind of coordinates to return.
        "centroid" will return only the centroids of the labeled nuclei,
        while "all" will return all the coordinates of the nuclei.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with the coordinates of the nuclei in the image.
        The keys are "coords_Y", "coords_X", etc. depending on props_dim_order.
    """

    if kind == "all":
        # find the largest nuclei in the image because we
        # will need it for padding the coordinates later
        biggest_nuc_mask = max([p.num_pixels for p in props])  # type:ignore

    nuclei_coords: dict = {f"coords_{d}": [] for d in props_dim_order}
    for p in props:  # type:ignore
        match kind:
            case "centroid":
                # get only nuclei centroids
                # the ndmin=2 is so that the p.centroid shape is the same as p.coords
                # and will work in the function `get_num_nuclei_in_crops` correctly
                coords = np.array(p.centroid, ndmin=2).astype(float)
            case "all":
                # get all the nuclei coordinates
                coords = p.coords.astype(float)
                # define how much padding you need to add to these nuclei coordinates
                pad_width = ((0, biggest_nuc_mask - p.coords.shape[0]), (0, 0))
                # do the padding
                coords = np.pad(coords, pad_width, mode="constant", constant_values=np.nan)
        for dim in props_dim_order:
            nuclei_coords[f"coords_{dim}"].append(coords[..., props_dim_order.index(dim)])
    if any(nuclei_coords.values()):
        nuclei_coords_arrs = {
            dim: np.array(np.stack(coords).squeeze(), ndmin=1)
            for dim, coords in nuclei_coords.items()
        }
    else:
        nuclei_coords_arrs = {
            dim: np.array(coords, ndmin=1) for dim, coords in nuclei_coords.items()
        }

    return nuclei_coords_arrs


def get_num_unique_values_in_bounds_from_df(
    nuclei_coords_Y: pd.Series,
    nuclei_coords_X: pd.Series,
    crop_bounds_Y: tuple[pd.Series, pd.Series],
    crop_bounds_X: tuple[pd.Series, pd.Series],
) -> np.ndarray:
    """
    Returns the number of uniquely labeled coordinates within some crop bounds.
    This is used to count the number of nuclei in each crop.

    Parameters
    ----------
    nuclei_coords_Y : pd.Series
        The Y coordinates of the nuclei centroids.
    nuclei_coords_X : pd.Series
        The X coordinates of the nuclei centroids.
    crop_bounds_Y : tuple[pd.Series, pd.Series]
        The start and end Y coordinates of the crop bounds.
    crop_bounds_X : tuple[pd.Series, pd.Series]
        The start and end X coordinates of the crop bounds.

    Returns
    -------
    np.ndarray
        An array with the number of unique nuclei in each crop.

    Notes
    -----
    nuclei_coords_Y and nuclei_coords_X have the shape:
    (n_crops x n_unique_labels)
    crop_bounds_Y has the shape (n_crops, 2)
    """
    start_y, end_y = crop_bounds_Y
    start_x, end_x = crop_bounds_X

    coord_in_Y_bounds = np.logical_and(
        nuclei_coords_Y >= np.reshape(start_y, (len(start_y), 1)),  # type:ignore[arg-type]
        nuclei_coords_Y <= np.reshape(end_y, (len(end_y), 1)),  # type:ignore[arg-type]
    )
    coord_in_X_bounds = np.logical_and(
        nuclei_coords_X >= np.reshape(start_x, (len(start_x), 1)),  # type:ignore[arg-type]
        nuclei_coords_X <= np.reshape(end_x, (len(end_x), 1)),  # type:ignore[arg-type]
    )
    num_nuclei_in_crop = np.logical_and(coord_in_Y_bounds, coord_in_X_bounds).sum(axis=1)

    return num_nuclei_in_crop


def get_num_nuclei_in_array(img_arr: np.ndarray | dd.Array, crop: tuple[slice, ...] | None) -> int:
    """
    Get the number of labeled nuclei in an array or dask array.
    Array will be cropped before counting nuclei if crop is provided.
    If there is even 1 pixel of a labeled nuclei then it will be counted,
    therefore you may want to create an image of the centroids or cleaned
    up nuclei before counting.

    Parameters
    ----------
    img_arr : np.ndarray or dd.Array
        The array containing the labeled nuclei.
    crop : tuple[slice, ...] or None
        The crop to apply to the array before counting nuclei.

    Returns
    -------
    int
        The number of unique labeled nuclei in the array.

    Notes
    -----
    This function is not currently used but is still included for convenience.
    """
    if crop is not None:
        img_arr = img_arr[crop]

    if isinstance(img_arr, dd.Array):
        num_nuclei = np.unique(img_arr.compute()).size
        return num_nuclei
    elif isinstance(img_arr, np.ndarray):
        num_nuclei = np.unique(img_arr).size
        return num_nuclei
    else:
        raise TypeError(f"Unsupported type: {type(img_arr)}")


def compute_nuclei_centroids(
    dataset_name: str,
    position: int,
    timeframe: int,
) -> dict:
    """
    Compute the nuclei centroids for a given dataset, position, and timeframe.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to compute the nuclei centroids for.
    position : int
        The position of the dataset to compute the nuclei centroids for.
    timeframe : int
        The timeframe of the dataset to compute the nuclei centroids for.

    Returns
    -------
    dict
        A dictionary containing the centroids of the nuclei in the image.
        The keys include "coords_Y", "coords_X", etc. depending on the
        dimension order as well as "dataset_name", "position", and
        "image_index" (i.e. the timeframe).
    """

    # get the nuclei prediction
    dim_order = DIMENSION_ORDER
    dataset_config = load_dataset_config(dataset_name)
    seg_manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position)
    nuc_seg = load_image(seg_location, squeeze=False, compute=True, timepoints=timeframe)

    # get nuclei segmentation properties and dimension order of those properties
    props = regionprops(nuc_seg.squeeze())
    dim_shapes = dict(zip(dim_order, nuc_seg.shape, strict=True))
    dim_order_squeezed = "".join([d for d in dim_order if dim_shapes[d] > 1])

    centroids: dict[str, Any] = get_nuclei_coords(
        props=props,
        props_dim_order=dim_order_squeezed,
        kind="centroid",
    )
    centroids[Column.DATASET] = dataset_name
    centroids[Column.POSITION] = position
    centroids[Column.TIMEPOINT] = timeframe

    return centroids


def compute_nuclei_centroids_multiproc(args: tuple[str, int, int]) -> dict:
    """
    Wrapper function to compute nuclei centroids for multiprocessing.
    This function is used to unpack the arguments for multiprocessing.

    Parameters
    ----------
    args : tuple[str, int, int]
        A tuple containing the dataset name, position, and timeframe.

    Returns
    -------
    dict
        A dictionary containing the centroids of the nuclei in the image.
        The keys include "coords_Y", "coords_X", etc. depending on the
        dimension order as well as "dataset_name", "position", and
        "image_index" (i.e. the timeframe).
    """
    return compute_nuclei_centroids(*args)


def add_num_nuclei_in_crop_column(
    merged_feats_df: pd.DataFrame,
    use_precomputed: bool = False,
    max_cores: int | None = None,
) -> pd.DataFrame:
    """
    Add the number of nuclei in each crop to the merged features DataFrame.
    This function computes the number of nuclei in each crop by
    computing the nuclei centroids and then counting the number of
    unique nuclei coordinates that fall within the crop bounds.

    Parameters
    ----------
    merged_feats_df : pd.DataFrame
        The DataFrame containing the merged features, which includes
        the crop bounds and the nuclei coordinates.
    use_precomputed : bool, optional
        If True, the function will use precomputed nuclei centroids
        if they are available. If False, the function will always
        compute the nuclei centroids. Default is True.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional column "num_nuclei_in_crop"
        that contains the number of nuclei in each crop.

    Notes
    -----
    The merged_feats_df DataFrame should contain the following columns:
    - "dataset_name": the name of the dataset to analyze
    - "position": the position in the dataset to analyze
    - "image_index": the timeframe in the dataset to analyze
    - "start_y": the start Y coordinate of the crop
    - "end_y": the end Y coordinate of the crop
    - "start_x": the start X coordinate of the crop
    - "end_x": the end X coordinate of the crop
    This function will take a long time to run, so it will save the
    nuclei coordinates to a file locally so that it does not have to
    be computed each time this function is called.
    """
    # get the nuclei coordinates
    nuclei_centroids_dir = get_output_path(__file__, "nuclei_coords", include_timestamp=False)
    dataset_name = sequence_to_scalar(merged_feats_df[Column.DATASET])
    nuclei_centroids_path = nuclei_centroids_dir / f"{dataset_name}_nuclei_centroids.parquet"

    # if the nuclei coordinates are already computed, load them
    if use_precomputed and nuclei_centroids_path.exists():
        nuc_centroid_indices = pd.read_parquet(nuclei_centroids_path)
    # otherwise, compute and save them
    # (this will take about 60 minutes divided by n_cores used)
    else:
        # compute the nuclei prediction centroids
        groups = merged_feats_df.groupby([Column.DATASET, Column.POSITION, Column.TIMEPOINT])
        args = groups.groups.keys()
        if max_cores == 1:
            results = [  # type:ignore[misc]
                compute_nuclei_centroids(dataset_name, position, timeframe)  # type:ignore[has-type]
                for dataset_name, position, timeframe in tqdm(
                    args,
                    desc=f"Computing nuclei centroids (SP): {dataset_name}",
                )
            ]
        else:
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_cores, mp_context=mp_context) as executor:
                results = list(
                    tqdm(
                        executor.map(compute_nuclei_centroids_multiproc, args),
                        total=len(groups),
                        desc=f"Computing nuclei centroids (MP): {dataset_name}",
                    )
                )
        # convert results to DataFrame
        nuc_centroid_indices = pd.DataFrame(results)
        # save results for so this step doesn't have to be rerun each time
        nuc_centroid_indices.to_parquet(nuclei_centroids_path, index=False)

    # combine the nuclei centroids with the merged features DataFrames
    merged_feats_df = pd.merge(
        merged_feats_df,
        nuc_centroid_indices,
        on=[Column.DATASET, Column.POSITION, Column.TIMEPOINT],
        how="left",
    )
    groups = merged_feats_df.groupby([Column.DATASET, Column.POSITION, Column.TIMEPOINT])

    num_nuclei_in_crop = []
    for nm, df in tqdm(groups, desc=f"Counting nuclei in crops: {dataset_name}"):
        # get the number of nuclei in the crops at each timepoint
        num_nuc_centroids = get_num_unique_values_in_bounds_from_df(
            nuclei_coords_Y=np.stack(list(df["coords_Y"])),
            nuclei_coords_X=np.stack(list(df["coords_X"])),
            crop_bounds_Y=(df[Column.SegData.START_Y_RES_0], df[Column.SegData.END_Y_RES_0]),
            crop_bounds_X=(df[Column.SegData.START_X_RES_0], df[Column.SegData.END_X_RES_0]),
        )
        num_nuclei_in_crop.append(pd.Series(num_nuc_centroids, index=df.index))

    merged_feats_df[Column.SegData.NUM_NUCLEI_IN_CROP] = pd.concat(
        num_nuclei_in_crop, axis=0, ignore_index=False
    )
    # drop the nuclei coordinates lists since they are not needed anymore
    merged_feats_df = merged_feats_df.drop(columns=["coords_Y", "coords_X"])
    return merged_feats_df


def get_labels_in_crop(
    segmentation_image: np.ndarray, region_of_interest: tuple[slice, ...]
) -> list:
    """Returns a list of the unique labels that are found in the region of
    interest of the provided segmentation image.
    """
    labels_in_crop = np.unique(segmentation_image[region_of_interest])
    return labels_in_crop.tolist()


def create_labels_in_crop_columns(df_sub: pd.DataFrame, out_dir: Path) -> None:
    """Create an intermediate parquet file with the "all_labels_in_crop" column
    for a subset of the main DataFrame that contains one row per labeled
    segmentation and includes the "all_labels_in_crop" column.

    Note: This function saves parquet tables so that it can be distributed to
    multiple processes to compute the "all_labels_in_crop" column in parallel
    for each timepoint and position, and these parquet tables are then later
    concatenated together and merged with the main DataFrame.

    Parameters
    ----------
    df_sub:
        A subset of the main DataFrame that contains one row per labeled segmentation and includes the "all_labels_in_crop" column.
    out_dir:
        The directory to save the parquet file with the "all_labels_in_crop" column for this subset of the main DataFrame.
    """
    ds_nm = sequence_to_scalar(df_sub[Column.DATASET])
    pos = sequence_to_scalar(df_sub[Column.POSITION])
    tp = sequence_to_scalar(df_sub[Column.TIMEPOINT])

    # load image
    dataset_config = load_dataset_config(ds_nm)
    image_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    image_loc = get_image_location_for_dataset(image_manifest, dataset_config, pos)
    img = load_image(image_loc, compute=True, squeeze=True, timepoints=tp)

    # find other cell labels that are also in the crop
    df_sub[Column.SegData.LABELS_IN_CROP] = df_sub.apply(
        lambda row: get_labels_in_crop(
            segmentation_image=img,
            region_of_interest=(
                slice(row[Column.SegData.START_Y_RES_0], row[Column.SegData.END_Y_RES_0]),
                slice(row[Column.SegData.START_X_RES_0], row[Column.SegData.END_X_RES_0]),
            ),
        ),
        axis=1,
    )

    fname = f"{ds_nm}_pos{pos}_tp{tp}_labels_in_crop.parquet"
    col_subset = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.SegData.LABEL,
        Column.SegData.LABELS_IN_CROP,
    ]
    df_sub[col_subset].to_parquet(out_dir / fname, index=False)


def add_all_labels_in_crop_column(
    big_table: pd.DataFrame, use_precomputed: bool = False, max_cores: int | None = None
) -> pd.DataFrame:
    """Return the provided dataframe with a column added to the DataFrame that
    contains a list of the labels of all segmentations that are found in a
    cell-centric crop.

    Parameters
    ----------
    df:
        The DataFrame to add the column to. This DataFrame should contain the
        following columns: "dataset_name", "position", "image_index", "label",
        "start_y", "end_y", "start_x", and "end_x".
    use_precomputed:
        If True, the function will use precomputed "all_labels_in_crop" data.
        This saves time but should only be used if you are sure that the precomputed
        data is correct and corresponds to the data in the provided DataFrame.
        Default is False.
    max_cores:
        The maximum number of CPU cores to use for multiprocessing when computing
        the labels in a crop. If None, it will use all available cores.
        Only used if use_precomputed = False.
    """
    # make temporary output directory to save "all_labels_in_crop" data
    labels_in_crop_dir = get_output_path(__file__, "labels_in_crop", include_timestamp=False)
    dataset = sequence_to_scalar(big_table[Column.DATASET])
    labels_in_crop_subdir = labels_in_crop_dir / dataset
    labels_in_crop_subdir.mkdir(parents=True, exist_ok=True)
    labels_in_crop_path = labels_in_crop_dir / f"{dataset}_labels_in_crop.parquet"

    df = big_table[big_table[Column.SegDataFilters.IS_VALID_BBOX]]

    if use_precomputed:
        df = pd.read_parquet(labels_in_crop_dir / f"{dataset}_labels_in_crop.parquet")
    else:
        groupby_cols = [Column.DATASET, Column.POSITION, Column.TIMEPOINT]
        _, df_grps = zip(*df.groupby(groupby_cols), strict=True)

        if max_cores == 1:
            for df_grp in tqdm(
                df_grps,
                desc="Creating labels in crop columns (SP)",
            ):
                create_labels_in_crop_columns(df_grp, labels_in_crop_subdir)
        else:
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_cores, mp_context=mp_context) as executor:
                list(
                    tqdm(
                        executor.map(
                            create_labels_in_crop_columns,
                            df_grps,
                            [labels_in_crop_subdir] * len(df_grps),
                        ),
                        total=len(df_grps),
                        desc=f"Creating labels in crop columns (MP): {dataset}",
                    )
                )

        # concatenate all the temporary tables into a single dataframe
        df_lab_in_crop = pd.concat(
            [
                pd.read_parquet(fp)
                for fp in labels_in_crop_subdir.glob("*_pos*_tp*_labels_in_crop.parquet")
            ]
        )

        df = df.merge(
            df_lab_in_crop,
            on=[*groupby_cols, Column.SegData.LABEL],
            how="left",
            validate="one_to_one",
        ).reset_index(drop=True)

        df.to_parquet(labels_in_crop_path, index=False)

        # remove the temporary files and the temporary folder
        for fp in labels_in_crop_subdir.glob("*_pos*_tp*_labels_in_crop.parquet"):
            fp.unlink()
        labels_in_crop_subdir.rmdir()

    return df


def map_label_to_column(df_sub: pd.DataFrame, column_name_to_map: str) -> list:
    """Uses the "all_labels_in_crop" column to map the label of each cell in the
    crop to the value in the specified column for that label, and returns a list
    of those values for each cell in the crop as a list.

    Parameters
    ----------
    df_sub:
        A subset of the main DataFrame that contains one row per labeled segmentation
        and includes the "all_labels_in_crop" column.
    column_name_to_map:
        The name of the column to map the labels to. This column should be
        present in df_sub and should contain the value that you want to map for
        each label.

    Returns
    -------
    list:
        A list where each entry is a list of the values from the specified column
        for all the labels in the crop of that row.
    """
    check_required_columns_in_dataframe(
        df_sub,
        required_columns=[Column.SegData.LABEL, column_name_to_map, Column.SegData.LABELS_IN_CROP],
    )
    label_velocity_dict = dict(
        zip(df_sub[Column.SegData.LABEL], df_sub[column_name_to_map], strict=True)
    )
    df_sub[Column.SegData.LABELS_IN_CROP] = df_sub[Column.SegData.LABELS_IN_CROP].transform(
        lambda ls: [] if isinstance(ls, type(None)) else ls
    )
    return df_sub[Column.SegData.LABELS_IN_CROP].map(lambda ls: [*map(label_velocity_dict.get, ls)])


def sanitize_list_to_numbers(ls: list) -> list:
    """Returns the provided list with all empty, None, and non-finite values removed."""
    return [x for x in ls if x and np.isfinite(x)]


def add_vector_mean_of_migration_in_crop_column(
    df: pd.DataFrame, velocity_column_name: str
) -> pd.DataFrame:
    """
    Returns the provided dataframe with two new columns:
    - "vec_mean_angle_in_crop": the vector mean of the migration angles of all cells in the crop
    - "vec_mean_mag_in_crop": the vector mean of the migration magnitudes of all cells in the crop
    """

    df[Column.SegData.VELOCITY_ANGLES_IN_CROP] = (
        df.groupby([Column.DATASET, Column.POSITION, Column.TIMEPOINT])
        .apply(lambda df_sub: pd.DataFrame(map_label_to_column(df_sub, velocity_column_name)))
        .droplevel([0, 1, 2])
    )

    # calculate the vector means of all cells within the crop
    df[Column.SegData.VELOCITY_ANGLES_IN_CROP] = df[
        Column.SegData.VELOCITY_ANGLES_IN_CROP
    ].transform(sanitize_list_to_numbers)

    df[
        [Column.SegData.VECTOR_MEAN_FOR_CROP_ANGLE, Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE]
    ] = pd.DataFrame(
        df[Column.SegData.VELOCITY_ANGLES_IN_CROP]
        .transform(
            lambda angles: (
                vector_mean_angle_and_mag(angles) if len(angles) > 1 else (np.nan, np.nan)
            )
        )
        .tolist(),
        index=df.index,
    )

    return df


def add_normalized_time(
    df_all_positions: pd.DataFrame,
    time_col: str = Column.SegData.TIME_HRS,
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

    for _, df_pos in df_all_positions.groupby(Column.POSITION):
        for _, df_track in df_pos.groupby(Column.TRACK_ID):

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
                Column.SegData.NORMALIZED_TIME_PER_TRACK,
            ] = normalized_time_values

    return df_all_positions
