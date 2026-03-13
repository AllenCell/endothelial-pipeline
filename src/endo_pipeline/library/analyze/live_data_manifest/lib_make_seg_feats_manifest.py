import logging
import math
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
from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.lib_init_density_vs_flow import vector_mean_angle_and_mag
from endo_pipeline.library.model.eval_model import add_diffae_model_eval_crop_columns
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest
from endo_pipeline.settings.image_data import (
    DIMENSION_ORDER,
    IMG_SHAPE_RESOLUTION_0_3i_X,
    IMG_SHAPE_RESOLUTION_0_3i_Y,
)
from endo_pipeline.settings.segmentation_feature_dataframes import ColumnNameSeg as ColNmSeg

logger = logging.getLogger(__name__)


def merge_measured_segmentation_features_tables(
    cellprops_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    nucprops_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function merges the outputs from the tracking
    workflow (cdh5_classic_seg_tracking.py), the
    segmentation measurement workflow
    (cdh5_get_measured_features.py), and the labelfree
    nuclei measurement workflow (nuc_get_measured_features.py).
    """
    big_table = pd.merge(
        left=tracking_df,
        right=cellprops_df,
        left_on=["dataset_name", "position", "T", "label"],
        right_on=["dataset_name", "position", "T", "cell_label"],
    )
    big_table = pd.merge(
        left=big_table,
        right=nucprops_df,
        left_on=["dataset_name", "position", "T", "label"],
        right_on=["dataset_name", "position", "T", "cdh5_segmentation_label"],
    )

    big_table = remove_redundant_columns(big_table)

    big_table = sanitize_column_names(big_table)

    return big_table


def remove_redundant_columns(big_table: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated columns resulting from the merging of dataframes."""
    # the following columns are redundant with another in the table and
    # can be dropped:
    duplicate_cols = [
        "T",  # redundant with "image_index"
        "label",  # redundant with "cell_label"
        "cdh5_segmentation_label",  # redundant with "label"
        "centroid",  # redundant with "cell_centroid"
        "area",  # redundant with "cell_area (px**2)"
        "perimeter",  # redundant with "cell_perimeter (px)"
        "eccentricity",  # redundant with "cell_eccentricity"
        "touches_border",  # redundant with "touches_image_border"
        "orientation",  # redundant with "cell orientation"; though has a different phase shift
        "centroid",  # redundant with "cell_centroid"
        "centroid_X",
        "centroid_Y",
    ]

    return big_table.drop(columns=duplicate_cols)


def sanitize_column_names(big_table: pd.DataFrame) -> pd.DataFrame:
    """Make the column names consistent with elsewhere in the code base."""
    # NOTE: maybe you don't need to rename ALL of the columns and can instead just
    # rename the ones that are shared with the dynamics workflow
    dataset_info_cols = {
        "dataset_name": ColNmSeg.DATASET,
        "position": ColNmSeg.POSITION,
        "image_index": ColNmSeg.TIMEPOINT,
        "track_id": ColNmSeg.TRACK_ID,
        "cell_label": ColNmSeg.LABEL,
        "num_unique_tracks_after_filtering_at_T": ColNmSeg.NUM_TRACKS_AFTER_FILTERING,
        "num_unique_tracks_before_filtering_at_T": ColNmSeg.NUM_TRACKS_BEFORE_FILTERING,
        "shear_stress_regime": ColNmSeg.SHEAR_STRESS_REGIME,
    }
    filter_cols = {
        "is_included": ColNmSeg.IS_INCLUDED,
        "touches_image_border": ColNmSeg.IS_EDGE_SEGMENTATION,
        "is_less_than_max_smoothed_area_normd_change": ColNmSeg.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE,
        "is_greater_than_min_track_duration": ColNmSeg.IS_GREATER_THAN_MIN_TRACK_DURATION,
        "has_more_than_min_num_valid_points_per_track": ColNmSeg.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK,
        "bbox_is_in_bounds": ColNmSeg.IS_VALID_BBOX,
    }
    # annotation_cols_to_rename = {
    #     "auto_bf_scope_error": ColNmSeg.AUTO_BF_SCOPE_ERROR,
    #     "auto_bf_temp_artifact": ColNmSeg.AUTO_BF_TEMP_ARTIFACT,
    #     "auto_gfp_scope_error": ColNmSeg.AUTO_GFP_SCOPE_ERROR,
    #     "bf_scope_error": ColNmSeg.BF_SCOPE_ERROR,
    #     "bf_temp_artifact": ColNmSeg.BF_TEMP_ARTIFACT,
    #     "gfp_scope_error": ColNmSeg.GFP_SCOPE_ERROR,
    #     "cell_piling": ColNmSeg.CELL_PILING,
    #     "not_steady_state": ColNmSeg.NOT_STEADY_STATE,
    # }
    temporal_feature_cols = {
        "time_hours": ColNmSeg.TIME_HRS,
        "time_minutes": ColNmSeg.TIME_MINS,
        "track_duration": ColNmSeg.TRACK_LENGTH,
    }
    morpho_feature_cols = {
        "cell_orientation": ColNmSeg.ORIENTATION,
        "alignment_rel_to_flow": ColNmSeg.ALIGNMENT,
        "alignment_deg_rel_to_flow": ColNmSeg.ALIGNMENT_DEG,
        "orientation_deg": ColNmSeg.ORIENTATION_DEG,
        "nematic_order": ColNmSeg.NEMATIC_ORDER,
        "aspect_ratio": ColNmSeg.ASPECT_RATIO,
        "cell_eccentricity": ColNmSeg.ECCENTRICITY,
        "major_axis_length": ColNmSeg.MAJOR_AXIS,
        "minor_axis_length": ColNmSeg.MINOR_AXIS,
        "cell_solidity": ColNmSeg.SOLIDITY,
        "cell_area (px**2)": ColNmSeg.AREA_PX_SQ,
        "cell_perimeter (px)": ColNmSeg.PERIMETER_PX,
        "nucpos_rel_cell_X": ColNmSeg.NUCLEI_POSITION_X,
        "nucpos_rel_cell_Y": ColNmSeg.NUCLEI_POSITION_Y,
        "nucpos_rel_cell_angle": ColNmSeg.NUCLEI_POSITION_ANGLE,
        "nucpos_rel_cell_angle_deg": ColNmSeg.NUCLEI_POSITION_ANGLE_DEG,
        "nuc_pos_rel_cell_magnitude": ColNmSeg.NUCLEI_POSITION_DISTANCE,
    }
    fluorescence_feature_cols = {
        "edge_fluorescences (a.u.)": ColNmSeg.EDGE_FLUOR,
        "node_fluorescences (a.u.)": ColNmSeg.NODE_FLUOR,
        "cell_fluorescence_mean (a.u.)": ColNmSeg.CELL_FLUOR_MEAN,
        "cell_fluorescence_std (a.u.)": ColNmSeg.CELL_FLUOR_STD,
        "cell_fluorescence_median (a.u.)": ColNmSeg.CELL_FLUOR_MEDIAN,
        "cell_fluorescence_min (a.u.)": ColNmSeg.CELL_FLUOR_MIN,
        "cell_fluorescence_max (a.u.)": ColNmSeg.CELL_FLUOR_MAX,
        "cell_fluorescence_pct25 (a.u.)": ColNmSeg.CELL_FLUOR_PCT25,
        "cell_fluorescence_pct75 (a.u.)": ColNmSeg.CELL_FLUOR_PCT75,
        # "edge_fluorescence_means (a.u.)": ColNmSeg.EDGE_FLUOR_MEAN,
        # "node_fluorescence_means (a.u.)": ColNmSeg.NODE_FLUOR_MEAN,
        # "edge_and_node_fluorescence_means (a.u.)": ColNmSeg.EDGE_AND_NODE_FLUOR_MEAN,
        # "edge_fluorescence_std (a.u.)": ColNmSeg.EDGE_FLUOR_STD,
        # "node_fluorescence_std (a.u.)": ColNmSeg.NODE_FLUOR_STD,
        # "edge_and_node_fluorescence_std (a.u.)": ColNmSeg.EDGE_AND_NODE_FLUOR_STD,
    }
    crop_based_feature_cols = {
        "num_nuclei_in_crop": ColNmSeg.NUM_NUCLEI_IN_CROP,
        "all_labels_in_crop": ColNmSeg.LABELS_IN_CROP,
        # "start_x": ColNmSeg.START_X,
        # "start_y": ColNmSeg.START_Y,
        # "end_x": ColNmSeg.END_X,
        # "end_y": ColNmSeg.END_Y,
        # "crop_size": ColNmSeg.CROP_SIZE,
        "filepath_raw_image": ColNmSeg.TIMELAPSE_PATH,
    }
    other_feature_cols = {
        "number_of_neighbors": ColNmSeg.NUM_NEIGHBORS,
        "neighboring_cell_labels": ColNmSeg.NEIGHBOR_LABELS,
        "cell_centroid": ColNmSeg.CENTROID,
        "filepath_segmentation_image": ColNmSeg.SEGMENTATION_PATH,
    }
    cols_to_rename = {
        **dataset_info_cols,
        **filter_cols,
        **temporal_feature_cols,
        **morpho_feature_cols,
        **fluorescence_feature_cols,
        **crop_based_feature_cols,
        **other_feature_cols,
    }

    return big_table.rename(columns=cols_to_rename)


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
    for (dataset_nm, position), df in big_table_filtered.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION]
    ):
        summary = df.groupby(ColNmSeg.TIMEPOINT)[
            [
                ColNmSeg.TIMEPOINT,
                ColNmSeg.NUM_TRACKS_BEFORE_FILTERING,
                ColNmSeg.NUM_TRACKS_AFTER_FILTERING,
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
            x=ColNmSeg.TIMEPOINT,
            y=ColNmSeg.NUM_TRACKS_BEFORE_FILTERING,
            data=summary,
            ax=ax,
            label="Before filtering",
        )
        sns.lineplot(
            x=ColNmSeg.TIMEPOINT,
            y=ColNmSeg.NUM_TRACKS_AFTER_FILTERING,
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
        big_table.groupby([ColNmSeg.DATASET, ColNmSeg.POSITION])[ColNmSeg.TRACK_ID].nunique().sum()
    )

    # keep only tracks with duration longer than min_track_duration
    big_table[ColNmSeg.MIN_TRACK_DURATION] = min_track_duration
    big_table[ColNmSeg.IS_GREATER_THAN_MIN_TRACK_DURATION] = (
        big_table[ColNmSeg.TRACK_LENGTH] > min_track_duration
    )

    # keep only tracks where area_change is not too large
    big_table[ColNmSeg.MAX_SMOOTHED_AREA_NORMALIZED_CHANGE] = max_area_change
    big_table[ColNmSeg.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE] = (
        big_table[ColNmSeg.SMOOTHED_AREA_NORMD_DIFF].abs() < max_area_change
    )

    # is_included is just all the previous filters combined with the
    # filter to exclude segmentations that touch the edges of the image
    big_table[ColNmSeg.IS_INCLUDED] = (
        big_table[ColNmSeg.IS_GREATER_THAN_MIN_TRACK_DURATION]
        & big_table[ColNmSeg.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE]
        & ~big_table[ColNmSeg.IS_EDGE_SEGMENTATION]
    )

    # drop because there are insufficient valid timepoints
    big_table[ColNmSeg.NUM_VALID_TIMEPOINTS_IN_TRACK] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TRACK_ID]
    )[ColNmSeg.IS_INCLUDED].transform(sum)
    big_table[ColNmSeg.MIN_NUM_VALID_TIMEPOINTS_PER_TRACK] = min_num_valid_points_per_track
    big_table[ColNmSeg.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK] = (
        big_table[ColNmSeg.NUM_VALID_TIMEPOINTS_IN_TRACK] > min_num_valid_points_per_track
    )

    # update is_included column with valid_tp_per_track
    big_table[ColNmSeg.IS_INCLUDED] = (
        big_table[ColNmSeg.IS_INCLUDED]
        & big_table[ColNmSeg.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK]
    )

    # get the number of unique tracks after filtering in total and per timepoint
    num_rows_after_filtering = np.count_nonzero(big_table[ColNmSeg.IS_INCLUDED])
    num_unique_tracks_after_filtering = (
        big_table[big_table[ColNmSeg.IS_INCLUDED]]
        .groupby([ColNmSeg.DATASET, ColNmSeg.POSITION])[ColNmSeg.TRACK_ID]
        .nunique()
        .sum()
    )
    big_table[ColNmSeg.NUM_TRACKS_AFTER_FILTERING] = (
        big_table[big_table[ColNmSeg.IS_INCLUDED]]
        .groupby([ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT])[ColNmSeg.TRACK_ID]
        .transform(lambda x: x.nunique())
    )

    # save a log file of the filtering that was done if saving the results
    if out_dir:
        # save a log file and create some plots showing number of
        # tracks before and after filtering
        datasets_analyzed = big_table[ColNmSeg.DATASET].unique().tolist()
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
            big_table[big_table[ColNmSeg.IS_INCLUDED]],
            min_track_duration,
        )
    return big_table


def add_cell_piling_and_steady_state_annotation_columns(big_table: pd.DataFrame) -> pd.DataFrame:
    """Adds the annotations about cell piling and steady state that were done by
    hand as columns to the data table.
    """
    # load dataset config and timepoint annotations
    dataset = sequence_to_scalar(big_table[ColNmSeg.DATASET])
    dataset_config = load_dataset_config(dataset)
    if dataset_config.timepoint_annotations is not None:
        filters_for_dataset = list(dataset_config.timepoint_annotations.keys())
        for filt in filters_for_dataset:
            # add the timepoint annotations as filter columns
            big_table[filt] = (
                big_table.groupby(ColNmSeg.POSITION, as_index=True)
                .apply(
                    lambda df, filt=filt: (
                        pd.DataFrame(
                            (
                                df[ColNmSeg.TIMEPOINT].isin(
                                    get_annotated_timepoints_for_position(
                                        dataset_config,
                                        sequence_to_scalar(df[ColNmSeg.POSITION]),
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


def calculate_derived_data_dynamics_independent(big_table: pd.DataFrame) -> pd.DataFrame:
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
    dataset_name = sequence_to_scalar(big_table[ColNmSeg.DATASET])
    data_config = load_dataset_config(dataset_name)

    # add the shear stress regime to the data table
    logger.info("Adding shear stress regime...")
    shear_stress_regime = "_to_".join([shear.value for shear in data_config.shear_stress_regime])
    big_table[ColNmSeg.SHEAR_STRESS_REGIME] = shear_stress_regime

    # dimensionalize the time column
    logger.info("Adding time intervals per timepoint...")
    big_table[ColNmSeg.TIME_RESOLUTION_MINUTES] = data_config.time_interval_in_minutes

    logger.info("Calculating time in minutes and hours...")
    big_table[ColNmSeg.TIME_MINS] = (
        big_table[ColNmSeg.TIMEPOINT] * big_table[ColNmSeg.TIME_RESOLUTION_MINUTES]
    )
    big_table[ColNmSeg.TIME_HRS] = big_table[ColNmSeg.TIME_MINS] / 60

    # add a column for the number of unique tracks
    # per dataset per position per timepoint
    # (this should be 1 everywhere)
    big_table[ColNmSeg.NUM_UNIQUE_TRACKS_PER_TIMEPOINT] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT, ColNmSeg.TRACK_ID]
    ).transform("size")

    # add the columns for the fold change in area
    logger.info("Calculating locally-normalized area...")
    sigma = 2.0
    big_table[ColNmSeg.SIGMA_FOR_AREA_SMOOTHING] = sigma
    big_table[ColNmSeg.SMOOTHED_AREA_NORMALIZED] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TRACK_ID]
    )[ColNmSeg.AREA_PX_SQ].transform(
        lambda x: calculate_smoothed_normd_area(x, smoothing_sigma=sigma)
    )
    big_table[ColNmSeg.SMOOTHED_AREA_NORMD_DIFF] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TRACK_ID]
    )[ColNmSeg.SMOOTHED_AREA_NORMALIZED].transform(lambda x: x.diff())

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    logger.info("Adding number of tracks for each timepoint...")
    big_table[ColNmSeg.NUM_TRACKS_BEFORE_FILTERING] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT]
    )[ColNmSeg.TRACK_ID].transform(lambda x: x.nunique())

    # add the duration of each track
    logger.info("Calculating track durations...")
    big_table[ColNmSeg.TRACK_LENGTH] = big_table.groupby(
        [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TRACK_ID]
    )[ColNmSeg.TIMEPOINT].transform(lambda t: t.max() - t.min())

    # add column for orientation in degrees of the
    # ellipse fitted to each segmentation in degrees
    logger.info("Converting orientation to degrees...")
    big_table[ColNmSeg.ALIGNMENT] = big_table[ColNmSeg.ORIENTATION].transform(
        lambda x: make_orientation_relative_to_flow(x)
    )
    big_table[ColNmSeg.ALIGNMENT_DEG] = np.rad2deg(big_table[ColNmSeg.ALIGNMENT])

    # add column for the orientation in degrees
    big_table[ColNmSeg.ORIENTATION_DEG] = np.rad2deg(big_table[ColNmSeg.ORIENTATION])

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    logger.info("Calculating nematic order and aspect ratio...")
    big_table[ColNmSeg.NEMATIC_ORDER] = big_table[ColNmSeg.ORIENTATION].transform(get_nematic_order)
    big_table[ColNmSeg.ASPECT_RATIO] = big_table[ColNmSeg.ECCENTRICITY].transform(get_aspect_ratio)

    # add pixel sizes
    big_table[ColNmSeg.PIXEL_SIZE_XY_IN_UM] = data_config.pixel_size_xy_in_um
    big_table[ColNmSeg.AREA] = (
        big_table[ColNmSeg.AREA_PX_SQ] * big_table[ColNmSeg.PIXEL_SIZE_XY_IN_UM] ** 2
    )
    big_table[ColNmSeg.PERIMETER] = (
        big_table[ColNmSeg.PERIMETER_PX] * big_table[ColNmSeg.PIXEL_SIZE_XY_IN_UM]
    )

    # compute intensity means and standard deviations for edge and node pixels
    # separately and together
    big_table[ColNmSeg.EDGE_FLUOR_MEAN] = big_table[ColNmSeg.EDGE_FLUOR].transform(
        lambda x: x.mean()
    )
    big_table[ColNmSeg.EDGE_FLUOR_STD] = big_table[ColNmSeg.EDGE_FLUOR].transform(lambda x: x.std())
    big_table[ColNmSeg.NODE_FLUOR_MEAN] = big_table[ColNmSeg.NODE_FLUOR].transform(
        lambda x: x.mean()
    )
    big_table[ColNmSeg.NODE_FLUOR_STD] = big_table[ColNmSeg.NODE_FLUOR].transform(lambda x: x.std())
    big_table[ColNmSeg.EDGE_AND_NODE_FLUOR_MEAN] = big_table.apply(
        lambda row: np.mean(row[ColNmSeg.EDGE_FLUOR].tolist() + row[ColNmSeg.NODE_FLUOR].tolist()),
        axis=1,
    )
    big_table[ColNmSeg.EDGE_AND_NODE_FLUOR_STD] = big_table.apply(
        lambda row: np.std(row[ColNmSeg.EDGE_FLUOR].tolist() + row[ColNmSeg.NODE_FLUOR].tolist()),
        axis=1,
    )

    # add a column for the number of neighbors
    # touching each region that is being tracked
    logger.info("Calculating number of neighbors...")
    big_table[ColNmSeg.NUM_NEIGHBORS] = big_table[ColNmSeg.NEIGHBOR_LABELS].transform(
        lambda x: len(x)
    )

    # add the image size and channel indices to the data table
    big_table[ColNmSeg.IMAGE_SIZE_X] = IMG_SHAPE_RESOLUTION_0_3i_X
    big_table[ColNmSeg.IMAGE_SIZE_Y] = IMG_SHAPE_RESOLUTION_0_3i_Y
    big_table[ColNmSeg.CDH5_CHANNEL_INDEX_ZARR] = data_config.zarr_channel_indices.channel_488
    big_table[ColNmSeg.BF_CHANNEL_INDEX_ZARR] = data_config.zarr_channel_indices.brightfield

    # add the number of nuclei that overlap the most with each cell
    # (this can be used as a filter later so we only measure cells
    # with a single clearly distinguishable nuclei)
    big_table[ColNmSeg.NUM_NUC_WITH_MOST_OVERLAP] = big_table[
        ColNmSeg.NUCLEI_LABELS_IN_CDH5_SEGMENTATION
    ].transform(len)

    # split the centroid column into separate x and y columns
    big_table[[ColNmSeg.CENTROID_Y, ColNmSeg.CENTROID_X]] = pd.DataFrame(
        big_table[ColNmSeg.CENTROID].tolist(), index=big_table.index
    )

    # add the nuclei centroids relative to the cell centroids
    big_table[ColNmSeg.NUCLEI_POSITION_X], big_table[ColNmSeg.NUCLEI_POSITION_Y] = (
        get_nuclei_rel_to_cell_position(
            big_table[ColNmSeg.CENTROID_X],
            big_table[ColNmSeg.CENTROID_Y],
            big_table[ColNmSeg.NUCLEI_CENTROID_X],
            big_table[ColNmSeg.NUCLEI_CENTROID_Y],
        )
    )

    # get the angles and distances of the nuclei relative positions
    big_table[ColNmSeg.NUCLEI_POSITION_DISTANCE] = np.linalg.norm(
        [big_table[ColNmSeg.NUCLEI_POSITION_X], big_table[ColNmSeg.NUCLEI_POSITION_Y]], axis=0
    )
    big_table[ColNmSeg.NUCLEI_POSITION_ANGLE] = np.arctan2(
        big_table[ColNmSeg.NUCLEI_POSITION_Y], big_table[ColNmSeg.NUCLEI_POSITION_X]
    )
    big_table[ColNmSeg.NUCLEI_POSITION_ANGLE_DEG] = np.rad2deg(
        big_table[ColNmSeg.NUCLEI_POSITION_ANGLE]
    )

    # add the DiffAE crop locations and binning level; these can be used to load
    # a crop from the zarr files and compute the number of nuclei in that crop
    big_table = add_diffae_model_eval_crop_columns(big_table)

    # compute the number of nuclei found in a defined crop size
    # (first take a subset using only the required columns to reduce memory usage)
    required_columns = [
        ColNmSeg.DATASET,
        ColNmSeg.POSITION,
        ColNmSeg.TIMEPOINT,
        ColNmSeg.TRACK_ID,
        ColNmSeg.LABEL,
        ColNmSeg.CENTROID_Y,
        ColNmSeg.CENTROID_X,
        ColNmSeg.IMAGE_SIZE_Y,
        ColNmSeg.IMAGE_SIZE_X,
        ColNmSeg.CROP_SIZE,
        ColNmSeg.START_Y,
        ColNmSeg.END_Y,
        ColNmSeg.START_X,
        ColNmSeg.END_X,
        ColNmSeg.IS_VALID_BBOX,
    ]
    num_nuclei_in_crop_df = add_num_nuclei_in_crop_column(
        big_table[required_columns], use_precomputed=False
    )
    crops = [
        str(col)
        for col in [
            ColNmSeg.DATASET,
            ColNmSeg.POSITION,
            ColNmSeg.TIMEPOINT,
            ColNmSeg.TRACK_ID,
        ]
    ]  # this is needed to avoid a mypy error about the type of the columns when merging
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
        big_table[required_columns], use_precomputed=False
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
    max_timeframes_to_average_for_velocity: int = 5,
) -> pd.DataFrame:
    """
    This function calculates dynamics-dependent features and
    adds them to the main segmentation features table.
    Added columns include:
    - centroid_dx_dt: cdh5-based cell segmentation centroid velocity in x (units in um/min)
    - centroid_dy_dt: cdh5_based cell segmentation centroid velocity in y (units in um/min)
    - centroid_velocity_magnitude: magnitude of the cdh5-based cell segmentation centroid velocity
    - centroid_velocity_angle: the angle of the cdh5-based cell segmentation centroid velocity (from -pi to pi with 0 being to the right)
    - dalignment_dt_deg_rel_to_flow: the change in alignment angle (in degrees/min)
    - num_tracks_at_T: the number of tracks at a given timepoint per dataset per position
        (this number is affected by any filtering that was done to the passed in table)

    NOTE: The accuracy of these metrics are affected by how
    clean the data in the table is, therefore it should only
    be used after filtering out incorrect segmentations from
    the data table.
    """
    # recalculate the centroid speeds of each track
    # after filtering
    logger.info("Calculating centroid velocities...")
    big_table["centroid_x_um"] = big_table["centroid_X"] * big_table["pixel_size_xy_in_um"]
    big_table["centroid_y_um"] = big_table["centroid_Y"] * big_table["pixel_size_xy_in_um"]
    big_table[["centroid_dx_dt", "centroid_dy_dt"]] = (
        big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)[
            ["centroid_x_um", "centroid_y_um", "time_minutes"]
        ]
        .apply(
            lambda df: pd.DataFrame(  # type: ignore[arg-type, return-value]
                columns=["centroid_dx_dt", "centroid_dy_dt"],
                data=zip(
                    *get_centroid_velocity(
                        df["centroid_x_um"].values,  # type: ignore[arg-type, call-overload, return-value]
                        df["centroid_y_um"].values,  # type: ignore[arg-type, call-overload, return-value]
                        df["time_minutes"].values,  # type: ignore[arg-type, call-overload, return-value]
                    ),
                    strict=True,
                ),  # type: ignore[return-value]
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    # get the windowed mean of the centroid velocities to smooth out noise
    for window in range(2, max_timeframes_to_average_for_velocity + 1):
        big_table["time_minutes_timedelta"] = pd.to_timedelta(big_table["time_minutes"], unit="m")
        window_in_minutes = window * sequence_to_scalar(big_table["time_resolution_minutes"])

        big_table[f"centroid_dx_dt_rolling_mean_window_{window_in_minutes}min"] = (
            big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)[
                ["centroid_dx_dt", "time_minutes_timedelta"]
            ].apply(
                lambda df, window_in_minutes=window_in_minutes: df.rolling(
                    f"{window_in_minutes}min", min_periods=1, on="time_minutes_timedelta"
                )["centroid_dx_dt"].mean()
            )
        ).droplevel([0, 1, 2])

        big_table[f"centroid_dy_dt_rolling_mean_window_{window_in_minutes}min"] = (
            big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)[
                ["centroid_dy_dt", "time_minutes_timedelta"]
            ].apply(
                lambda df, window_in_minutes=window_in_minutes: df.rolling(
                    f"{window_in_minutes}min", min_periods=1, on="time_minutes_timedelta"
                )["centroid_dy_dt"].mean()
            )
        ).droplevel([0, 1, 2])

    logger.info("Calculating centroid velocity magnitude and angle...")
    big_table["centroid_velocity_magnitude"] = np.linalg.norm(
        [big_table["centroid_dx_dt"], big_table["centroid_dy_dt"]], axis=0
    )
    big_table["centroid_velocity_angle"] = np.arctan2(
        big_table["centroid_dy_dt"], big_table["centroid_dx_dt"]
    )
    big_table["centroid_velocity_angle_deg"] = np.rad2deg(big_table["centroid_velocity_angle"])

    big_table["dalignment_dt_deg_rel_to_flow"] = (
        big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df["alignment_deg_rel_to_flow"].diff() / df["time_minutes"].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    big_table["cell_nuc_orientation_deg_rel_to_migration"] = get_smallest_angle_difference(
        big_table["nuc_pos_rel_cell_angle_deg"], big_table["centroid_velocity_angle_deg"]
    )

    big_table["nuc_pos_rel_cell_X_um"] = (
        big_table["nuc_pos_rel_cell_X"] * big_table["pixel_size_xy_in_um"]
    )
    big_table["nuc_pos_rel_cell_Y_um"] = (
        big_table["nuc_pos_rel_cell_Y"] * big_table["pixel_size_xy_in_um"]
    )
    big_table["nuc_pos_vs_cell_veloc_dotprod"] = np.einsum(
        "ij,ij->i",
        big_table[["centroid_dx_dt", "centroid_dy_dt"]],
        big_table[["nuc_pos_rel_cell_X_um", "nuc_pos_rel_cell_Y_um"]],
    )

    # add fluorescence intensity dynamics column
    logger.info("Calculating fluorescence intensity dynamics...")
    big_table["dmean_EGFP_intensity_dt"] = (
        big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df["cell_fluorescence_mean (a.u.)"].diff() / df["time_minutes"].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    # add approximate cell density dynamics column
    logger.info("Calculating approximate cell density dynamics...")
    big_table["dnum_nuclei_in_crop_dt"] = (
        big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)
        .apply(
            lambda df: pd.DataFrame(
                df["num_nuclei_in_crop"].diff() / df["time_minutes"].diff(),
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    if compute_per_crop_metrics:
        # find the migration vectors for all cells in a crop
        logger.info("Calculating vector mean of migration per crop...")
        big_table = add_vector_mean_of_migration_in_crop_column(big_table)

    # add column for the number of tracks at a given timepoint per dataset per position
    logger.info("Adding number of tracks for each timepoint...")
    big_table["num_tracks_at_T"] = big_table.groupby(["dataset_name", "position", "T"])[
        "track_id"
    ].transform(lambda x: x.nunique())

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
    nuclei_coords_arrs = {dim: np.stack(coords).squeeze() for dim, coords in nuclei_coords.items()}

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
    centroids[ColNmSeg.DATASET] = dataset_name
    centroids[ColNmSeg.POSITION] = position
    centroids[ColNmSeg.TIMEPOINT] = timeframe

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
    dataset_name = sequence_to_scalar(merged_feats_df[ColNmSeg.DATASET])
    nuclei_centroids_path = nuclei_centroids_dir / f"{dataset_name}_nuclei_centroids.parquet"

    # if the nuclei coordinates are already computed, load them
    if use_precomputed and nuclei_centroids_path.exists():
        nuc_centroid_indices = pd.read_parquet(nuclei_centroids_path)
    # otherwise, compute and save them
    # (this will take about 60 minutes divided by n_cores used)
    else:
        # compute the nuclei prediction centroids
        groups = merged_feats_df.groupby([ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT])
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
            with ProcessPoolExecutor(max_workers=max_cores) as executor:
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
        on=[ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT],
        how="left",
    )
    groups = merged_feats_df.groupby([ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT])

    num_nuclei_in_crop = []
    for nm, df in tqdm(groups, desc=f"Counting nuclei in crops: {dataset_name}"):
        # get the number of nuclei in the crops at each timepoint
        num_nuc_centroids = get_num_unique_values_in_bounds_from_df(
            nuclei_coords_Y=np.stack(list(df["coords_Y"])),
            nuclei_coords_X=np.stack(list(df["coords_X"])),
            crop_bounds_Y=(df[ColNmSeg.START_Y], df[ColNmSeg.END_Y]),
            crop_bounds_X=(df[ColNmSeg.START_X], df[ColNmSeg.END_X]),
        )
        num_nuclei_in_crop.append(pd.Series(num_nuc_centroids, index=df.index))

    merged_feats_df[ColNmSeg.NUM_NUCLEI_IN_CROP] = pd.concat(
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
    ds_nm = sequence_to_scalar(df_sub[ColNmSeg.DATASET])
    pos = sequence_to_scalar(df_sub[ColNmSeg.POSITION])
    tp = sequence_to_scalar(df_sub[ColNmSeg.TIMEPOINT])

    # load image
    dataset_config = load_dataset_config(ds_nm)
    image_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    image_loc = get_image_location_for_dataset(image_manifest, dataset_config, pos)
    img = load_image(image_loc, compute=True, squeeze=True, timepoints=tp)

    # find other cell labels that are also in the crop
    df_sub[ColNmSeg.LABELS_IN_CROP] = df_sub.apply(
        lambda row: get_labels_in_crop(
            segmentation_image=img,
            region_of_interest=(
                slice(row[ColNmSeg.START_Y], row[ColNmSeg.END_Y]),
                slice(row[ColNmSeg.START_X], row[ColNmSeg.END_X]),
            ),
        ),
        axis=1,
    )

    fname = f"{ds_nm}_pos{pos}_tp{tp}_labels_in_crop.parquet"
    col_subset = [
        ColNmSeg.DATASET,
        ColNmSeg.POSITION,
        ColNmSeg.TIMEPOINT,
        ColNmSeg.LABEL,
        ColNmSeg.LABELS_IN_CROP,
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
    dataset = sequence_to_scalar(big_table[ColNmSeg.DATASET])
    labels_in_crop_subdir = labels_in_crop_dir / dataset
    labels_in_crop_subdir.mkdir(parents=True, exist_ok=True)
    labels_in_crop_path = labels_in_crop_dir / f"{dataset}_labels_in_crop.parquet"

    df = big_table[big_table[ColNmSeg.IS_VALID_BBOX]]

    if use_precomputed:
        df = pd.read_parquet(labels_in_crop_dir / f"{dataset}_labels_in_crop.parquet")
    else:
        groupby_cols = [ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT]
        _, df_grps = zip(*df.groupby(groupby_cols), strict=True)

        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            list(
                tqdm(
                    executor.map(
                        create_labels_in_crop_columns,
                        df_grps,
                        [labels_in_crop_subdir] * len(df_grps),
                    ),
                    total=len(df_grps),
                    desc=f"Creating labels in crop columns: {dataset}",
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
            df_lab_in_crop, on=[*groupby_cols, ColNmSeg.LABEL], how="left", validate="one_to_one"
        ).reset_index(drop=True)

        df.to_parquet(labels_in_crop_path, index=False)

        # remove the temporary files and the temporary folder
        for fp in labels_in_crop_subdir.glob("*_pos*_tp*_labels_in_crop.parquet"):
            fp.unlink()
        labels_in_crop_subdir.rmdir()

    return df


def map_label_to_column(
    df_sub: pd.DataFrame, column_name_to_map: str = "centroid_velocity_angle"
) -> list:
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
        df_sub, required_columns=["label", column_name_to_map, "all_labels_in_crop"]
    )
    label_velocity_dict = dict(zip(df_sub.label, df_sub[column_name_to_map], strict=True))
    return df_sub.all_labels_in_crop.map(lambda ls: [*map(label_velocity_dict.get, ls)])


def sanitize_list_to_numbers(ls: list) -> list:
    """Returns the provided list with all empty, None, and non-finite values removed."""
    return [x for x in ls if x and np.isfinite(x)]


def add_vector_mean_of_migration_in_crop_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the provided dataframe with two new columns:
    - "vec_mean_angle_in_crop": the vector mean of the migration angles of all cells in the crop
    - "vec_mean_mag_in_crop": the vector mean of the migration magnitudes of all cells in the crop
    """

    df["all_velocity_angles_in_crop"] = (
        df.groupby(["dataset_name", "position", "T"])
        .apply(lambda df_sub: pd.DataFrame(map_label_to_column(df_sub, "centroid_velocity_angle")))
        .droplevel([0, 1, 2])
    )
    df["all_velocity_magnitudes_in_crop"] = (
        df.groupby(["dataset_name", "position", "T"])
        .apply(
            lambda df_sub: pd.DataFrame(map_label_to_column(df_sub, "centroid_velocity_magnitude"))
        )
        .droplevel([0, 1, 2])
    )
    df["all_centroid_dx_dt_in_crop"] = (
        df.groupby(["dataset_name", "position", "T"])
        .apply(lambda df_sub: pd.DataFrame(map_label_to_column(df_sub, "centroid_dx_dt")))
        .droplevel([0, 1, 2])
    )
    df["all_centroid_dy_dt_in_crop"] = (
        df.groupby(["dataset_name", "position", "T"])
        .apply(lambda df_sub: pd.DataFrame(map_label_to_column(df_sub, "centroid_dy_dt")))
        .droplevel([0, 1, 2])
    )

    # calculate the vector means of all cells within the crop
    df["all_velocity_angles_in_crop"] = df["all_velocity_angles_in_crop"].transform(
        sanitize_list_to_numbers
    )
    df["all_centroid_dx_dt_in_crop"] = df["all_centroid_dx_dt_in_crop"].transform(
        sanitize_list_to_numbers
    )
    df["all_centroid_dy_dt_in_crop"] = df["all_centroid_dy_dt_in_crop"].transform(
        sanitize_list_to_numbers
    )

    df[["vec_mean_angle_in_crop", "vec_mean_mag_in_crop"]] = pd.DataFrame(
        df["all_velocity_angles_in_crop"]
        .transform(
            lambda angles: (
                vector_mean_angle_and_mag(angles) if len(angles) > 1 else (np.nan, np.nan)
            )
        )
        .tolist(),
        index=df.index,
    )

    return df
