import concurrent
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import dask.array as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bioio import BioImage
from scipy.ndimage import gaussian_filter1d
from skimage.measure import regionprops
from tqdm import tqdm

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.io.input import load_image
from endo_pipeline.library.model.eval_model import add_diffae_model_eval_crop_columns
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest
from endo_pipeline.settings import DIMENSION_ORDER

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
    # the following columns are redundant with another in the table and
    # can be dropped:
    duplicate_cols = [
        "cell_label",
        "cdh5_segmentation_label",
        "cell_centroid",
        "cell_area (px**2)",
        "cell_perimeter (px)",
        "touches_border",
    ]
    big_table = big_table.drop(columns=duplicate_cols)

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
    for (dataset_nm, position), df in big_table_filtered.groupby(["dataset_name", "position"]):
        summary = df.groupby("T")[
            [
                "T",
                "num_unique_tracks_before_filtering_at_T",
                "num_unique_tracks_after_filtering_at_T",
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
            x="T",
            y="num_unique_tracks_before_filtering_at_T",
            data=summary,
            ax=ax,
            label="Before filtering",
        )
        sns.lineplot(
            x="T",
            y="num_unique_tracks_after_filtering_at_T",
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
        boxes = zip(top_of_boxes, left_of_boxes, right_of_boxes, strict=False)
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
        big_table.groupby(["dataset_name", "position"])["track_id"].nunique().sum()
    )

    # keep only tracks with duration longer than min_track_duration
    big_table["min_track_duration"] = min_track_duration
    big_table["is_greater_than_min_track_duration"] = (
        big_table["track_duration"] > min_track_duration
    )

    # keep only tracks where area_change is not too large
    big_table["max_smoothed_area_normd_change"] = max_area_change
    big_table["is_less_than_max_smoothed_area_normd_change"] = (
        big_table["smoothed_area_normd_diff"].abs() < max_area_change
    )

    # drop segmentation touches_image_border
    big_table.rename(
        columns={"touches_image_border": "is_edge_segmentation"},
        inplace=True,
    )

    # is_included is just all the previous filters combined
    big_table["is_included"] = (
        big_table[f"is_greater_than_min_track_duration"]
        & big_table[f"is_less_than_max_smoothed_area_normd_change"]
        & ~big_table["is_edge_segmentation"]
    )

    # drop because there are insufficient valid timepoints
    big_table["num_valid_tp_per_track"] = big_table.groupby(
        ["dataset_name", "position", "track_id"]
    )["is_included"].transform(sum)
    big_table["min_num_valid_tp_per_track"] = min_num_valid_points_per_track
    big_table["has_more_than_min_num_valid_points_per_track"] = (
        big_table["num_valid_tp_per_track"] > min_num_valid_points_per_track
    )

    # update is_included column with valid_tp_per_track
    big_table["is_included"] = (
        big_table["is_included"] & big_table["has_more_than_min_num_valid_points_per_track"]
    )

    # get the number of unique tracks after filtering in total and per timepoint
    num_rows_after_filtering = np.count_nonzero(big_table["is_included"])
    num_unique_tracks_after_filtering = (
        big_table[big_table["is_included"]]
        .groupby(["dataset_name", "position"])["track_id"]
        .nunique()
        .sum()
    )
    big_table["num_unique_tracks_after_filtering_at_T"] = (
        big_table[big_table["is_included"]]
        .groupby(["dataset_name", "position", "T"])["track_id"]
        .transform(lambda x: x.nunique())
    )

    # save a log file of the filtering that was done if saving the results
    if out_dir:
        # save a log file and create some plots showing number of
        # tracks before and after filtering
        datasets_analyzed = big_table["dataset_name"].unique().tolist()
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
            big_table[big_table["is_included"]],
            min_track_duration,
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
    big_table.rename(columns={"filepath_raw_image": "zarr_path"}, inplace=True)

    um_per_px_map = {}
    time_res_map = {}
    shear_stress_regime_map = {}

    for dataset_name in big_table["dataset_name"].unique():
        data_config = load_dataset_config(dataset_name)
        um_per_px_map[dataset_name] = data_config.pixel_size_xy_in_um
        time_res_map[dataset_name] = data_config.time_interval_in_minutes
        shear_stress_regime_map[dataset_name] = data_config.shear_stress_regime

    # add the shear stress regime to the data table
    logger.info("Adding shear stress regime...")
    big_table["shear_stress_regime"] = big_table["dataset_name"].transform(
        lambda dataset_name: shear_stress_regime_map[dataset_name]
    )

    # dimensionalize the time column
    logger.info("Adding time intervals per timepoint...")
    big_table["time_resolution_minutes"] = big_table["dataset_name"].transform(
        lambda dataset_name: time_res_map[dataset_name]
    )
    logger.info("Calculating time in minutes and hours...")
    big_table["time_minutes"] = big_table["image_index"] * big_table["time_resolution_minutes"]
    big_table["time_hours"] = big_table["time_minutes"] / 60
    # (NOTE the image index column is produced in the
    # tracking workflow, and is used instead of the
    # "T" column because that one may not represent
    # the acquisition timepoint for datasets that were
    # collected as a montage, and therefore have their
    # many positions represented in the T dimension;
    # e.g. position 0 may have their first, second,
    # third, etc. timepoints represented as
    # T = 0, 6, 12, etc...; the zarr-converted data
    # will not have this problem, and therefore using
    # the image index will be consistent across both
    # versions of the data)

    # add a column for the number of unique tracks
    # per dataset per position per timepoint
    # (this should be 1 everywhere)
    big_table["num_unique_tracks_per_timeframe"] = big_table.groupby(
        ["dataset_name", "position", "image_index", "track_id"]
    ).transform("size")

    # add the columns for the fold change in area
    logger.info("Calculating locally-normalized area...")
    sigma = 2.0
    big_table["gaussian_sigma_for_area_smoothing"] = sigma
    big_table["smoothed_area_normd"] = big_table.groupby(["dataset_name", "position", "track_id"])[
        "area"
    ].transform(lambda x: calculate_smoothed_normd_area(x, smoothing_sigma=sigma))
    big_table["smoothed_area_normd_diff"] = big_table.groupby(
        ["dataset_name", "position", "track_id"]
    )["smoothed_area_normd"].transform(lambda x: x.diff())

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    logger.info("Adding number of tracks for each timepoint...")
    big_table["num_unique_tracks_before_filtering_at_T"] = big_table.groupby(
        ["dataset_name", "position", "T"]
    )["track_id"].transform(lambda x: x.nunique())

    # add the duration of each track
    logger.info("Calculating track durations...")
    big_table["track_duration"] = big_table.groupby(["dataset_name", "position", "track_id"])[
        "image_index"
    ].transform(lambda t: t.max() - t.min())

    # add column for orientation in degrees of the
    # ellipse fitted to each segmentation in degrees
    logger.info("Converting orientation to degrees...")
    big_table["alignment_rel_to_flow"] = big_table["orientation"].transform(
        lambda x: make_orientation_relative_to_flow(x)
    )
    big_table["alignment_deg_rel_to_flow"] = np.rad2deg(big_table["alignment_rel_to_flow"])

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    logger.info("Calculating nematic order and aspect ratio...")
    big_table["nematic_order"] = big_table["orientation"].transform(
        lambda x: get_nematic_order(x - np.pi / 2)
    )
    big_table["aspect_ratio"] = big_table["eccentricity"].transform(get_aspect_ratio)

    # add pixel sizes
    big_table["pixel_size_xy_in_um"] = big_table["dataset_name"].transform(
        lambda dataset_name: um_per_px_map[dataset_name]
    )
    big_table["area (um**2)"] = big_table["area"] * big_table["pixel_size_xy_in_um"] ** 2
    big_table["perimeter (um)"] = big_table["perimeter"] * big_table["pixel_size_xy_in_um"]

    # add a column for the number of neighbors
    # touching each region that is being tracked
    logger.info("Calculating number of neighbors...")
    big_table["number_of_neighbors"] = big_table["neighboring_cell_labels"].transform(
        lambda x: len(x)
    )

    # add the image size to the data table
    new_cols = {}
    for (ds_nm, pos), grp in big_table.groupby(["dataset_name", "position"]):
        data_config = load_dataset_config(ds_nm)

        zarr_path = get_zarr_file_for_position(data_config, pos)
        assert (grp["zarr_path"].transform(Path) == zarr_path).all(), "Zarr path mismatch in group."

        logger.info(f"getting image size for {ds_nm} position {pos}...")
        img = BioImage(zarr_path)
        img.set_resolution_level(0)
        image_size_y, image_size_x = img.dims.Y, img.dims.X

        new_cols[(ds_nm, pos)] = {
            "image_size_x": image_size_x,
            "image_size_y": image_size_y,
            "EGFP_channel_index_zarr": data_config.zarr_channel_indices.channel_488,
            "brightfield_channel_index_zarr": data_config.zarr_channel_indices.brightfield,
        }
    big_table = big_table.merge(
        big_table.groupby(["dataset_name", "position"])
        .apply(
            lambda df: pd.DataFrame(
                columns=new_cols[tuple(df.name)].keys(),
                data=new_cols[tuple(df.name)],
                index=df.index,
            ),  # type: ignore[call-overload]
            include_groups=False,
        )
        .droplevel([0, 1]),
        left_index=True,
        right_index=True,
    )

    # add the number of nuclei that overlap the most with each cell
    # (this can be used as a filter later so we only measure cells
    # with a single clearly distinguishable nuclei)
    big_table["num_nuclei_with_most_overlap"] = big_table["nuclei_seg_in_cdh5_seg_frac"].transform(
        len
    )

    # add the nuclei centroids relative to the cell centroids
    big_table["nuc_pos_rel_cell_X"], big_table["nuc_pos_rel_cell_Y"] = (
        get_nuclei_rel_to_cell_position(
            big_table["centroid_X"],
            big_table["centroid_Y"],
            big_table["nuc_with_most_overlap_0_centroid_X"],
            big_table["nuc_with_most_overlap_0_centroid_Y"],
        )
    )

    # get the angles and magnitudes of the nuclei relative positions
    big_table["nuc_pos_rel_cell_magnitude"] = np.linalg.norm(
        [big_table["nuc_pos_rel_cell_X"], big_table["nuc_pos_rel_cell_Y"]], axis=0
    )
    big_table["nuc_pos_rel_cell_angle"] = np.arctan2(
        big_table["nuc_pos_rel_cell_Y"], big_table["nuc_pos_rel_cell_X"]
    )
    big_table["nuc_pos_rel_cell_angle_deg"] = np.rad2deg(big_table["nuc_pos_rel_cell_angle"])

    # add the DiffAE crop locations and binning level; these can be used to load
    # a crop from the zarr files and compute the number of nuclei in that crop
    big_table = add_diffae_model_eval_crop_columns(big_table)

    # compute the number of nuclei found in a defined crop size
    # (first take a subset using only the required columns to reduce memory usage)
    required_columns = [
        "dataset_name",
        "position",
        "image_index",
        "track_id",
        "label",
        "centroid_Y",
        "centroid_X",
        "image_size_y",
        "image_size_x",
        "crop_size",
        "start_y",
        "end_y",
        "start_x",
        "end_x",
        "diffae_resolution_level_to_use",
    ]
    num_nuclei_in_crop_df = add_num_nuclei_in_crop_column(
        big_table[required_columns], use_precomputed=False
    )
    crops = ["dataset_name", "position", "image_index", "track_id"]
    added_cols = list(set(num_nuclei_in_crop_df.columns) - set(big_table.columns))
    big_table = pd.merge(
        left=big_table,
        right=num_nuclei_in_crop_df[crops + added_cols],
        on=crops,
        how="left",
        validate="one_to_one",
    )

    return big_table


def calculate_derived_data_dynamics_dependent(big_table: pd.DataFrame) -> pd.DataFrame:
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
                    strict=False,
                ),  # type: ignore[return-value]
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

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

    big_table["nuc_pos_vs_cell_veloc_dotprod"] = np.einsum(
        "ij,ij->i",
        big_table[["centroid_dx_dt", "centroid_dy_dt"]],
        big_table[["nuc_pos_rel_cell_X", "nuc_pos_rel_cell_Y"]],
    )

    # add column for the number of tracks at a given
    # timepoint per dataset per position
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


# restrict orientation to be between 0 and pi/2 instead of between -pi/2 and pi/2 so that
# we can interpret this orientation as being either parallel or perpendicular to flow
def shift_orientation_phase(orientation: float) -> float:
    return orientation - np.pi / 2


def restrict_orientation_to_positive(orientation: float) -> float:
    return abs(orientation)


def make_orientation_relative_to_flow(orientation: float) -> float:
    """
    Changes 0 degrees from being the positive Y-axis (up)
    to being the positive X-axis (to the right).
    Also adjusts the range of possible angles from being
    between -pi/2 and pi/2 to being between 0 and pi/2.
    """
    # you can visualize this process as folding a paper circle in half
    # (the top half) and then rotating this half circle 90 degrees to
    # the right, and then folding it in half again so you are only
    # left with the top right quadrant of the circle.
    return restrict_orientation_to_positive(
        shift_orientation_phase(restrict_orientation_to_positive(orientation))
    )


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
) -> np.ndarray | pd.Series:
    """
    Returns the smallest difference between angles and reference_angles.
    The result is signed, so if the returned angle is positive then
    the angle is counter-clockwise from the reference angle, and if
    the returned angle is negative then the angle is clockwise from
    the reference angle.

    Parameters
    ----------
    angles : np.ndarray | pd.Series
        The angles to compare.
    reference_angles : np.ndarray | pd.Series
        The reference angles to compare against.
    units : Literal["deg", "rad"]
        The units of the angles. Either "deg" for degrees or "rad" for radians.

    Returns
    -------
    np.ndarray | pd.Series
        The smallest difference between the angles and the reference angles.

    Note: This solution was not my idea and was taken from StackOverflow:
    https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
    """
    if units == "rad":
        circle = np.pi
    elif units == "deg":
        circle = 360
    else:
        raise ValueError("units must be either 'deg' or 'rad'")
    half_circle = circle / 2
    angle_diff = angles - reference_angles
    angle_diff = (angle_diff + half_circle) % circle - half_circle
    return angle_diff


def get_segmentation_path_dict(dataset_name: str, position: int) -> dict:
    dataset = load_dataset_config(dataset_name)
    manifest = load_image_manifest("cdh5_classic_seg")
    return {
        timepoint: get_image_location_for_dataset(manifest, dataset_name, position, timepoint)
        for timepoint in range(dataset.duration)
    }


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
    seg_manifest = load_image_manifest("nuclear_labelfree_seg")
    seg_location = get_image_location_for_dataset(
        manifest=seg_manifest,
        dataset_name=dataset_name,
        position=position,
        timepoint=timeframe,
    )
    nuc_seg = load_image(seg_location, squeeze=False)

    # get nuclei segmentation properties and dimension order of those properties
    props = regionprops(nuc_seg.squeeze())
    dim_shapes = dict(zip(dim_order, nuc_seg.shape))
    dim_order_squeezed = "".join([d for d in dim_order if dim_shapes[d] > 1])

    centroids: dict[str, Any] = get_nuclei_coords(
        props=props,
        props_dim_order=dim_order_squeezed,
        kind="centroid",
    )
    centroids["dataset_name"] = dataset_name
    centroids["position"] = position
    centroids["image_index"] = timeframe

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
    dataset_name = sequence_to_scalar(merged_feats_df["dataset_name"])
    nuclei_centroids_path = nuclei_centroids_dir / f"{dataset_name}_nuclei_centroids.parquet"

    # if the nuclei coordinates are already computed, load them
    if use_precomputed and nuclei_centroids_path.exists():
        nuc_centroid_indices = pd.read_parquet(nuclei_centroids_path)
    # otherwise, compute and save them
    # (this will take about 60 minutes divided by n_cores used)
    else:
        # compute the nuclei prediction centroids
        groups = merged_feats_df.groupby(["dataset_name", "position", "image_index"])
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
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
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
        on=["dataset_name", "position", "image_index"],
        how="left",
    )
    groups = merged_feats_df.groupby(["dataset_name", "position", "image_index"])

    num_nuclei_in_crop = []
    for nm, df in tqdm(groups, desc=f"Counting nuclei in crops: {dataset_name}"):
        # get the number of nuclei in the crops at each timepoint
        num_nuc_centroids = get_num_unique_values_in_bounds_from_df(
            nuclei_coords_Y=np.stack(list(df["coords_Y"])),
            nuclei_coords_X=np.stack(list(df["coords_X"])),
            crop_bounds_Y=(df["start_y"], df["end_y"]),
            crop_bounds_X=(df["start_x"], df["end_x"]),
        )
        num_nuclei_in_crop.append(pd.Series(num_nuc_centroids, index=df.index))

    merged_feats_df["num_nuclei_in_crop"] = pd.concat(
        num_nuclei_in_crop, axis=0, ignore_index=False
    )
    # drop the nuclei coordinates lists since they are not needed anymore
    merged_feats_df = merged_feats_df.drop(columns=["coords_Y", "coords_X"])
    return merged_feats_df
