from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bioio import BioImage
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    extract_T,
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_measured_segmentation_data_raws,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
)
from cellsmap.util.set_output import get_output_path


def merge_measured_segmentation_features_tables(
    segprops_df: pd.DataFrame,
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
        right=segprops_df,
        left_on=["dataset_name", "position", "T", "label"],
        right_on=["dataset_name", "position", "T", "cell_label"],
    )
    big_table = pd.merge(
        left=big_table,
        right=nucprops_df,
        left_on=["dataset_name", "position", "T", "label"],
        right_on=["dataset_name", "position", "T", "cdh5_segmentation_label"],
    )
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
        timelapse_duration = get_dataset_info(dataset_nm)["duration"]
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
    `big_table[big_table["filter_edge_FOV"] == False]`
    (or equivalently: `big_table[~big_table["filter_edge_FOV"]]`)
    """

    # get the number of segmentations in total and per timepoint
    num_rows_before_filtering = len(big_table)
    num_unique_tracks_before_filtering = (
        big_table.groupby(["dataset_name", "position"])["track_id"].nunique().sum()
    )

    # drop because min_track_duration not exceeded
    big_table["min_track_duration"] = min_track_duration
    big_table[f"filter_min_track_duration_{min_track_duration}"] = (
        big_table["track_duration"] <= min_track_duration
    )

    # drop because area_change is too large
    big_table["max_smoothed_area_normd_change"] = max_area_change
    big_table[f"filter_max_smoothed_area_normd_change_{max_area_change}"] = (
        big_table["smoothed_area_normd_diff"].abs() >= max_area_change
    )

    # drop because segmentation touches_image_border
    big_table.rename(
        columns={"touches_image_border": "filter_edge_FOV"},
        inplace=True,
    )

    # filter_global is just all the previous filters combined
    big_table["filter_global"] = (
        big_table[f"filter_min_track_duration_{min_track_duration}"]
        + big_table[f"filter_max_smoothed_area_normd_change_{max_area_change}"]
        + big_table["filter_edge_FOV"]
    )

    # drop because there are insufficient valid timepoints
    big_table["min_num_valid_points_per_track"] = min_num_valid_points_per_track
    big_table["valid_points"] = big_table.groupby(["dataset_name", "position", "track_id"])[
        "image_index"
    ].transform("nunique")
    big_table[f"filter_valid_points_{min_num_valid_points_per_track}"] = (
        big_table["valid_points"] < min_num_valid_points_per_track
    )

    # update filter_global
    big_table["filter_global"] = (
        big_table["filter_global"]
        + big_table[f"filter_valid_points_{min_num_valid_points_per_track}"]
    )

    # get the number of unique tracks after filtering in total and per timepoint
    num_rows_after_filtering = np.count_nonzero(~big_table["filter_global"])
    num_unique_tracks_after_filtering = (
        big_table[~big_table["filter_global"]]
        .groupby(["dataset_name", "position"])["track_id"]
        .nunique()
        .sum()
    )
    big_table["num_unique_tracks_after_filtering_at_T"] = (
        big_table[~big_table["filter_global"]]
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
            big_table[~big_table["filter_global"]],
            min_track_duration,
        )
    return big_table


def calculate_derived_data_dynamics_independent(
    big_table: pd.DataFrame, verbose: bool = False
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
    um_per_px_map = {
        dataset_name: get_dataset_info(dataset_name)["pixel_size_xy_in_um"]
        for dataset_name in big_table["dataset_name"].unique()
    }
    time_res_map = {
        dataset_name: get_dataset_info(dataset_name)["time_interval_in_minutes"]
        for dataset_name in big_table["dataset_name"].unique()
    }
    shear_stress_regime_map = {
        dataset_name: get_dataset_info(dataset_name)["shear_stress_regime"]
        for dataset_name in big_table["dataset_name"].unique()
    }

    # add the shear stress regime to the data table
    print("Adding shear stress regime...") if verbose else None
    big_table["shear_stress_regime"] = big_table["dataset_name"].transform(
        lambda dataset_name: shear_stress_regime_map[dataset_name]
    )

    # dimensionalize the time column
    print("Adding time intervals per timepoint...") if verbose else None
    big_table["time_resolution_minutes"] = big_table["dataset_name"].transform(
        lambda dataset_name: time_res_map[dataset_name]
    )
    print("Calculating time in minutes and hours...") if verbose else None
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
    print("Calculating locally-normalized area...") if verbose else None
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
    print("Adding number of tracks for each timepoint...") if verbose else None
    big_table["num_unique_tracks_before_filtering_at_T"] = big_table.groupby(
        ["dataset_name", "position", "T"]
    )["track_id"].transform(lambda x: x.nunique())

    # add the duration of each track
    print("Calculating track durations...") if verbose else None
    big_table["track_duration"] = big_table.groupby(["dataset_name", "position", "track_id"])[
        "image_index"
    ].transform(lambda t: t.max() - t.min())

    # add column for orientation in degrees of the
    # ellipse fitted to each segmentation in degrees
    print("Converting orientation to degrees...") if verbose else None
    big_table["alignment_rel_to_flow"] = big_table["orientation"].transform(
        lambda x: make_orientation_relative_to_flow(x)
    )
    big_table["alignment_deg_rel_to_flow"] = np.rad2deg(big_table["alignment_rel_to_flow"])

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    print("Calculating nematic order and aspect ratio...") if verbose else None
    big_table["nematic_order"] = big_table["orientation"].transform(
        lambda x: get_nematic_order(x - np.pi / 2)
    )
    big_table["aspect_ratio"] = big_table["eccentricity"].transform(get_aspect_ratio)

    # dimensionalize the area
    print("Dimensionalizing area and perimeter...") if verbose else None
    big_table["pixel_size_xy_in_um"] = big_table["dataset_name"].transform(
        lambda dataset_name: um_per_px_map[dataset_name]
    )
    big_table["area (um**2)"] = big_table["area"] * big_table["pixel_size_xy_in_um"] ** 2
    big_table["perimeter (um)"] = big_table["perimeter"] * big_table["pixel_size_xy_in_um"]

    # add a column for the number of neighbors
    # touching each region that is being tracked
    print("Calculating number of neighbors...") if verbose else None
    big_table["neighboring_cell_labels"] = big_table["neighboring_cell_labels"].transform(
        lambda x: stringified_floatlist_to_floatlist(x)
    )
    big_table["number_of_neighbors"] = big_table["neighboring_cell_labels"].transform(
        lambda x: len(x)
    )

    # add the image size to the data table
    new_cols = {}
    for (ds_nm, pos), grp in big_table.groupby(["dataset_name", "position"]):

        zarr_name = get_zarr_name(ds_nm, pos)
        zarr_path = Path(get_zarr_path(ds_nm, zarr_name)[zarr_name])

        print(f"getting image size for {ds_nm} position {pos}...") if verbose else None
        # NOTE the zarr paths are not working for 20241203_9db6173b3da7452b91756b6e86b0da61_P3
        try:
            img = BioImage(zarr_path)
            img.set_resolution_level(0)
            channel_index = dict(
                zip(img.channel_names, range(len(img.channel_names)), strict=False)
            )
        except:
            print("loading zarr failed, falling back to original path...")
            og_path = get_original_path(ds_nm)
            img = BioImage(og_path)
            channel_index = dict(zip(["EGFP", "BF"], range(len(img.channel_names)), strict=False))

        image_size_y, image_size_x = img.dims.Y, img.dims.X

        new_cols[(ds_nm, pos)] = {
            "zarr_path": zarr_path.as_posix(),
            "image_size_x": image_size_x,
            "image_size_y": image_size_y,
            "EGFP_channel_index_zarr": channel_index["EGFP"],
            "brightfield_channel_index_zarr": channel_index["BF"],
        }
    big_table = big_table.merge(
        big_table.groupby(["dataset_name", "position"])
        .apply(
            lambda df: pd.DataFrame(
                columns=new_cols[tuple(df.name)].keys(),
                data=new_cols[tuple(df.name)],
                index=df.index,
            ),  # type: ignore
            include_groups=False,
        )
        .droplevel([0, 1]),
        left_index=True,
        right_index=True,
    )

    # add the number of nuclei that overlap the most with each cell
    # (this can be used as a filter later so we only measure cells
    # with a single clearly distinguishable nuclei)
    big_table["nuclei_seg_in_cdh5_seg_frac"] = big_table["nuclei_seg_in_cdh5_seg_frac"].transform(
        lambda x: stringified_floatlist_to_floatlist(x, to_tuple=True)
    )
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
    big_table["nuc_pos_rel_cell_angle_rel_to_flow"] = big_table["nuc_pos_rel_cell_angle"].transform(
        lambda x: make_orientation_relative_to_flow(x)
    )
    big_table["nuc_pos_rel_cell_angle_deg_rel_to_flow"] = np.rad2deg(
        big_table["nuc_pos_rel_cell_angle_rel_to_flow"]
    )

    return big_table


def calculate_derived_data_dynamics_dependent(
    big_table: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """
    NOTE: The accuracy of these metrics are affected by how
    clean the data in the table is, therefore it should only
    be used after filtering out incorrect segmentations from
    the data table.
    """
    # recalculate the centroid speeds of each track
    # after filtering
    print("Calculating centroid velocities...") if verbose else None
    big_table[["centroid_y", "centroid_x"]] = (
        big_table["centroid"].transform(lambda c: stringified_floatlist_to_floatlist(c)).tolist()
    )
    big_table["centroid_x_um"] = big_table["centroid_x"] * big_table["pixel_size_xy_in_um"]
    big_table["centroid_y_um"] = big_table["centroid_y"] * big_table["pixel_size_xy_in_um"]
    big_table[["centroid_dx_dt", "centroid_dy_dt"]] = (
        big_table.groupby(["dataset_name", "position", "track_id"], as_index=True)[
            ["centroid_x_um", "centroid_y_um", "time_minutes"]
        ]
        .apply(
            lambda df: pd.DataFrame(  # type: ignore
                columns=["centroid_dx_dt", "centroid_dy_dt"],
                data=zip(
                    *get_centroid_velocity(
                        df["centroid_x_um"].values,  # type: ignore
                        df["centroid_y_um"].values,  # type: ignore
                        df["time_minutes"].values,  # type: ignore
                    ),
                    strict=False,
                ),  # type: ignore
                index=df.index,
            )
        )
        .droplevel([0, 1, 2])
    )

    print("Calculating centroid velocity magnitude and angle...") if verbose else None
    big_table["centroid_velocity_magnitude"] = np.linalg.norm(
        [big_table["centroid_dx_dt"], big_table["centroid_dy_dt"]], axis=0
    )
    big_table["centroid_velocity_angle"] = np.arctan2(
        big_table["centroid_dy_dt"], big_table["centroid_dx_dt"]
    )
    big_table["centroid_velocity_angle_deg"] = np.rad2deg(big_table["centroid_velocity_angle"])
    big_table["centroid_velocity_angle_rel_to_flow"] = big_table[
        "centroid_velocity_angle"
    ].transform(lambda x: make_orientation_relative_to_flow(x))
    big_table["centroid_velocity_angle_deg_rel_to_flow"] = np.rad2deg(
        big_table["centroid_velocity_angle_rel_to_flow"]
    )

    big_table["dalignment_dt_deg_rel_to_flow"] = (
        big_table["alignment_deg_rel_to_flow"].diff() / big_table["time_minutes"].diff()
    )

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    print("Adding number of tracks for each timepoint...") if verbose else None
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


def stringified_floatlist_to_floatlist(ls: str, to_tuple: bool = False) -> list | tuple:
    """Converts a list that is saved as a string back to a list object.
    Assumes that there is only one set of brackets (either '[]' or '()').
    """
    # if 'ls' is already a list of floats then return the input
    if isinstance(ls, list) and all([isinstance(x, float) for x in ls]):
        return tuple(ls) if to_tuple else ls
    # otherwise procede with the conversion
    else:
        strfloats = ls.strip("[]")
        strfloats = strfloats.strip("()")
        float_list: list[Any] = []
        for x in strfloats.split(","):
            try:
                float_list.append(float(x))
            # handle allowed special cases or raise an error
            except ValueError:
                if "masked" in x:
                    float_list.append(np.ma.masked)
                elif "nan" in x:
                    float_list.append(np.nan)
                elif x == "":
                    pass
                else:
                    raise ValueError(f'Could not convert "{x}" to float.')
        return tuple(float_list) if to_tuple else float_list


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


def filter_and_save_track_data_for_landscape_integration(
    big_table: pd.DataFrame,
    out_filename: str | Path | None = None,
    crop_size: int = 256,
    min_num_points_per_track: int = 0,
    return_df: bool = False,
) -> pd.DataFrame | None:

    big_table = big_table[
        big_table.groupby(["dataset_name", "position", "track_id"])["track_id"].transform(
            lambda x: x.count() > min_num_points_per_track
        )
    ]

    integration_table = big_table[
        [
            "zarr_path",
            "image_index",
            "track_id",
            "label",
            "centroid_x",
            "centroid_y",
            "image_size_x",
            "image_size_y",
        ]
    ].copy()
    integration_table["crop_size"] = crop_size

    # remove all the centroids that are closer than 128 pixels
    # to the image border
    integration_table = integration_table[integration_table["centroid_x"] > crop_size // 2]
    integration_table = integration_table[integration_table["centroid_y"] > crop_size // 2]
    integration_table = integration_table[
        integration_table["centroid_x"] < integration_table["image_size_x"] - crop_size // 2
    ]
    integration_table = integration_table[
        integration_table["centroid_y"] < integration_table["image_size_y"] - crop_size // 2
    ]

    if out_filename:
        # save the filtered data to a file
        integration_table.to_csv(out_filename, index=False)

    return integration_table if return_df else None


# restrict orientation to be between 0 and pi/2 instead of between -pi/2 and pi/2 so that
# we can interpret this orientation as being either parallel or perpendicular to flow
def shift_orientation_phase(orientation: float) -> float:
    return orientation - np.pi / 2


def restrict_orientation_to_positive(orientation: float) -> float:
    return abs(orientation)


def make_orientation_relative_to_flow(orientation: float) -> float:
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


def add_cell_segmentation_path_column(
    big_table: pd.DataFrame,
) -> pd.DataFrame:
    seg_path_per_pos_dict = dict()
    for ds_nm, pos in big_table.groupby(["dataset_name", "position"]).groups.keys():  # type: ignore
        seg_path_per_pos_dict[pos] = get_segmentation_path_dict(ds_nm, pos)  # type: ignore
    big_table["cdh5_classic_segmentation_path"] = big_table.apply(
        lambda df: (seg_path_per_pos_dict[df["position"]][df["T"]].as_posix()), axis=1
    )
    return big_table


def get_segmentation_path_dict(dataset_name: str, position: int) -> dict:
    cdh5_seg_dir = get_cdh5_classic_segmentation_path(dataset_name, position)
    if cdh5_seg_dir is None:
        raise ValueError(
            f"No segmentation directory found for dataset {dataset_name} position {position}."
        )
    seg_path_dict = {
        extract_T(fp.stem): fp for fp in sorted(cdh5_seg_dir.glob("**/*.ome.tif*"), key=extract_T)
    }
    return seg_path_dict


def process_and_plot_tracking_data_multiproc_wrapper(args: Sequence) -> None:
    dataset_name, out_dir, verbose = args
    process_and_plot_tracking_data(dataset_name, out_dir, verbose=verbose)


def process_and_plot_tracking_data(
    dataset_name: str,
    out_dir: str | Path,
    verbose: bool = False,
    plot_figures: bool = False,
) -> None:

    # make the output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the tracking data and the segmentation feature data
    tracking_df = get_measured_segmentation_data_raws(
        dataset_name_list=[dataset_name],
        kind="cdh5_tracking",
        as_dask=False,
    )
    segprops_df = get_measured_segmentation_data_raws(
        dataset_name_list=[dataset_name],
        kind="cdh5_segmentations",
        as_dask=False,
    )
    nucprops_df = get_measured_segmentation_data_raws(
        dataset_name_list=[dataset_name],
        kind="nuclei_labelfree",
        as_dask=False,
    )
    if tracking_df.empty or segprops_df.empty or nucprops_df.empty:
        print(
            f"No tracking data or segmentation properties data found for {dataset_name}. Skipping..."
        )
        return
    else:
        print(f"Working on {dataset_name}...") if verbose else None

    # combine the tracking data with the segmentation
    # properties data
    (print("Combining tracking data with segmentation properties data...") if verbose else None)
    big_table = merge_measured_segmentation_features_tables(segprops_df, tracking_df, nucprops_df)

    # add some columns to the data table that are
    # calculated from existing columns and do not
    # depend on dynamics / require clean tracks
    (
        print("Calculating dynamics-independent metrics from existing measurements...")
        if verbose
        else None
    )
    big_table = add_cell_segmentation_path_column(big_table)
    big_table = calculate_derived_data_dynamics_independent(big_table, verbose)

    # add the size of the crop used to get DiffAE features at full res
    crop_size = 256
    big_table["crop_size"] = crop_size

    # filter the segprops data to remove regions that
    # touch the image borders and keep only tracks that
    # have a minimum number of datapoints after this
    (
        print("Filtering out regions that touch the image borders and tracks that are too short...")
        if verbose
        else None
    )
    big_table = add_filter_columns(big_table, out_dir, min_track_duration=24, max_area_change=0.1)
    big_table_filtered = big_table[~big_table["filter_global"]]

    # NOTE THIS TABLE WILL BE UPLOADED TO FMS
    # save the raw combined data tables
    # (we want to have an accessible version of the raw data)
    out_dir_raw = out_dir / "segmentation_features_manifests/"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_path_raw = out_dir_raw / f"{dataset_name}_segmentation_features.tsv"
    big_table.to_csv(out_path_raw, sep="\t", index=False)

    # add some columns that are calculated from the
    # existing columns include:
    # orientation in degrees, velocities, nematic order,
    # aspect ratio, number of tracks (i.e. approximate
    # number of detected cells)
    (
        print("Calculating dynamics-dependent metrics from existing measurements...")
        if verbose
        else None
    )
    big_table_filtered = calculate_derived_data_dynamics_dependent(
        big_table_filtered.copy(deep=True), verbose
    )

    # create a subset of the data that is used for cell track integration
    (
        print("Outputting a subset of the cell tracking data for integration with landscapes...")
        if verbose
        else None
    )
    out_dir_for_integration = Path(out_dir) / "single_cell_track_integration/"
    out_dir_for_integration.mkdir(parents=True, exist_ok=True)
    out_path_integration_table = (
        out_dir_for_integration / f"{dataset_name}_single_cell_track_integration.csv"
    )
    filter_and_save_track_data_for_landscape_integration(
        big_table_filtered,
        out_path_integration_table,
        crop_size=crop_size,
        min_num_points_per_track=120,
        return_df=False,
    )


def main(
    dataset_name: str | None = None,
    n_proc: int = 1,
    verbose: bool = False,
    make_plots: bool = False,
) -> None:

    out_dir = get_output_path(Path(__file__).stem, verbose=False)

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    if n_proc > 1:
        n_proc = min(n_proc, len(dataset_name_list))
        with Pool(processes=n_proc) as pool:
            args = zip(
                dataset_name_list,
                [out_dir] * len(dataset_name_list),
                [verbose] * len(dataset_name_list),
                strict=False,
            )
            list(
                tqdm(
                    pool.imap(process_and_plot_tracking_data_multiproc_wrapper, args),
                    total=len(dataset_name_list),
                    desc="Processing datasets (MP)",
                    unit="datasets",
                )
            )
            pool.close()
            pool.join()
    else:
        for dataset_name in tqdm(
            dataset_name_list,
            total=len(dataset_name_list),
            desc="Processing datasets (1P)",
            unit="datasets",
        ):
            process_and_plot_tracking_data(
                dataset_name, out_dir, verbose=verbose, plot_figures=make_plots
            )


if __name__ == "__main__":
    ipython_cli_flexecute(main)
