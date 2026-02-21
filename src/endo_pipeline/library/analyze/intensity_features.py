from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_image_manifest,
)
from endo_pipeline.settings.image_data import DIMENSION_ORDER

EXAMPLES = {
    "high_flow": [
        {
            "dataset_name": "20250611_20X",
            "position": 0,
            "timepoint": 275,
            "track_id": 5175,
            "label": 135,
        }
    ],
    "low_flow": [
        # {
        #     "dataset_name": "20250402_20X",
        #     "position": 0,
        #     "timepoint": 166,
        #     "track_id": 775,
        #     "label": 141,
        # },
        # {
        #     "dataset_name": "20250402_20X",
        #     "position": 0,
        #     "timepoint": 166,
        #     "track_id": 4076,
        #     "label": 343,
        # },
        {
            "dataset_name": "20250402_20X",
            "position": 0,
            "timepoint": 166,
            "track_id": 3647,
            "label": 203,
        },
        {
            "dataset_name": "20250402_20X",
            "position": 0,
            "timepoint": 166,
            "track_id": 1781,
            "label": 213,
        },
        {
            "dataset_name": "20250402_20X",
            "position": 0,
            "timepoint": 166,
            "track_id": 2964,
            "label": 193,
        },
        {
            "dataset_name": "20250402_20X",
            "position": 0,
            "timepoint": 166,
            "track_id": 3064,
            "label": 47,
        },
    ],
}


def calculate_edge_intensity_distribution_for_segmentations(
    dataset_name, position, timepoint, df_at_tp, dim_order=DIMENSION_ORDER
):

    out_dir = Path(df_at_tp["output_dir"].unique().item())
    dataset_config = load_dataset_config(dataset_name)

    image_loc = get_zarr_location_for_position(dataset_config, position)
    raw_arr = load_image(image_loc, channels=["EGFP"], timepoints=timepoint, level=0)
    raw_arr = raw_arr.max(axis=dim_order.index("Z")).squeeze().compute()

    seg_manifest = load_image_manifest("cdh5_classic_seg")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position, timepoint)
    seg_arr = load_image(seg_location, squeeze=True, compute=True)

    # initialize dataframe
    df_edge_intens = pd.DataFrame(columns=df_at_tp.columns)
    df_edge_intens = df_edge_intens.drop("is_included", axis=1)

    for label, seg_record in df_at_tp.groupby("label"):
        seg_bound = find_boundaries(seg_arr == label)
        seg_bound_locs = np.where(seg_bound)
        seg_centroid = (
            seg_record["centroid_Y"].values.item(),
            seg_record["centroid_X"].values.item(),
        )

        # get the angle from each pixel in seg_bound to seg_centroid and also
        # the fluorescence intensity at each of those pixels
        angles = np.arctan2(
            seg_bound_locs[0] - seg_centroid[0],
            seg_bound_locs[1] - seg_centroid[1],
        )
        intensities = raw_arr[seg_bound_locs]

        seg_record["angle"] = [angles.tolist()]
        seg_record["intensity"] = [intensities.tolist()]

        if df_edge_intens.empty:
            df_edge_intens = seg_record.copy(deep=True)
        else:
            df_edge_intens = pd.concat([df_edge_intens, seg_record], ignore_index=True)
    out_subdir = out_dir / dataset_name / f"P{position}"
    out_subdir.mkdir(exist_ok=True, parents=True)
    df_edge_intens.to_parquet(
        out_subdir / f"{dataset_name}_P{position}_T{timepoint}_edge_intensities.parquet"
    )


def calculate_edge_intensity_distribution_for_segmentations_mp(args):
    (dataset_name, position, timepoint), df_at_tp = args
    dataset_name = str(dataset_name)
    position = int(position)
    timepoint = int(timepoint)
    return calculate_edge_intensity_distribution_for_segmentations(
        dataset_name, position, timepoint, df_at_tp
    )


def show_intensity_measure_example(
    df: pd.DataFrame,
    dataset_name: str,
    position: int,
    timepoint: int,
    seg_label: int,
    dim_order: str = DIMENSION_ORDER,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    dataset_config = load_dataset_config(dataset_name)

    image_loc = get_zarr_location_for_position(dataset_config, position)
    raw_arr = load_image(image_loc, channels=["EGFP"], timepoints=timepoint, level=0)
    raw_arr = raw_arr.max(axis=dim_order.index("Z")).squeeze().compute()

    seg_manifest = load_image_manifest("cdh5_classic_seg")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position, timepoint)
    seg_arr = load_image(seg_location, squeeze=True, compute=True)

    record = df[
        (df.dataset == dataset_name)
        & (df.position == position)
        & (df.image_index == timepoint)
        & (df.label == seg_label)
    ]
    y_slice, x_slice = [slice(d.min() - 10, d.max() + 11) for d in np.where(seg_arr == seg_label)]

    overlay = label2rgb(
        find_boundaries(seg_arr == seg_label),
        rescale_intensity(np.clip(raw_arr, a_min=20, a_max=150), out_range=(0, 1)),
        bg_label=0,
        alpha=0.3,
    )

    seg_bound = find_boundaries(seg_arr == seg_label)
    seg_bound_locs = np.where(seg_bound)

    seg_centroid = (
        record["centroid_Y"].values.item(),
        record["centroid_X"].values.item(),
    )

    # get the angle from each pixel in seg_bound to seg_centroid and also
    # the fluorescence intensity at each of those pixels
    angles = np.arctan2(
        seg_bound_locs[0] - seg_centroid[0],
        seg_bound_locs[1] - seg_centroid[1],
    )
    intensities = raw_arr[seg_bound_locs]

    edge_data, peak_angles, peak_intensities, peak_details, peak_width_details = (
        get_peaks_of_edge_intensities(angles, intensities)
    )
    peak_args = tuple(
        zip(
            peak_angles,
            peak_intensities,
            peak_details["prominences"].tolist(),
            peak_width_details["width_heights"].tolist(),
            peak_width_details["left_ips"].tolist(),
            peak_width_details["right_ips"].tolist(),
            strict=True,
        )
    )

    fig = plt.figure(figsize=(9, 3))
    gs = fig.add_gridspec(nrows=1, ncols=3)  # , width_ratios=[1, 1], wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax1.imshow(overlay[y_slice, x_slice])
    ax1.set_frame_on(False)
    ax1.axis("off")
    ax2.scatter(
        record["angle"].values[0], record["intensity"].values[0], marker="o", color="tab:red", s=5
    )
    ax2.scatter(edge_data.angle, edge_data.intensity, marker=".", color="tab:blue", s=3)
    ax2.plot(edge_data.angle, edge_data.intensity_smoothed, color="k", lw=1, ls="--")
    ax2.set_xlabel("Angle (rad)")
    ax2.set_xticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
    ax2.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    for (
        angle,
        intens,
        prominence,
        width_height,
        left_width_idx,
        right_width_idx,
    ) in peak_args:
        ax2.vlines(x=angle, ymin=intens - prominence, ymax=intens, color="tab:orange", lw=1, ls="-")
        ax2.hlines(
            y=width_height,
            xmin=edge_data.angle.iloc[int(left_width_idx)],
            xmax=edge_data.angle.iloc[int(right_width_idx)],
            color="tab:orange",
            lw=1,
            ls="-",
        )
        ax2.annotate(
            text=" ",
            xy=(angle, intens),
            xytext=(angle + 0.3, intens + 20),
            xycoords="data",
            arrowprops={
                "arrowstyle": "-|>",
                "linestyle": "-",
                "color": "tab:orange",
                "lw": 1,
            },
        )

    return fig, (ax1, ax2)


def get_peaks_of_edge_intensities(
    angles: np.ndarray,
    intensities: np.ndarray,
    peak_prominence: int = 50,
    peak_to_peak_distance_minimum: int = 50,
    smoothing_window_size: int = 5,
    rel_dist_from_peak_to_measure_width: float = 0.8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    edge_data = pd.DataFrame({"angle": angles, "intensity": intensities}).sort_values("angle")
    # edge_data["intensity_smoothed"] = ndimage.gaussian_filter1d(edge_data["intensity"], sigma=3)
    # edge_data["intensity_smoothed"] = (
    #     edge_data["intensity"]
    #     .rolling(window=smoothing_window_size, center=True, win_type="gaussian")
    #     .mean(std=7)
    # )
    edge_data["intensity_smoothed"] = (
        edge_data["intensity"].rolling(window=smoothing_window_size, center=True).quantile(0.5)
    )
    peak_locs, peak_details = find_peaks(
        edge_data.intensity_smoothed,
        distance=peak_to_peak_distance_minimum,
        prominence=peak_prominence,
    )
    peak_width_details = peak_widths(
        edge_data.intensity_smoothed,
        peaks=peak_locs,
        rel_height=rel_dist_from_peak_to_measure_width,
        prominence_data=tuple(peak_details.values()),
    )
    peak_width_details = dict(
        zip(("widths", "width_heights", "left_ips", "right_ips"), peak_width_details, strict=True)
    )

    peak_angles = edge_data.angle.iloc[peak_locs].tolist()
    peak_intensities = edge_data.intensity_smoothed.iloc[peak_locs].tolist()
    return edge_data, peak_angles, peak_intensities, peak_details, peak_width_details
