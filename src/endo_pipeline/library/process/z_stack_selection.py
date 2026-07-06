import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator

from endo_pipeline.configs import ChannelName, DatasetConfig, load_dataset_config
from endo_pipeline.io import load_image, save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching, crop_image
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dataset_annotations import REPRESENTATIVE_ANNOTATION_TIMEPOINT
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM
from endo_pipeline.settings.image_data import (
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
    PIXEL_SIZE_3i_20x_RESOLUTION_1,
)

logger = logging.getLogger(__name__)


def calculate_global_center_plane(
    dataset_config: DatasetConfig, position: int, max_timepoints: int | None = None
) -> dict[str, int | list[int]]:
    """
    Calculate the global center plane for a single position in a dataset.

    This function computes the center plane for each frame in a brightfield (BF)
    z-stack by finding the slice with the minimum standard deviation. It then
    calculates the mean and standard deviation of the center planes across all
    frames.

    Parameters
    ----------
    dataset_config
        Configuration object containing metadata and paths for the dataset.
    position
        The position index within the dataset to analyze.
    max_timepoints
        Maximum number of timepoints to use for calculating plane.

    Returns
    -------
    :
        Dictionary containing calculated center plane information.
    """

    zarr_location = get_zarr_location_for_position(dataset_config, position)
    bf_stack_all_frames = load_image(zarr_location, channels=[ChannelName.BF], level=1)

    center_planes = []

    timepoints = max_timepoints or dataset_config.duration

    for frame in range(0, timepoints, 1):
        # Extract the BF stack for the current frame
        bf_stack = bf_stack_all_frames[frame].squeeze()

        # Compute standard deviations for all slices in the current frame
        stdevs = bf_stack.std(axis=(1, 2)).compute()

        # Find the center plane with the minimum standard deviation
        center_plane = max(0, np.argmin(stdevs).astype(int))
        center_planes.append(center_plane)

    # Calculate mean and std dev for selected center planes across timepoints
    mean = np.mean(center_planes)
    std_dev = np.std(center_planes)

    # Calculate std dev of each slice for first timepoint
    representative_slices = bf_stack_all_frames[REPRESENTATIVE_ANNOTATION_TIMEPOINT].squeeze()
    slice_std_devs = [plane.std().compute() for plane in representative_slices]

    return {
        Column.POSITION: position,
        Column.Annotations.CENTER_PLANES: center_planes,
        Column.Annotations.CENTER_PLANE_SLICES_STD_DEVS: slice_std_devs,
        Column.Annotations.CENTER_PLANE_MEAN: round(mean),
        Column.Annotations.CENTER_PLANE_STD_DEV: round(std_dev),
    }


def plot_standard_devs_per_slice(
    stdevs: list,
    center_plane: int,
    dataset: str,
    position: int,
    frame: int,
    output_dir: Path,
    figure_size: tuple = (2.5, 2.15),
) -> None:
    """
    Plot the standard deviations of each slice vs plane index, highlighting the
    center plane.

    Parameters
    ----------
    stdevs
        List of standard deviation values for each BF plane in the z-stack.
    center_plane
        The index of the center plane to highlight on the plot.
    dataset
        The name of the dataset.
    position
        The position index.
    frame
        The frame index.
    output_dir
        Directory where the plot will be saved.
    figure_size
        Size of the figure to be created (width, height).
    """

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.plot(stdevs)

    # Add a vertical line at the center plane index
    ax.axvline(
        center_plane,
        color="red",
        linestyle="--",
        label=f"In-focus Z-slice\n(index = {center_plane})",
    )

    ax.set_title("In-focus Z-slice per timepoint", fontsize=FONTSIZE_MEDIUM)
    ax.set_xlabel("Z-slice (index)")
    ax.set_ylabel("Std. dev. of BF\npixel intensities (a.u.)")
    ax.legend(loc="lower right", handlelength=1.5, handletextpad=0.4)

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3

    fname = f"standard_devs_{dataset}_P{position}_{frame}"
    save_plot_to_path(
        fig,
        output_dir,
        fname,
        file_format=".svg",
        tight_layout=False,
    )
    plt.show()


def visualize_slice_selection(
    dataset_config: DatasetConfig,
    center_plane: int,
    position: int,
    frame: int,
    output_dir: Path,
    figure_size: tuple = (5.0, 3.0),
    lower_offset: int = LOWER_Z_SLICE_OFFSET,
    upper_offest: int = UPPER_Z_SLICE_OFFSET,
    pixel_size: float = PIXEL_SIZE_3i_20x_RESOLUTION_1,
    example_crop_x_start: int = 100,
    example_crop_y_start: int = 100,
    crop_size: int = 500,
) -> None:
    """
    Plot the center z-slice with slices n planes above and below the center slice
    for the BF and CDH5 channels.

    Parameters
    ----------
    bf_stack
        Brightfield image stack.
    cdh5_stack
        CDH5 image stack.
    center_plane
        Index of the center plane.
    lower_offset
        Number of planes below the center plane to visualize.
    upper_offest
        Number of planes above the center plane to visualize.
    dataset
        Dataset name.
    position
        Zarr position index.
    frame
        Frame index.
    output_dir
        Directory to save the output plot.
    figure_size
        Size of the figure to be created (width, height).
    """

    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    bf_stack = load_image(
        zarr_loc, channels=[ChannelName.BF], timepoints=frame, level=1, squeeze=True
    )
    cdh5_stack = load_image(
        zarr_loc, channels=[ChannelName.EGFP], timepoints=frame, level=1, squeeze=True
    )

    method = "min-max"
    center_im = bf_stack[center_plane].compute()
    custom_low = np.percentile(center_im, 1.5)
    custom_high = np.percentile(center_im, 98.5)

    im_center = contrast_stretching(
        center_im, method=method, custom_range=(custom_low, custom_high)
    )
    im_below = contrast_stretching(
        bf_stack[center_plane - lower_offset].compute(),
        method=method,
        custom_range=(custom_low, custom_high),
    )
    im_above = contrast_stretching(
        bf_stack[center_plane + upper_offest].compute(),
        method=method,
        custom_range=(custom_low, custom_high),
    )

    cdh5_im = cdh5_stack[center_plane].compute()
    cdh5_low = np.percentile(cdh5_im, 1.5)
    cdh5_high = np.percentile(cdh5_im, 98.5)

    cdh5_center = contrast_stretching(
        cdh5_stack[center_plane].compute(), method=method, custom_range=(cdh5_low, cdh5_high)
    )
    cdh5_below = contrast_stretching(
        cdh5_stack[center_plane - lower_offset].compute(),
        method=method,
        custom_range=(cdh5_low, cdh5_high),
    )
    cdh5_above = contrast_stretching(
        cdh5_stack[center_plane + upper_offest].compute(),
        method=method,
        custom_range=(cdh5_low, cdh5_high),
    )

    im_center = crop_image(im_center, example_crop_x_start, example_crop_y_start, crop_size)
    im_below = crop_image(im_below, example_crop_x_start, example_crop_y_start, crop_size)
    im_above = crop_image(im_above, example_crop_x_start, example_crop_y_start, crop_size)
    cdh5_center = crop_image(cdh5_center, example_crop_x_start, example_crop_y_start, crop_size)
    cdh5_below = crop_image(cdh5_below, example_crop_x_start, example_crop_y_start, crop_size)
    cdh5_above = crop_image(cdh5_above, example_crop_x_start, example_crop_y_start, crop_size)

    panels = [im_below, im_center, im_above, cdh5_below, cdh5_center, cdh5_above]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=2,
        max_cols=3,
        col_titles=[
            f"Lower Z-slice (-{lower_offset} offset)",
            "In focus Z-slice",
            f"Upper Z-slice (+{upper_offest} offset)",
        ],
        row_titles=["BF", "VE-cadherin"],
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    scale_bar_um = 100

    for i, ax in enumerate(fig.axes):

        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=pixel_size,
            location="lower right",
            bar_thickness=12,
            padding=12,
            label_xy=(0.96, 0.06),
            include_label=True if i == 0 else False,
        )

    dataset = dataset_config.name
    fname = f"plane_selection_vis_{dataset}_P{position}_{frame}_offset{lower_offset}_{upper_offest}_scalebar{scale_bar_um}um"
    save_plot_to_path(fig, output_dir, fname, tight_layout=False, file_format=".svg")
    plt.show()


def plot_global_center_plane(
    center_planes: list,
    dataset: str,
    position: int,
    output_dir: Path,
    figure_size: tuple = (2.5, 2.15),
    show_histogram: bool = True,
) -> None:
    """
    Plot the global center plane for a given dataset and position.

    Parameters
    ----------
    center_planes
        List of center planes for each timepoint.
    dataset
        Name of the dataset.
    position
        Position index.
    output_dir
        Directory to save the output plot.
    figure_size
        Size of the figure.
    show_histogram
        True to include histogram, False to only show scatter plot.
    """

    mean_cp = np.mean(center_planes)
    y_min, y_max = 0, 25  # fixed y-axis range

    if show_histogram:
        counts, bins = np.histogram(center_planes, bins=25)
        # 1 row, 2 cols, histogram narrower
        fig, ax = plt.subplots(
            1, 2, figsize=figure_size, sharey=True, gridspec_kw={"width_ratios": [3, 1]}
        )
    else:
        # Only scatter plot
        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
        ax = [ax]  # make it indexable like a list

    # Scatter plot
    ax[0].plot(
        range(len(center_planes)),
        center_planes,
        "ro",
        alpha=0.5,
        label="In-focus Z-slice per timepoint",
    )
    ax[0].set_xlabel("Timepoint (frames)")
    ax[0].set_ylabel("Z-slice (index)")
    ax[0].set_title("In-focus Z-slice per position", fontsize=FONTSIZE_MEDIUM)
    ax[0].set_ylim(y_min, y_max)
    ax[0].axhline(
        mean_cp, color="black", linestyle="-", label=f"Mean in-focus Z-slice\n(index={mean_cp:.0f})"
    )
    ax[0].legend(handlelength=1.0, handletextpad=0.4)

    # reduce label padding
    ax[0].xaxis.labelpad = 3
    ax[0].yaxis.labelpad = 3

    # Histogram (optional)
    if show_histogram:
        ax[1].hist(center_planes, bins=25, orientation="horizontal", color="gray", alpha=0.7)
        ax[1].axhline(mean_cp, color="black", linestyle="-")
        ax[1].set_xlabel("Count")
        ax[1].set_xlim(0, counts.max() * 1.1)  # tighter x-limits
        ax[1].set_ylim(y_min, y_max)

        # Reduce whitespace
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.05)

    fname = f"global_center_plane_{dataset}_P{position}"
    save_plot_to_path(fig, output_dir, fname, file_format=".svg", tight_layout=False)
    plt.show()
    plt.close(fig)


def plot_histogram_upper_slices_available(
    datasets: list[str], save_dir: Path, figure_size: tuple
) -> None:
    """Plot histogram of available slices above center slice across datasets."""
    data = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)
        for position in dataset_config.zarr_positions:
            if dataset_config.center_z_plane is None:
                logger.warning(
                    "Center z-plane information is missing for" " dataset [ %s ], skipping", dataset
                )
                continue
            center_slice = dataset_config.center_z_plane.get(position)
            if center_slice is None:
                logger.warning(
                    "Center z-slice information missing for position [ %s ] "
                    "in dataset [ %s ], skipping.",
                    position,
                    dataset,
                )
                continue
            top_slice = 24
            available_slices_above = top_slice - center_slice
            data.append(
                {
                    "dataset": dataset_config.name,
                    "position": position,
                    "available_slices_above": available_slices_above,
                }
            )

    df = pd.DataFrame(data)

    fig = plt.figure(figsize=figure_size, layout="constrained")
    plt.hist(
        df["available_slices_above"],
        bins=range(10, 18, 1),
        align="left",
        edgecolor="black",
        facecolor="grey",
    )
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # Set x-axis ticks at every 1 interval
    plt.xlim(10, None)  # Set 10 as the minimum value on the x-axis
    plt.xlabel("Number of slices\nabove in-focus Z")
    plt.ylabel("Number of positions")

    # reduce label padding
    plt.gca().xaxis.labelpad = 3
    plt.gca().yaxis.labelpad = 3

    save_plot_to_path(
        fig, save_dir, "n_slices_above_in_focus_z_histogram", file_format=".svg", tight_layout=False
    )
