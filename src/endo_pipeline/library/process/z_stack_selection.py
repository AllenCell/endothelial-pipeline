import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.array import Array
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator, MultipleLocator

from endo_pipeline.configs import DatasetConfig, load_dataset_config
from endo_pipeline.io import load_image, save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching, crop_image
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import ImageLocation, get_zarr_location_for_position
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM
from endo_pipeline.settings.image_data import (
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
    PIXEL_SIZE_3i_20x_RESOLUTION_1,
)

logger = logging.getLogger(__name__)


def calculate_global_center_plane(
    dataset_config: DatasetConfig, position: int, save_dir: Path
) -> dict[str, int]:
    """
    Calculate the global center plane for a single position in a dataset.

    This function computes the center plane for each frame in a brightfield (BF) z-stack
    by finding the slice with the minimum standard deviation. It then calculates the
    mean and standard deviation of the center planes across all frames and optionally
    visualizes the results.

    Parameters
    ----------
    dataset_config
        Configuration object containing metadata and paths for the dataset.
    position
        The position index within the dataset to analyze.
    save_dir
        Directory where the visualization of the global center plane will be saved.

    Returns
    -------
    A dictionary containing:
    - "position": The analyzed position index.
    - "mean_center_plane": The mean center plane across all frames.
    - "std_dev_center_plane": The standard deviation of the center plane across all frames.
    """

    zarr_location = get_zarr_location_for_position(dataset_config, position)
    bf_stack_all_frames = load_image(zarr_location, channels=["BF"], level=1)

    center_planes = []

    for frame in range(0, dataset_config.duration, 1):
        # Extract the BF stack for the current frame
        bf_stack = bf_stack_all_frames[frame].squeeze()

        # Compute standard deviations for all slices in the current frame
        stdevs = bf_stack.std(axis=(1, 2)).compute()

        # Find the center plane with the minimum standard deviation
        center_plane = max(0, np.argmin(stdevs))
        center_planes.append(center_plane)

    mean, std_dev = plot_global_center_plane(center_planes, dataset_config.name, position, save_dir)

    return {
        "position": position,
        "mean_center_plane": round(mean),
        "std_dev_center_plane": round(std_dev),
    }


def get_center_plane_for_position(dataset_config: DatasetConfig, position: int) -> int:
    """
    Calculate the global center plane for a single position across all frames.

    This function determines the center plane of a brightfield (BF) z-stack for a given position
    by analyzing the standard deviations of pixel intensities across all frames. The center plane
    is calculated as the plane with the lowest standard deviation for each frame, and the global
    center plane is determined as the average of these values across all frames.

    Parameters
    ----------
    dataset_config
        Configuration object with dataset-specific information.
    position
        The position index.

    Returns
    -------
    int
        The global center plane index for the specified position.
    """

    zarr_location = get_zarr_location_for_position(dataset_config, position)
    bf_stack_all_frames = load_image(zarr_location, channels=["BF"], level=1)

    center_planes = []

    for frame in range(0, dataset_config.duration, 1):
        bf_stack = bf_stack_all_frames[frame].squeeze()
        stdevs = bf_stack.std(axis=(1, 2)).compute()
        center_plane_selection = cast(float, max(0, np.argmin(stdevs)))
        center_planes.append(center_plane_selection)

    del bf_stack_all_frames, stdevs  # Free memory after processing

    mean_center_plane = np.mean(center_planes)
    global_center_plane = round(mean_center_plane, 0)

    return int(global_center_plane)


def get_plane_indices(
    dataset_config: DatasetConfig,
    position: int,
    lower_offset: int,
    upper_offset: int,
) -> list[int]:
    """
    Get a list of plane indices based on the provided outputs about the global center.

    The indices are constrained between 0 and 24.

    Parameters
    ----------
    dataset_config
        Configuration object containing dataset-specific information.
    position
        The position index for which the plane indices are calculated.
    lower_offset
        The number of planes below the center plane to include.
    upper_offset
        The number of planes above the center plane to include.

    Returns
    -------
    list
        A list of plane indices within the specified range, constrained between 0 and 24.
    """
    if dataset_config.center_z_plane is None:
        logger.error(
            "Center z-plane information is missing for dataset [ %s ].", dataset_config.name
        )
        raise ValueError("Center z-plane information is missing in the dataset configuration.")
    global_center_plane = dataset_config.center_z_plane[position]
    lower_bound = max(0, global_center_plane - lower_offset)
    upper_bound = min(24, global_center_plane + upper_offset)

    return list(range(lower_bound, upper_bound + 1))


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
    Plot the standard deviations of each slice vs plane index, highlighting the center plane.

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


def calculate_center_planes_all_tp_for_pos(
    dataset_config: DatasetConfig, position: int
) -> list[int]:
    """
    Calculate the center plane for each frame in a brightfield (BF) z-stack for a given position.

    Args:
        dataset_config: Configuration object containing metadata for the dataset.
        position: The position index within the dataset to analyze.

    Returns:
        center_plantes: A list of center planes for each frame in the dataset.
    """

    zarr_location = get_zarr_location_for_position(dataset_config, position)
    bf_stack_all_frames = load_image(zarr_location, channels=["BF"], level=1)
    center_planes = []

    for frame in range(0, dataset_config.duration, 1):
        # Extract the BF stack for the current frame
        bf_stack = bf_stack_all_frames[frame].squeeze()

        # Compute standard deviations for all slices in the current frame
        stdevs = bf_stack.std(axis=(1, 2)).compute()

        # Find the center plane with the minimum standard deviation
        center_plane_tp = max(0, np.argmin(stdevs))
        center_planes.append(center_plane_tp)

    return center_planes


def visualize_slice_selection(
    bf_stack: Array,
    cdh5_stack: Array,
    center_plane: int,
    dataset: str,
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

    Returns
    -------
    None
    """
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
) -> tuple[float, float]:
    """
    Plot the global center plane for a given dataset and position.

    Parameters
    ----------
    center_planes : list
        List of center planes for each timepoint.
    dataset : str
        Name of the dataset.
    position : int
        Position index.
    output_dir : Path
        Directory to save the output plot.
    figure_size : tuple
        Size of the figure.
    show_histogram : bool, optional
        Whether to show the histogram, by default True.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the center planes.
    """
    mean_cp = np.mean(center_planes)
    std_cp = np.std(center_planes)
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

    return mean_cp, std_cp


def save_projection_image(image: np.ndarray, save_path: Path) -> None:
    """
    Save a processed 2D image to disk using grayscale colormap.

    Intentionally not using save_plot_to_path here to avoid saving as a figure and
    keep the image at its original resolution.

    Parameters
    ----------
    image
        The processed image array to be saved.
    save_path
        The file path where the image will be saved.
    """
    plt.imsave(save_path, image, cmap="gray")


def append_projection_outputs(
    stack: Array,
    zslice: list[int],
    process_fn: Callable,
    image_list: list,
    title_list: list,
    bottom_list: list,
    top_list: list,
) -> None:
    """
    Process a z-slice from a stack and append outputs to given containers.

    Parameters
    ----------
    stack
        The full 3D image stack.
    zslice
        Start and end indices for slicing the z-axis (e.g., [5, 20]).
    process_fn
        A function that processes the sliced stack and returns outputs, with
        the final output being the 2D projection.
    image_list
        List to collect the processed projection images.
    title_list
        List to collect slice label strings for titles.
    bottom_list
        List to collect the bottom slice of the projection range.
    top_list
        List to collect the top slice of the projection range.
    """
    slice_str = f"{zslice[0]}_{zslice[1]}"
    sliced = stack[zslice[0] : zslice[1]]
    outputs = process_fn(sliced)
    processed = outputs[-1]

    image_list.append(processed)
    title_list.append(slice_str)
    bottom_list.append(stack[zslice[0]])
    top_list.append(stack[zslice[1]])


def plot_image_row(
    images: list[np.ndarray],
    titles: list[str],
    dataset: str,
    position: int,
    timepoint: int,
    save_dir: Path,
    row_title: str = "Image",
    figsize: tuple[int, int] = (16, 4),
) -> None:
    """
    Plot a single row of images with corresponding titles.

    Parameters
    ----------
    images
        List of images to display.
    titles
        Titles corresponding to each image.
    dataset
        Name of the dataset for labeling the plot.
    position
        Position index for labeling the plot.
    timepoint
        Timepoint index for labeling the plot.
    save_dir
        Directory where the plot will be saved.
    row_title
        Prefix for each subplot title. Default is "Image".
    figsize
        Figure size for the matplotlib plot. Default is (16, 4).

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles, strict=True):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{row_title} {title}")
        ax.axis("off")
    plt.suptitle(f"{dataset} P{position}_T{timepoint}")
    plt.tight_layout()
    plt.show()
    fname = f"{dataset}_P{position}_T{timepoint}_{row_title.replace(' ', '_').lower()}_comparison"
    save_plot_to_path(fig, save_dir, fname)


def plot_bottom_top_slices(
    bottoms: list,
    tops: list,
    titles: list[str],
    dataset: str,
    position: int,
    timepoint: int,
    save_dir: Path,
    label: str,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Plot two rows of images showing bottom and top z-slices from each projection range.

    Parameters
    ----------
    bottoms
        List of bottom slices (as dask arrays) from each projection range.
    tops
        List of top slices (as dask arrays) from each projection range.
    titles
        Slice range labels (e.g., "0_16", "9_24") for each column.
    dataset
        Name of the dataset for labeling the plot.
    position
        Position index for labeling the plot.
    timepoint
        Timepoint index for labeling the plot.
    save_dir
        Directory where the plot will be saved.
    label
        Label prefix to distinguish BF or CDH5 channels in titles.
    figsize
        Size of the figure to plot. Default is (16, 8).

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, len(bottoms), figsize=figsize)

    for ax, img, title in zip(axes[0], bottoms, titles, strict=True):
        ax.imshow(contrast_stretching(img.compute()), cmap="gray")
        ax.set_title(f"{label} bottom {title}")
        ax.axis("off")

    for ax, img, title in zip(axes[1], tops, titles, strict=True):
        ax.imshow(contrast_stretching(img.compute()), cmap="gray")
        ax.set_title(f"{label} top {title}")
        ax.axis("off")

    plt.suptitle(f"{dataset} P{position}_T{timepoint}")
    plt.tight_layout()
    plt.show()

    fname = f"{dataset}_P{position}_T{timepoint}_{label}_bottom_top_slices"
    save_plot_to_path(fig, save_dir, fname)


def plot_vlines(
    axis: plt.Axes,
    center: int,
    lower_offset: int,
    upper_offset: int,
    y_min: float,
    y_max: float,
) -> None:
    """Plot vertical lines of global center and offsets."""
    axis.vlines(
        center,
        ymin=y_min,
        ymax=y_max,
        colors="red",
        linestyles="solid",
        label=f"Global Center Slice {center}",
    )
    axis.vlines(
        center - lower_offset,
        ymin=y_min,
        ymax=y_max,
        colors="magenta",
        linestyles="dashed",
        label=f"Center - {lower_offset} Slices {center - lower_offset}",
    )
    axis.vlines(
        center + upper_offset,
        ymin=y_min,
        ymax=y_max,
        colors="black",
        linestyles="dashed",
        label=f"Center + {upper_offset} Slices {center + upper_offset}",
    )


def visualize_z_slices_with_offsets(
    dataset_config: DatasetConfig, position: int, timepoint: int, save_dir: Path
) -> None:
    """Visualize specific z-slices from BF and CDH5 stacks based on center plane and offsets."""

    zarr_location = get_zarr_location_for_position(dataset_config, position)
    bf_stack = load_image(
        zarr_location, channels=["BF"], timepoints=timepoint, level=1, squeeze=True
    )
    cdh5_stack = load_image(
        zarr_location, channels=["EGFP"], timepoints=timepoint, level=1, squeeze=True
    )

    if dataset_config.center_z_plane is None:
        # Handle the case where the value is None
        print(f"The center slice is None for dataset {dataset_config.name}")
        return
    else:
        center_slice = dataset_config.center_z_plane[position]

    top_slice = 24
    available_slices_above = top_slice - center_slice

    if UPPER_Z_SLICE_OFFSET > available_slices_above:
        print(f"Not enough slices above center for dataset {dataset_config.name}, skipping...")
        return

    # Brightfield (bf) variables
    bf_center = bf_stack[center_slice, :, :].compute()
    bf_top = bf_stack[top_slice, :, :].compute()
    bf_lower_offset = bf_stack[center_slice - LOWER_Z_SLICE_OFFSET, :, :].compute()
    bf_upper_offset = bf_stack[center_slice + UPPER_Z_SLICE_OFFSET, :, :].compute()

    # CDH5 (cdh5) variables
    cdh5_center = cdh5_stack[center_slice, :, :].compute()
    cdh5_top = cdh5_stack[top_slice, :, :].compute()
    cdh5_lower_offset = cdh5_stack[center_slice - LOWER_Z_SLICE_OFFSET, :, :].compute()
    cdh5_upper_offset = cdh5_stack[center_slice + UPPER_Z_SLICE_OFFSET, :, :].compute()

    # Brightfield (bf) min and max calculations
    min_bf = np.percentile(bf_center, 0.2)
    max_bf = np.percentile(bf_center, 99.8)

    # CDH5 (cdh5) min and max calculations
    min_cdh5 = np.percentile(cdh5_center, 0.2)
    max_cdh5 = np.percentile(cdh5_center, 99.8)

    # Contrast stretching
    bf_center = contrast_stretching(bf_center, custom_range=(min_bf, max_bf))
    bf_top = contrast_stretching(bf_top, custom_range=(min_bf, max_bf))
    bf_lower_offset = contrast_stretching(bf_lower_offset, custom_range=(min_bf, max_bf))
    bf_upper_offset = contrast_stretching(bf_upper_offset, custom_range=(min_bf, max_bf))

    cdh5_center = contrast_stretching(cdh5_center, custom_range=(min_cdh5, max_cdh5))
    cdh5_top = contrast_stretching(cdh5_top, custom_range=(min_cdh5, max_cdh5))
    cdh5_lower_offset = contrast_stretching(cdh5_lower_offset, custom_range=(min_cdh5, max_cdh5))
    cdh5_upper_offset = contrast_stretching(cdh5_upper_offset, custom_range=(min_cdh5, max_cdh5))

    # Define the data and titles for each subplot
    bf_slices = [bf_lower_offset, bf_center, bf_upper_offset, bf_top]
    cdh5_slices = [cdh5_lower_offset, cdh5_center, cdh5_upper_offset, cdh5_top]
    titles = [
        f"Lower Offset Slice - {center_slice - LOWER_Z_SLICE_OFFSET}",
        f"Center Slice - {center_slice}",
        f"Upper Offset Slice - {center_slice + UPPER_Z_SLICE_OFFSET}",
        "Top Slice - 24",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))  # 2 rows, 4 columns
    for i in range(4):
        # Brightfield (BF) slices
        axes[0, i].imshow(bf_slices[i], cmap="gray")
        axes[0, i].set_title(f"BF {titles[i]}")
        axes[0, i].axis("off")

        # CDH5 slices
        axes[1, i].imshow(cdh5_slices[i], cmap="gray")
        axes[1, i].set_title(f"CDH5 {titles[i]}")
        axes[1, i].axis("off")

    plt.suptitle(f"{dataset_config.name} Position {position} Timepoint {timepoint}\n")
    plt.tight_layout()
    plt.show()

    save_plot_to_path(fig, save_dir, f"{dataset_config.name}_pos{position}_tp{timepoint}_im_slices")
    plt.close()


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

    plt.show()
    save_plot_to_path(fig, save_dir, "n_slices_above_in_focus_z_histogram", file_format=".svg")


def compute_profiles(
    zarr_location: ImageLocation, center_slice: int, timepoint: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized BF std and CDH5 hist profiles for a given position/timepoint."""

    # Load stacks
    bf_stack = load_image(
        zarr_location, channels=["BF"], timepoints=timepoint, level=1, squeeze=True
    )
    cdh5_stack = load_image(
        zarr_location, channels=["EGFP"], timepoints=timepoint, level=1, squeeze=True
    )

    # Calculate histograms
    cdh5_hist = np.array(
        [np.sum(cdh5_stack[z, :, :].compute()) for z in range(cdh5_stack.shape[0])]
    )
    bf_std = np.array([np.std(bf_stack[z, :, :].compute()) for z in range(bf_stack.shape[0])])

    # Normalize
    normalized_x = np.arange(len(bf_std)) - center_slice
    bf_std_norm = bf_std / bf_std[center_slice] if bf_std[center_slice] != 0 else bf_std
    cdh5_hist_norm = cdh5_hist / np.max(cdh5_hist) if np.max(cdh5_hist) != 0 else cdh5_hist

    return normalized_x, bf_std_norm, cdh5_hist_norm


def plot_normalized_profiles(
    datasets: list[str],
    timepoints: list[int],
    save_dir: Path,
    mode: Literal["by_position", "by_dataset"] = "by_position",
    lower_offset: int = LOWER_Z_SLICE_OFFSET,
    upper_offset: int = UPPER_Z_SLICE_OFFSET,
    n_positions: int = 6,
) -> None:
    """
    Plot normalized BF std and CDH5 hist profiles.

    Parameters
    ----------
    datasets
        List of dataset names.
    timepoints
        List of timepoints (e.g., [0, 90, 180, 270]).
    save_dir
        Directory to save the plots.
    mode
        Mode for looping (e.g., "by_position" or "by_dataset").
    lower_offset
        Z-slice offset below the center slice.
    upper_offset
        Z-slice offset above the center slice.
    n_positions
        Number of positions to visualize
    """

    colormap = colormaps["tab20"]
    colors = [colormap(i / len(datasets)) for i in range(len(datasets))]

    if mode not in ["by_position", "by_dataset"]:
        logger.error("Invalid mode: [ %s ]. Choose 'by_position' or 'by_dataset'.", mode)
        raise ValueError("mode must be 'by_position' or 'by_dataset'")

    if mode == "by_position":
        for timepoint in timepoints:
            for position in range(n_positions):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                for i, dataset in enumerate(datasets):
                    dataset_config = load_dataset_config(dataset)
                    if dataset_config.center_z_plane is None:
                        logger.warning(
                            "Center z-plane information is missing for dataset [ %s ], skipping",
                            dataset,
                        )
                        continue
                    center_slice = dataset_config.center_z_plane.get(position)
                    if center_slice is None:
                        logger.warning(
                            "Center z-slice information missing for position [ %s ] "
                            "in dataset [ %s ], skipping",
                            position,
                            dataset,
                        )
                        continue

                    zarr_location = get_zarr_location_for_position(dataset_config, position)
                    x, bf_std_norm, cdh5_hist_norm = compute_profiles(
                        zarr_location, center_slice, timepoint
                    )

                    axes[0].plot(x, bf_std_norm, color=colors[i])
                    axes[1].plot(x, cdh5_hist_norm, label=dataset_config.name, color=colors[i])

                # formatting + vlines
                axes[0].set_xlabel("Normalized Z Slice (Center = 0)")
                axes[0].set_ylabel("Normalized BF Standard Deviation")
                axes[0].set_ylim(0.99, 1.35)

                axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
                axes[1].set_ylabel("Normalized CDH5 Total Intensity")
                axes[1].set_ylim(0.84, 1.1)
                axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)

                plot_vlines(axes[0], 0, lower_offset, upper_offset, *axes[0].get_ylim())
                plot_vlines(axes[1], 0, lower_offset, upper_offset, *axes[1].get_ylim())

                plt.suptitle(
                    f"Position {position} TP {timepoint}, Offset -{lower_offset} to +{upper_offset}"
                )
                save_plot_to_path(fig, save_dir, f"pos{position}_tp{timepoint}_normalized_profiles")
                plt.show()

    elif mode == "by_dataset":
        for timepoint in timepoints:
            for dataset in datasets:
                dataset_config = load_dataset_config(dataset)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                for position in range(n_positions):
                    if dataset_config.center_z_plane is None:
                        logger.warning(
                            "Center z-plane information is missing for dataset [ %s ], skipping",
                            dataset,
                        )
                        continue
                    center_slice = dataset_config.center_z_plane.get(position)
                    if center_slice is None:
                        logger.warning(
                            "Center z-slice information missing for position [ %s ] "
                            "in dataset [ %s ], skipping",
                            position,
                            dataset,
                        )
                        continue

                    zarr_location = get_zarr_location_for_position(dataset_config, position)

                    x, bf_std_norm, cdh5_hist_norm = compute_profiles(
                        zarr_location, center_slice, timepoint
                    )

                    axes[0].plot(x, bf_std_norm)
                    axes[1].plot(x, cdh5_hist_norm, label=f"P{position}")

                # formatting + vlines
                axes[0].set_xlabel("Normalized Z Slice (Center = 0)")
                axes[0].set_ylabel("Normalized BF Standard Deviation")
                axes[0].set_ylim(0.99, 1.35)

                axes[1].set_xlabel("Normalized Z Slice (Center = 0)")
                axes[1].set_ylabel("Normalized CDH5 Total Intensity")
                axes[1].set_ylim(0.84, 1.1)
                axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)

                plot_vlines(axes[0], 0, lower_offset, upper_offset, *axes[0].get_ylim())
                plot_vlines(axes[1], 0, lower_offset, upper_offset, *axes[1].get_ylim())

                plt.suptitle(
                    f"{dataset_config.name} TP {timepoint}, "
                    f"Offset -{lower_offset} to +{upper_offset}"
                )
                save_plot_to_path(
                    fig, save_dir, f"{dataset_config.name}_tp{timepoint}_normalized_profiles"
                )
                plt.show()

    else:
        raise ValueError("mode must be 'by_position' or 'by_dataset'")
