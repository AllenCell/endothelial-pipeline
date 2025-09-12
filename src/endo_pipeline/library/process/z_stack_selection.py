import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.array import Array
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

from endo_pipeline.configs import DatasetConfig, get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import load_zarr_as_dask_array, save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching
from endo_pipeline.settings import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

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
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    bf_stack_all_frames = load_zarr_as_dask_array(zarr_file, channels=["BF"], level=1)

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
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    bf_stack_all_frames = load_zarr_as_dask_array(zarr_file, channels=["BF"], level=1)

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
    stdevs: list, center_plane: int, dataset: str, position: int, frame: int, output_dir: Path
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

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stdevs)

    # Add a vertical line at the center plane index
    ax.axvline(center_plane, color="red", linestyle="--", label=f"Center Plane ({center_plane})")

    ax.set_title(f"{dataset} P{position} frame: {frame}")
    ax.set_xlabel("Plane Index")
    ax.set_ylabel("Standard Deviations of BF Planes")
    ax.legend()  # Add legend for the center plane label

    fname = f"standard_devs_{dataset}_P{position}_{frame}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()


def visualize_slice_selection(
    bf_stack: np.ndarray,
    cdh5_stack: np.ndarray,
    center_plane: int,
    lower_offset: int,
    upper_offest: int,
    dataset: str,
    position: int,
    frame: int,
    output_dir: Path,
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

    Returns
    -------
    None
    """
    im_center = contrast_stretching(bf_stack[center_plane].compute())
    im_below = contrast_stretching(bf_stack[center_plane - lower_offset].compute())
    im_above = contrast_stretching(bf_stack[center_plane + upper_offest].compute())

    cdh5_center = contrast_stretching(cdh5_stack[center_plane].compute())
    cdh5_below = contrast_stretching(cdh5_stack[center_plane - lower_offset].compute())
    cdh5_above = contrast_stretching(cdh5_stack[center_plane + upper_offest].compute())

    # Create subplots with a 2x3 grid
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10)
    )  # Adjusted figure size for 2 rows and 3 columns

    # First row: BF stack
    axes[0, 0].imshow(im_below, cmap="gray")
    axes[0, 0].set_title(f"BF Plane {center_plane - lower_offset} (-{lower_offset})")

    axes[0, 1].imshow(im_center, cmap="gray")
    axes[0, 1].set_title(f"BF Center Plane {center_plane}")

    axes[0, 2].imshow(im_above, cmap="gray")
    axes[0, 2].set_title(f"BF Plane {center_plane + upper_offest} (+{upper_offest})")

    # Second row: CDH5 stack
    axes[1, 0].imshow(cdh5_below, cmap="gray")
    axes[1, 0].set_title(f"CDH5 Plane {center_plane - lower_offset} (-{lower_offset})")

    axes[1, 1].imshow(cdh5_center, cmap="gray")
    axes[1, 1].set_title(f"CDH5 Center Plane {center_plane}")

    axes[1, 2].imshow(cdh5_above, cmap="gray")
    axes[1, 2].set_title(f"CDH5 Plane {center_plane + upper_offest} (+{upper_offest})")

    for ax in axes.flat:
        ax.axis("off")

    # Adjust layout and add a title
    fig.suptitle(f"{dataset} P{position} Frame {frame}", fontsize=16)
    plt.tight_layout()

    # Save the plot
    fname = f"plane_selection_vis_{dataset}_P{position}_{frame}_offset{lower_offset}_{upper_offest}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()


def plot_global_center_plane(
    center_planes: list, dataset: str, position: int, output_dir: Path
) -> tuple[float, float]:
    """
    Plot center planes for a dataset and return mean center plane and standard deviation.

    Parameters
    ----------
    center_planes
        List of center plane indices.
    dataset
        Dataset name for labeling the plot.
    position
        Position index for labeling the plot.
    output_dir
        Directory to save the output plot.

    Returns
    -------
    tuple
        Mean and standard deviation of center planes.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # Create two subplots

    # Compute mean and standard deviation of center planes
    mean_center_plane = np.mean(center_planes)
    std_center_plane = np.std(center_planes)

    # First plot: Center plane vs frame index
    ax[0].plot(range(len(center_planes)), center_planes, "ro", alpha=0.7)
    ax[0].set_xlabel("Timepoint (frames)")
    ax[0].set_ylabel("Center Plane Index")
    ax[0].set_title(f"{dataset}, Position {position}")
    ax[0].set_ylim(0, 25)

    # Add horizontal lines for mean and ±1 standard deviation
    ax[0].axhline(
        mean_center_plane, color="black", linestyle="-", label=f"Mean: {mean_center_plane:.2f}"
    )
    ax[0].axhline(
        mean_center_plane - std_center_plane,
        color="cyan",
        linestyle="--",
        label=f"-1 Std Dev: {mean_center_plane - std_center_plane:.2f}",
    )
    ax[0].axhline(
        mean_center_plane + std_center_plane,
        color="cyan",
        linestyle="--",
        label=f"+1 Std Dev: {mean_center_plane + std_center_plane:.2f}",
    )
    ax[0].legend()

    # Second plot: Histogram of center planes
    ax[1].hist(center_planes, bins=25, color="gray", alpha=0.7)
    ax[1].axvline(
        mean_center_plane, color="black", linestyle="-", label=f"Mean: {mean_center_plane:.2f}"
    )
    ax[1].axvline(mean_center_plane - std_center_plane, color="cyan", linestyle="--")
    ax[1].axvline(mean_center_plane + std_center_plane, color="cyan", linestyle="--")
    ax[1].set_xlabel("Center Plane Index")
    ax[1].set_xlim(0, 25)
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    fname = f"global_center_plane_{dataset}_P{position}"
    save_plot_to_path(fig, output_dir, fname)
    plt.show()
    plt.close(fig)

    return mean_center_plane, std_center_plane


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
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timepoint, level=1, squeeze=True
    )
    cdh5_stack = load_zarr_as_dask_array(
        zarr_file, channels=["EGFP"], timepoints=timepoint, level=1, squeeze=True
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


def plot_histogram_upper_slices_available(datasets: list[str], save_dir: Path) -> None:
    """Plot histogram of available slices above center slice across datasets."""
    data = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)
        for position in dataset_config.zarr_positions:
            center_slice = dataset_config.center_z_plane[position]
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

    fig = plt.figure(figsize=(6, 6))
    plt.hist(df["available_slices_above"], bins=range(8, 20, 1), align="left", edgecolor="black")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Available Slices Above Center Slice")
    plt.ylabel("Number of Positions")

    df_11 = df[df["available_slices_above"] == 11]
    limiting_datasets = df_11.dataset.unique()
    text = "Datasets in 11:\n" + "\n".join(limiting_datasets)
    plt.text(
        0.95,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    plt.show()
    save_plot_to_path(fig, save_dir, "available_slices_above_center_histogram")


def compute_profiles(zarr_file, center_slice: int, timepoint: int):
    """Compute normalized BF std and CDH5 hist profiles for a given position/timepoint."""

    # Load stacks
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timepoint, level=1, squeeze=True
    )
    cdh5_stack = load_zarr_as_dask_array(
        zarr_file, channels=["EGFP"], timepoints=timepoint, level=1, squeeze=True
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
    """

    colormap = colormaps["tab20"]
    colors = [colormap(i / len(datasets)) for i in range(len(datasets))]

    if mode not in ["by_position", "by_dataset"]:
        logger.error("Invalid mode: [ %s ]. Choose 'by_position' or 'by_dataset'.", mode)
        raise ValueError("mode must be 'by_position' or 'by_dataset'")

    if mode == "by_position":
        for timepoint in timepoints:
            for position in range(6):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                for i, dataset in enumerate(datasets):
                    dataset_config = load_dataset_config(dataset)
                    if dataset_config.center_z_plane is None:
                        print(f"Center slice is None for dataset {dataset_config.name}, skipping")
                        continue
                    center_slice = dataset_config.center_z_plane[position]
                    zarr_file = get_zarr_file_for_position(dataset_config, position)

                    x, bf_std_norm, cdh5_hist_norm = compute_profiles(
                        zarr_file, center_slice, timepoint
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

                for position in dataset_config.zarr_positions:
                    center_slice = dataset_config.center_z_plane[position]
                    zarr_file = get_zarr_file_for_position(dataset_config, position)

                    x, bf_std_norm, cdh5_hist_norm = compute_profiles(
                        zarr_file, center_slice, timepoint
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
