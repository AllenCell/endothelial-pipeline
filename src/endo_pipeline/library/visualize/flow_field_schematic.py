"""Methods for constructing schematics for the flow field supplementary figure."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image, save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    crop_image,
    log_normalize_image,
    std_dev,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.examples import FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_HEIGHT
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


def make_real_image_panel(
    savedir: Path,
    figsize: tuple[float, float] = (MAX_FIGURE_HEIGHT // 4, MAX_FIGURE_HEIGHT // 2),
    scale_bar_um: int = 20,
    grid_crop_position: tuple[int, int] = (0, 0),
) -> Path:
    """Build the panel showing a grid crop from t to t+1 for a given example image."""

    processed_images = []
    for example in FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES:
        dataset_config = load_dataset_config(example.dataset_name)
        location = get_zarr_location_for_position(dataset_config, position=example.position)
        bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

        bf_std_dev = std_dev(bf_image, axis=0)

        log_bf_std_dev = log_normalize_image(bf_std_dev)
        log_bf_std_dev = contrast_stretching(log_bf_std_dev)

        log_bf_std_dev = crop_image(
            log_bf_std_dev,
            example.crop_x_start,
            example.crop_y_start,
            2 * NATIVE_ZARR_RESOLUTION_CROP_SIZE,
        )
        processed_images.append(log_bf_std_dev)

    labels = ["t", "t+1"]
    fig: plt.Figure = make_contact_sheet(
        processed_images,
        max_rows=len(processed_images),
        max_cols=1,
        row_titles=labels,
        fig_kwargs={"figsize": figsize},
    )

    fig.subplots_adjust(hspace=2.5)

    for ax, img, label in zip(fig.axes, processed_images, labels, strict=True):
        ax.imshow(img, cmap="gray")
        ax.set_ylabel(label)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=15,
            padding=25,
        )

        # add highlighted box to show crop region used for flow field construction
        rect = plt.Rectangle(
            grid_crop_position,
            NATIVE_ZARR_RESOLUTION_CROP_SIZE,
            NATIVE_ZARR_RESOLUTION_CROP_SIZE,
            edgecolor="magenta",
            facecolor="none",
            linewidth=2,
            clip_on=False,
        )
        ax.add_patch(rect)

    fig.axes[-1].text(
        0.95,
        0.09,
        f"{scale_bar_um} {Unicode.MU}m",
        color="white",
        transform=fig.axes[-1].transAxes,
        fontsize=FONTSIZE_MEDIUM,
        va="bottom",
        ha="right",
    )

    filename = "flow_field_example_t_to_tp1"
    save_plot_to_path(fig, savedir, filename, file_format=".svg")
    image_panel_path = savedir / f"{filename}.svg"

    return image_panel_path


def _make_2d_pcolormesh(
    axes: plt.Axes,
    data_2d: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
) -> QuadMesh:
    """Make a 2D pcolormesh plot with consistent styling."""
    pcm = axes.pcolormesh(
        x_edges,
        y_edges,
        data_2d.T,
        cmap=cmap,
        shading="auto",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    if axes_xlabel is not None:
        axes.set_xlabel(axes_xlabel)
    if axes_ylabel is not None:
        axes.set_ylabel(axes_ylabel)
    if axes_aspect is not None:
        axes.set_aspect(axes_aspect)
    return pcm


def _add_target_bin_border(
    ax: plt.Axes,
    target_x: float,
    target_y: float,
    bin_width_x: float,
    bin_width_y: float,
    color: str = "magenta",
    linewidth: float = 2.5,
    label: str | None = "target bin",
) -> None:
    """Draw a square border around the target bin."""
    rect = Rectangle(
        (target_x - bin_width_x / 2, target_y - bin_width_y / 2),
        bin_width_x,
        bin_width_y,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label=label,
        zorder=5,
    )
    ax.add_patch(rect)
