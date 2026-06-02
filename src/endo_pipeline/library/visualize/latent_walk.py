"""Module for visualizing latent walks as grids of reconstructed image crops."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    column_names: list[str],
    save_path: Path,
    file_name: str,
    file_format: Literal[".png", ".svg", ".pdf"] = ".svg",
    show_values: bool = True,
    label_sigmas: bool = True,
    figsize: tuple[float, float] | None = None,
    scale_bar_um: int = 10,
) -> None:
    """Plot and save a grid of reconstructed image crops representing a latent walk.

    Parameters
    ----------
    array_of_crops
        Array of shape (num_dims, num_steps, h, w) containing the reconstructed
        image crops.
    coordinate_values
        Array of shape (num_dims, num_steps) containing the coordinate values
        for each dimension and step.
    column_names
        A list of column names corresponding to each dimension in the latent
        walk.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.
    file_format
        Format of the output figure file (e.g., ".png", ".svg", ".pdf").
    show_values
        True to show the coordinate value on the image, False otherwise.
    label_sigmas
        True to label the column titles with sigma values, False to label with
        step number.
    figsize
        Optional tuple specifying the figure size in inches (width, height). If
        not provided, defaults to (6.5, num_rows) where num_rows is the
        number of dimensions in the latent walk.
    scale_bar_um
        Length of the scale bar in micrometers to add to each subplot.
    """
    # Set up the grid
    num_rows = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]
    gs = GridSpec(num_rows, num_steps, wspace=0, hspace=0)

    # Desired figure dimensions in inches
    if figsize is None:
        figsize = (MAX_FIGURE_WIDTH, num_rows)

    # Set up the figure
    fig = plt.figure(figsize=figsize)

    for i in range(num_rows):
        for j in range(num_steps):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(array_of_crops[i, j], cmap="gray")

            # Turn off x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Ensure figures remains square
            ax.set_aspect("equal")

            # Remove axis border
            ax.spines["top"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")

            # Add value label
            if show_values:
                value_label = f"{np.round(coordinate_values[i][j], 2)}"
                ax.annotate(
                    value_label,
                    xy=(0, 1),
                    xycoords="axes fraction",
                    xytext=(+0.5, -0.5),
                    textcoords="offset fontsize",
                    fontsize=FONTSIZE_XSMALL,
                    verticalalignment="top",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

            # Titles only on first row
            if i == 0:
                if label_sigmas:
                    column_title = f"{j - (num_steps // 2)}{Unicode.SIGMA}"
                    ax.set_title(column_title, fontsize=10, pad=5)

            # Y labels only on first column
            if j == 0:
                ylabel = get_label_for_column(column_names[i])
                # if "pc" in the label, capitalize "PC"
                if "pc" in ylabel.lower():
                    ylabel = ylabel.upper()
                ax.set_ylabel(ylabel, labelpad=5)

    for i, ax in enumerate(fig.axes):
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=2.5,
            padding=5,
            include_label=True if i == 0 else False,
        )

    file_name = f"{file_name}_scale_bar_{scale_bar_um}um"
    save_plot_to_path(fig, save_path, file_name, file_format=file_format)
