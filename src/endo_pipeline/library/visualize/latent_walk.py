import logging
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    column_names: list[str],
    save_path: Path,
    file_name: str,
    show_values: bool = True,
    label_sigmas: bool = True,
) -> None:
    """
    Plot a grid of reconstructed image crops representing a latent walk.

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
    use_pcs
        True if latent walk was performed along PC axes, False otherwise.
    show_value
        True to show the coordinate value on the image, False otherwise.
    label_sigmas
        True to label the column titles with sigma values, False to label with
        step number.
    """
    # Set up the grid
    num_rows = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]
    gs = GridSpec(num_rows, num_steps, wspace=0)

    # Desired figure dimensions in inches
    width = 6.5
    height = num_rows

    # Set up the figure
    fig = plt.figure(figsize=(width, height))

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
                    fontsize=8,
                    verticalalignment="top",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

            # Titles only on first row
            if i == 0:
                if label_sigmas:
                    column_title = f"{j - (num_steps // 2)}\u03c3"
                else:
                    column_title = f"Step {j+1}"
                ax.set_title(column_title, fontsize=10, pad=5)

            # Y labels only on first column
            if j == 0:
                ylabel = get_label_for_column(column_names[i], capitalize=True)
                ax.set_ylabel(ylabel, labelpad=5)

            # Plot scalebar only on first image
            if i == 0 and j == 0:
                scalebar_um = 10
                add_scalebar(
                    ax,
                    scale_bar_um=scalebar_um,
                    pixel_size=PIXEL_SIZE_3i_20x,
                    bar_thickness=5,
                    padding=10,
                )

    gs.tight_layout(fig, pad=0.25)
    plt.show()

    output_file = (save_path / file_name).with_suffix(".pdf")
    fig.savefig(output_file)
    plt.close(fig)
