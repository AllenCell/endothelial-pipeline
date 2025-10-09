from pathlib import Path
from typing import Literal

import matplotlib.axes as maxes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.settings.figures import FIGURE_SAVE_DPI


def add_scalebar(
    ax: maxes.Axes,
    scale_bar_um: float,
    pixel_size: float,
    location: str = "lower left",
    bar_thickness: float = 10,
    padding: float = 20,
    color: str = "white",
) -> None:
    """
    Add a scale bar to an image displayed with imshow (no text label).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the scale bar to.
    scale_bar_um : float
        Length of the scale bar in micrometers.
    pixel_size : float
        Size of a pixel in micrometers.
    location : str, optional
        One of 'upper left', 'upper right', 'lower left', 'lower right'.
    bar_thickness : float, optional
        Thickness of the scale bar in pixels.
    padding : float, optional
        Padding from the edge of the image in pixels.
    color : str, optional
        Color of the scale bar.
    """

    scale_bar_px = scale_bar_um / pixel_size
    length_px = scale_bar_px

    ny, nx = ax.images[0].get_array().shape  # image dimensions

    # Determine bar position
    if location == "upper left":
        x_start = padding
        y_start = padding
    elif location == "upper right":
        x_start = nx - padding - length_px
        y_start = padding
    elif location == "lower left":
        x_start = padding
        y_start = ny - padding - bar_thickness
    elif location == "lower right":
        x_start = nx - padding - length_px
        y_start = ny - padding - bar_thickness
    else:
        raise ValueError(f"Invalid location: {location}")

    # Draw the scale bar
    rect = patches.Rectangle(
        (x_start, y_start),
        length_px,
        bar_thickness,
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(rect)


def plot_image_thumbnail(
    image: np.ndarray,
    image_name: str,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int = FIGURE_SAVE_DPI,
    file_format: Literal[".png", ".pdf"] = ".png",
    scalebar_size_um: float | None = None,
    pixel_size: float | None = None,
    scalebar_location: Literal[
        "lower right", "lower left", "upper right", "upper left"
    ] = "lower left",
    bar_thickness: int = 10,
    bar_padding: int = 20,
) -> None:
    """
    Save a thumbnail image to a specified file path.

    This function saves a given image as a thumbnail in the specified format
    (e.g., PNG or PDF) with the desired resolution and figure size.

    Parameters
    ----------
    image : numpy.ndarray
        The image to save, represented as a NumPy array.
    image_name : str
        The name of the output image file (without extension).
    output_path : str
        The directory where the image will be saved.
    figsize : tuple of float
        The size of the figure in inches (width, height).
    dpi : int, optional
        The resolution of the saved image in dots per inch.
    file_format : str, optional
        The file format for the saved image (default is ".png").
    scalebar_size_um : float, optional
        Optional size of the scale bar in micrometers.
    pixel_size : float, optional
        Size of a pixel in micrometers. Required if `scalebar_size_um` is provided.
    scalebar_location : str, optional
        Location of the scale bar on the image. Options are "lower right",
        "lower left", "upper right", "upper left".
    bar_thickness : int, optional
        Thickness of the scale bar in pixels (default is 10).
    bar_padding : int, optional
        Padding between the scale bar and the image edge in pixels (default is 20).
    """
    figure, ax = plt.subplots(figsize=figsize, frameon=False)

    # Create a figure and axis for displaying the image
    ax.imshow(image, cmap="gray")
    ax.axis("off")  # Remove axes for a clean thumbnail

    if scalebar_size_um is not None and pixel_size is not None:
        add_scalebar(
            ax,
            scale_bar_um=scalebar_size_um,
            pixel_size=pixel_size,
            location=scalebar_location,
            bar_thickness=bar_thickness,
            padding=bar_padding,
        )
        image_name += f"_scalebar{scalebar_size_um}um"

    plt.show()
    save_plot_to_path(
        figure, output_path, image_name, dpi=dpi, file_format=file_format, pad_inches=0
    )
    plt.close(figure)
