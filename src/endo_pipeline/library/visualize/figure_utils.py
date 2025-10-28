import logging
from pathlib import Path
from typing import Literal

import matplotlib.axes as maxes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.settings.figures import FIGURE_SAVE_DPI, FONTSIZE_LARGE

logger = logging.getLogger(__name__)


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


def make_contact_sheet(
    panels: list[np.ndarray],
    max_rows: int | None = None,
    max_cols: int | None = None,
    horizontal_titles: list[str] | None = None,
    vertical_titles: list[str] | None = None,
    panel_titles: list[str] | None = None,
    direction: Literal["left-right first", "top-down first"] = "left-right first",
    subplot_kwargs: dict | None = None,
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
) -> plt.Figure:
    """Create and save a contact sheet of images.
    Sequentially plots images from "panels" in a grid layout with optional titles for rows and
    columns in the specified direction (row-by-row for "left-right first" or column-by-column
    for "top-down first"). If neither max_rows nor max_cols is specified, all panels will be
    plotted in a single row or column depending "direction".

    Parameters
    ----------
    panels:
        List of 2D arrays representing the images to be plotted in the contact sheet.
    max_rows:
        Maximum number of rows in the contact sheet. If None, no limit is applied.
    max_cols:
        Maximum number of columns in the contact sheet. If None, no limit is applied.
    horizontal_titles:
        List of titles for each column. Length of horizontal_titles must match the
        number of columns that are plotted or have length of 1 if provided. If the
        length is 1 then the title will be repeated for each column.
        If None, no titles are added.
    vertical_titles:
        List of titles for each row. Length of vertical_titles must match the number
        of rows that are plotted or have length of 1 if provided. If the length is 1
        then the title will be repeated for each row.
        If None, no titles are added.
    panel_titles:
        List of titles for each panel. Length must match the number of panels provided.
        If None, no titles are added.
    direction:
        Direction to fill the contact sheet: "left-right first" or "top-down first".
        If left-right first, panels are filled row-wise (a row fills up until max_cols
        is reached before starting a new row).
        If top-down first, panels are filled column-wise (a column fills up until max_rows
        is reached before starting a new column).
    subplot_kwargs:
        Additional keyword arguments to pass to plt.subplots for each subplot.
        Example includes 'frame_on' to remove the lines around each subplot.
    gridspec_kwargs:
        Additional keyword arguments to pass to plt.subplots for the gridspec.
        Example includes 'wspace' and 'hspace' to adjust spacing between subplots.
    fig_kwargs:
        Additional keyword arguments to pass to plt.subplots for the figure.
        Example includes 'figsize' to set the overall figure size.

    Returns
    -------
    plt.Figure
        The created contact sheet figure. Can be displayed again in an interactive session
        by running `fig`, and axes can be accessed and further modified via `fig.axes`
        (for example if adding lines, arrows, a scalebar, or additional plots is desired).
    """

    if direction not in ["left-right first", "top-down first"]:
        raise ValueError("Invalid direction specified.")
    if (panel_titles is not None) and (len(panels) != len(panel_titles)):
        raise ValueError("Number of panel_titles must match number of panels if provided.")

    # 'Figure' out the number of panels you have to plot
    num_panels = len(panels)

    if direction == "left-right first":
        max_panels_in_direction = max_cols or num_panels
    else:
        max_panels_in_direction = max_rows or num_panels

    # Get the number of panels orthogonal to the specified direction that
    # are needed based on the max panels along the specified direction
    if num_panels % max_panels_in_direction:
        # this is in case only a single row/column is needed
        n_panels_ortho_direction = num_panels // max_panels_in_direction + 1
    else:
        n_panels_ortho_direction = num_panels // max_panels_in_direction

    # truncate the number of rows/columns along the specified direction if
    # the number of panels in a direction is less than the max allowed
    if (n_panels_ortho_direction == 1) and (num_panels < max_panels_in_direction):
        n_panels_in_direction = num_panels
    else:
        n_panels_in_direction = max_panels_in_direction

    # convert the number of panels in the specified
    # and orthogonal directions to nrows and ncols
    if direction == "left-right first":
        nrows = n_panels_ortho_direction
        ncols = n_panels_in_direction
    else:  # top-down first
        nrows = n_panels_in_direction
        ncols = n_panels_ortho_direction

    # adjust the length of the horizontal and vertical titles lists
    # if necessary
    if horizontal_titles is not None:
        if len(horizontal_titles) == 1:
            horizontal_titles = horizontal_titles * ncols
        elif len(horizontal_titles) != ncols:
            logger.warning(
                f"""Length of horizontal_titles is not 1 nor equal to number of columns.
                    ncols={ncols}, number of horizontal titles={len(horizontal_titles)}"""
            )
        else:
            pass
    if vertical_titles is not None:
        if len(vertical_titles) == 1:
            vertical_titles = vertical_titles * nrows
        elif len(vertical_titles) != nrows:
            logger.warning(
                f"""Length of vertical_titles is not 1 nor equal to number of rows.
                    nrows={nrows}, number of vertical titles={len(vertical_titles)}"""
            )
        else:
            pass

    # create the figure and axes
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw=subplot_kwargs or {},
        gridspec_kw=gridspec_kwargs or {},
        **(fig_kwargs or {}),
    )

    ## iterate through your contact sheet axs to get the
    ## (row, column) index for each ax in axs as a flat list...
    ## ... accounting for if there is only 1 row
    if nrows == 1:
        ax_indices = [(1, c) for c in range(ncols)]
    ## ... accounting for if there is only 1 column
    elif nrows != 1 and ncols == 1:
        ax_indices = [(r, 1) for r in range(nrows)]
    ## ... all other situations (i.e. more than 1 row and column)
    else:
        if direction == "left-right first":
            ax_indices = [(r, c) for r in range(nrows) for c in range(ncols)]
        else:  # top-down first
            ax_indices = [(r, c) for c in range(ncols) for r in range(nrows)]

    ## start axes indices at 0 and put a plot in each ax
    for i in range(num_panels):
        ax_row, ax_col = ax_indices[i]
        ax = axs[(ax_row, ax_col)]

        ax.imshow(panels[i], cmap="gray")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(panel_titles[i], fontsize=FONTSIZE_LARGE) if panel_titles is not None else None

        if (ax_col == 0) and (vertical_titles is not None):
            ax.set_ylabel(vertical_titles[ax_row], fontsize=FONTSIZE_LARGE)
        if (ax_row == 0) and (horizontal_titles is not None):
            ax.set_xlabel(horizontal_titles[ax_col], fontsize=FONTSIZE_LARGE)
            ax.xaxis.set_label_position("top")

    return fig
