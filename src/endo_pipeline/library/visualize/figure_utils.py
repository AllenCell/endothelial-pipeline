import logging
from pathlib import Path
from typing import Literal

import matplotlib.axes as maxes
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.settings.figures import FIGURE_SAVE_DPI, FONTSIZE_LARGE, FONTSIZE_SMALL
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


def set_axes_properties(
    axes: plt.Axes | Axes3D,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    xticks: list[float] | list[int] | None = None,
    yticks: list[float] | list[int] | None = None,
    zticks: list[float] | list[int] | None = None,
    xtick_kwargs: dict | None = None,
    ytick_kwargs: dict | None = None,
    ztick_kwargs: dict | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    ztick_labels: list[str] | None = None,
    xtick_label_kwargs: dict | None = None,
    ytick_label_kwargs: dict | None = None,
    ztick_label_kwargs: dict | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
    zlabel_kwargs: dict | None = None,
    title: str | None = None,
    title_kwargs: dict | None = None,
    aspect: Literal["auto", "equal"] | float | None = None,
    facecolor: str | None = None,
) -> None:
    """
    Set properties of the given axis, including limits, ticks, labels, and
    title.

    Parameters
    ----------
    axes
        The axis to set properties for.
    xlim
        Optional, tuple specifying the limits for the x-axis (min, max).
    ylim
        Optional, tuple specifying the limits for the y-axis (min, max).
    zlim
        Optional, tuple specifying the limits for the z-axis (min, max).
    xticks
        Optional, list of tick positions for the x-axis.
    yticks
        Optional, list of tick positions for the y-axis.
    zticks
        Optional, list of tick positions for the z-axis.
    xtick_kwargs
        Optional, dictionary of keyword arguments to pass to set_xticks.
    ytick_kwargs
        Optional, dictionary of keyword arguments to pass to set_yticks.
    ztick_kwargs
        Optional, dictionary of keyword arguments to pass to set_zticks.
    xtick_labels
        Optional, list of tick labels for the x-axis.
    ytick_labels
        Optional, list of tick labels for the y-axis.
    ztick_labels
        Optional, list of tick labels for the z-axis.
    xtick_label_kwargs
        Optional, dictionary of keyword arguments to pass to set_xticklabels.
    ytick_label_kwargs
        Optional, dictionary of keyword arguments to pass to set_yticklabels.
    ztick_label_kwargs
        Optional, dictionary of keyword arguments to pass to set_zticklabels.
    xlabel
        Optional, label for the x-axis.
    ylabel
        Optional, label for the y-axis.
    zlabel
        Optional, label for the z-axis.
    xlabel_kwargs
        Optional, dictionary of keyword arguments to pass to set_xlabel.
    ylabel_kwargs
        Optional, dictionary of keyword arguments to pass to set_ylabel.
    zlabel_kwargs
        Optional, dictionary of keyword arguments to pass to set_zlabel.
    title
        Optional, title for the axis.
    title_kwargs
        Optional, dictionary of keyword arguments to pass to set_title.
    aspect
        Optional, aspect ratio for the axis.
    facecolor
        Optional, background color for the axis.

    """
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    if xticks is not None:
        axes.set_xticks(xticks, **(xtick_kwargs or {}))
    if yticks is not None:
        axes.set_yticks(yticks, **(ytick_kwargs or {}))
    if xtick_labels is not None:
        axes.set_xticklabels(xtick_labels, **(xtick_label_kwargs or {}))
    if ytick_labels is not None:
        axes.set_yticklabels(ytick_labels, **(ytick_label_kwargs or {}))
    if xlabel is not None:
        axes.set_xlabel(xlabel, **(xlabel_kwargs or {}))
    if ylabel is not None:
        axes.set_ylabel(ylabel, **(ylabel_kwargs or {}))
    if title is not None:
        axes.set_title(title, **(title_kwargs or {}))
    if aspect is not None:
        axes.set_aspect(aspect)
    if facecolor is not None:
        axes.set_facecolor(facecolor)
    if isinstance(axes, Axes3D):
        if zlim is not None:
            axes.set_zlim(zlim)
        if zticks is not None:
            axes.set_zticks(zticks, **(ztick_kwargs or {}))
        if ztick_labels is not None:
            axes.set_zticklabels(ztick_labels, **(ztick_label_kwargs or {}))
        if zlabel is not None:
            axes.set_zlabel(zlabel, **(zlabel_kwargs or {}))


def add_scalebar(
    ax: maxes.Axes,
    scale_bar_um: float,
    pixel_size: float,
    location: str = "lower left",
    bar_thickness: float = 10,
    padding: float = 20,
    color: str = "white",
    include_label: bool = False,
    label_xy: tuple[float, float] = (0.96, 0.08),
    label_fontsize: int = FONTSIZE_SMALL,
) -> None:
    """
    Add a scale bar to an image displayed with imshow (no text label).

    Parameters
    ----------
    ax
        The axis to add the scale bar to.
    scale_bar_um
        Length of the scale bar in micrometers.
    pixel_size
        Size of a pixel in micrometers.
    location
        One of 'upper left', 'upper right', 'lower left', 'lower right'.
    bar_thickness
        Thickness of the scale bar in pixels.
    padding
        Padding from the edge of the image in pixels.
    color
        Color of the scale bar.
    include_label
        If True, adds a text label above the scale bar indicating the length in micrometers.
    label_xy
        Position (x, y) of the label in axis coordinates (0 to 1). Only used if include_label is True.
    label_fontsize
        Font size for the label text. Only used if include_label is True.

    """
    scale_bar_px = scale_bar_um / pixel_size
    length_px = scale_bar_px

    axes_values = ax.images[0].get_array()
    assert axes_values is not None
    ny, nx = axes_values.shape[:2]  # supports both grayscale (H, W) and RGB(A) (H, W, C)

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

    if include_label:
        glow_color = "black" if color == "white" else "white"
        ax.text(
            label_xy[0],
            label_xy[1],
            f"{scale_bar_um} {Unicode.MU}m",
            color=color,
            fontsize=label_fontsize,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground=glow_color),
            ],
        )


def plot_image_thumbnail(
    image: np.ndarray,
    image_name: str,
    output_path: Path | None,
    figsize: tuple[float, float],
    dpi: int = FIGURE_SAVE_DPI,
    file_format: Literal[".png", ".pdf", ".svg"] = ".png",
    scalebar_size_um: float | None = None,
    pixel_size: float | None = None,
    scalebar_location: Literal[
        "lower right", "lower left", "upper right", "upper left"
    ] = "lower left",
    bar_thickness: int = 10,
    bar_padding: int = 20,
    show_plot: bool = True,
    outline_color: str | None = None,
    image_colormap: str | Colormap | None = "gray",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Save a thumbnail image to a specified file path.

    This function saves a given image as a thumbnail in the specified format
    with the desired resolution and figure size.

    Parameters
    ----------
    image : numpy.ndarray
        The image to save, represented as a NumPy array.
    image_name : str
        The name of the output image file (without extension).
    output_path : Path | None
        The directory where the image will be saved. If None, the image is not saved.
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
    outline_color: str, optional
        Color of the outline around the image. If None, no outline is drawn.
    image_colormap: str, optional
        Colormap to use when displaying the image. Default is "gray".
    """
    figure = plt.figure(figsize=figsize, frameon=False)
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)

    ax.imshow(image, cmap=image_colormap)
    ax.axis("off")

    if outline_color is not None:
        height, width = image.shape[:2]
        rect = patches.Rectangle(
            (0, 0), width, height, linewidth=0.25, edgecolor=outline_color, facecolor="none"
        )
        ax.add_patch(rect)

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

    if show_plot:
        plt.show()

    if output_path is None:
        return figure, ax

    save_plot_to_path(
        figure,
        output_path,
        image_name,
        dpi=dpi,
        file_format=file_format,
        pad_inches=0,
        tight_layout=True,
    )
    return figure, ax


def add_timestamp(
    ax,
    frame: int,
    interval_minutes: int,
    fontsize: int = FONTSIZE_LARGE,
    shear_stress: float | None = None,
) -> matplotlib.text.Text:
    """
    Add a timestamp to the given axis based on frame number and interval (hr:min).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the timestamp to.
    frame : int
        The current frame number
    interval_minutes : int, optional
        Time interval between frames in minutes.
    fontsize : int
        Font size for the timestamp text.
    shear_stress : float, optional
        Shear stress value to display alongside the timestamp (in dyn/cm²).
    """
    duration_minutes = frame * interval_minutes
    hours = int(duration_minutes // 60)
    minutes = int(duration_minutes % 60)

    shear_stress_label = f"{shear_stress:.1f} dyn/cm²" if shear_stress is not None else ""
    timestamp = f"{hours:02d}:{minutes:02d} hr:min {shear_stress_label}"

    return ax.text(
        0.01,
        0.98,
        timestamp,
        transform=ax.transAxes,
        color="white",
        fontsize=fontsize,
        ha="left",
        va="top",
    )


def broadcast_title_list(title_list: list[str] | None, target_length: int) -> list[str] | None:
    """Broadcast a list of titles to a target length.

    Parameters
    ----------
    title_list:
        List of titles to broadcast. If None, returns None.
    target_length:
        Target length to broadcast the titles to.
    """
    if title_list is not None:
        if len(title_list) == 1:
            title_list = title_list * target_length
        elif len(title_list) != target_length:
            logger.warning(
                f"""Number of titles is not 1 nor equal to target_length.
                    target_length={target_length}, number of titles={len(title_list)}"""
            )
        else:
            pass
    return title_list


def reshape_panel_list_from_direction(
    num_panels: int,
    max_panels_per_line: int | None,
    direction: Literal["left-right first", "top-down first"],
) -> tuple[int, int]:
    """Reshape the list of panels based on the specified direction.

    Parameters
    ----------
    num_panels:
        Total number of panels to be plotted.
    max_panels_per_line:
        Maximum number of panels allowed in the specified direction.
    direction:
        Direction to fill the contact sheet: "left-right first" or "top-down first".

    Returns
    -------
    (nrows, ncols):
        Tuple with number of rows and columns for the contact sheet.
    """

    max_panels_in_direction = max_panels_per_line or num_panels

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

    return nrows, ncols


def make_contact_sheet(
    panels: list[np.ndarray],
    max_rows: int,
    max_cols: int,
    col_titles: list[str] | None = None,
    row_titles: list[str] | None = None,
    panel_titles: list[str] | None = None,
    direction: Literal["left-right first", "top-down first"] = "left-right first",
    font_size: int = FONTSIZE_LARGE,
    subplot_kwargs: dict | None = None,
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
    use_constrained_layout: bool = False,
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
        Maximum number of rows in the contact sheet.
    max_cols:
        Maximum number of columns in the contact sheet.
    col_titles:
        List of titles for each column. Length of col_titles must match the
        number of columns that are plotted or have length of 1 if provided. If the
        length is 1 then the title will be repeated for each column.
        If None, no titles are added.
    row_titles:
        List of titles for each row. Length of row_titles must match the number
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
    font_size:
        Font size for titles and labels.
    subplot_kwargs:
        Additional keyword arguments to pass to plt.subplots for each subplot.
        Example includes 'frame_on' to remove the lines around each subplot.
    gridspec_kwargs:
        Additional keyword arguments to pass to plt.subplots for the gridspec.
        Example includes 'wspace' and 'hspace' to adjust spacing between subplots.
    fig_kwargs:
        Additional keyword arguments to pass to plt.subplots for the figure.
        Example includes 'figsize' to set the overall figure size.
    use_constrained_layout:
        If True, create the figure with matplotlib's "constrained" layout engine, which
        fits subplots and decorations without modifying the requested figsize. When
        enabled, any 'wspace' / 'hspace' entries in ``gridspec_kwargs`` are dropped
        (constrained layout overrides them and emits a UserWarning otherwise); use
        ``fig.get_layout_engine().set(w_pad=..., h_pad=...)`` on the returned figure
        to tune spacing instead. Callers using constrained layout should also pass
        ``tight_layout=False`` to ``save_plot_to_path`` to avoid mixing layout engines.

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

    # get the number of rows and columns for panels/subplots based on the
    # provided number of panels, max rows, max columns, and direction
    max_panels_in_direction = max_cols if direction == "left-right first" else max_rows
    nrows, ncols = reshape_panel_list_from_direction(
        num_panels=num_panels, max_panels_per_line=max_panels_in_direction, direction=direction
    )

    # adjust the length of the column and row titles lists if necessary
    col_titles = broadcast_title_list(col_titles, ncols)
    row_titles = broadcast_title_list(row_titles, nrows)

    # create the figure and axes
    effective_gridspec_kwargs = dict(gridspec_kwargs or {})
    effective_fig_kwargs = dict(fig_kwargs or {})
    if use_constrained_layout:
        # constrained layout overrides explicit wspace/hspace and emits a
        # UserWarning if they are present; drop them defensively.
        effective_gridspec_kwargs.pop("wspace", None)
        effective_gridspec_kwargs.pop("hspace", None)
        # do not clobber an explicit caller-provided layout choice
        effective_fig_kwargs.setdefault("layout", "constrained")

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw=subplot_kwargs or {},
        gridspec_kw=effective_gridspec_kwargs,
        **effective_fig_kwargs,
    )

    ## iterate through your contact sheet axs to get the
    ## (row, column) index for each ax in axs as a flat list...
    ## ... accounting for if there is only 1 row
    if nrows == 1:
        ax_indices = [(0, c) for c in range(ncols)]
        # ensure axs is 2D array increasing column-wise for indexing
        axs = np.array(axs, ndmin=2)

    ## ... accounting for if there is only 1 column
    elif ncols == 1:
        ax_indices = [(r, 0) for r in range(nrows)]
        # ensure axs is 2D array increasing row-wise for indexing
        axs = np.array(axs, ndmin=2).T

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
        if panel_titles is not None:
            ax.set_title(panel_titles[i], fontsize=font_size)

        if (ax_col == 0) and (row_titles is not None):
            ax.set_ylabel(row_titles[ax_row], fontsize=font_size)
        if (ax_row == 0) and (col_titles is not None):
            ax.set_xlabel(col_titles[ax_col], fontsize=font_size)
            ax.xaxis.set_label_position("top")

    # remove any unused axes
    for i in range(num_panels, nrows * ncols):
        ax_row, ax_col = ax_indices[i]
        fig.delaxes(axs[(ax_row, ax_col)])

    return fig
