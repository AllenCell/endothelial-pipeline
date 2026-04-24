"""Methods for constructing schematics for the flow field supplementary figure."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import FancyArrowPatch

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
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_XLARGE
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


def _data_to_fig(
    fig: plt.Figure, ax: plt.Axes, data_point: tuple[float, float], x_offset: float
) -> tuple[float, float]:
    """
    Convert data coordinates to figure coordinates.

    Parameters
    ----------
    fig
        The figure object.
    ax
        The axes object.
    data_point
        Tuple of (x, y) coordinates in data space.
    x_offset
        Optional, horizontal offset to apply in figure coordinate units.

    Returns
    -------
    :
        Tuple of (x, y) coordinates in figure space.
    """
    display = ax.transData.transform(data_point)
    fig_coords = fig.transFigure.inverted().transform(display)
    fig_coords[0] += x_offset
    return tuple(fig_coords)


def _add_map_arrow_to_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    box_position: tuple[float, float],
    rad: float,
    text: str,
    text_y_offset: float,
    arrow_x_offset: float = 0.05,
    linewidth: float = 1.5,
    arrowstyle: str = "->,head_length=5,head_width=3",
) -> None:
    """
    Add a curved arrow from a highlighted box to a text label, with the label
    positioned above the box and horizontally aligned with its midpoint.

    This method is used to add the arrow "mapping" the highlighted box in the
    image panel to the corresponding (theta, r, rho) label in the kernel
    convolution schematic.

    Parameters
    ----------
    fig
        The figure object.
    ax
        The axes object containing the box.
    box_position
        Tuple of (x, y) coordinates for the bottom center of the highlighted box
        in data space.
    rad
        The curvature radius for the arrow. Positive values curve to the right,
        negative values curve to the left.
    text
        The text that the arrow points to.
    text_y_offset
        Vertical offset for the text label from the top of the box in figure
        coordinate units.
    arrow_x_offset
        Horizontal offset for the start of the arrow from the box in figure
        coordinate units.
    linewidth
        Line width for the arrow.
    arrowstyle
        Arrow style string for the arrow.

    """
    bbox = ax.get_position()
    label_y = bbox.y0 + text_y_offset
    # Align labels horizontally with the midpoint of each highlighted box
    arrow_start = _data_to_fig(fig, ax, box_position, x_offset=arrow_x_offset)
    label_x = arrow_start[0]

    # Text labels
    fig.text(
        label_x,
        label_y,
        text,
        ha="center",
        va="top",
        fontsize=FONTSIZE_XLARGE,
    )

    arrow = FancyArrowPatch(
        arrow_start,
        (label_x, label_y - 0.01),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        color="black",
        linewidth=linewidth,
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(arrow)


def _add_t_plus_1_arrow_to_plot(
    fig: plt.Figure,
    ax_t: plt.Axes,
    ax_t1: plt.Axes,
    box_position: tuple[float, float],
    text_box_x_offset: float,
    arrow_y_position: float,
    arrow_text: str,
    arrow_x_offset: float = 0.07,
    linewidth: float = 1.5,
    arrowstyle: str = "->,head_length=5,head_width=3",
    delta_text_y_offset: float = 0.02,
) -> None:
    """Add a straight arrow between the (theta, r, rho) labels for t and t+1."""

    arrow_start_x = _data_to_fig(
        fig, ax_t, box_position, x_offset=text_box_x_offset + arrow_x_offset
    )[0]
    arrow_end_x = _data_to_fig(
        fig, ax_t1, box_position, x_offset=text_box_x_offset - arrow_x_offset - 0.0085
    )[0]
    arrow_mid_x = (arrow_start_x + arrow_end_x) / 2

    fig.text(
        arrow_mid_x,
        arrow_y_position + delta_text_y_offset,
        arrow_text,
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_LARGE,
    )
    horizontal_arrow = FancyArrowPatch(
        (arrow_start_x, arrow_y_position),
        (arrow_end_x, arrow_y_position),
        arrowstyle=arrowstyle,
        color="black",
        linewidth=linewidth,
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(horizontal_arrow)


def make_real_image_panel(
    savedir: Path,
    contact_figsize: tuple[float, float] = (5.0, 1.75),
    fov_crop_size: int = 2 * NATIVE_ZARR_RESOLUTION_CROP_SIZE,
    scale_bar_um: int = 20,
    grid_crop_position: tuple[int, int] = (0, 0),
    grid_crop_size: int = NATIVE_ZARR_RESOLUTION_CROP_SIZE,
    axes_title_xloc: float = 0.25,
    map_arrow_x_offset: float = 0.065,
    map_arrow_rad: float = 0.3,
    map_arrow_linewidth: float = 1.5,
    map_arrow_arrowstyle: str = "->,head_length=5,head_width=3",
    horizontal_arrow_x_offset: float = 0.07,
    horizontal_arrow_y_offset: float = -0.025,
    horizontal_arrow_linewidth: float = 1.5,
    horizontal_arrow_arrowstyle: str = "->,head_length=5,head_width=3",
    text_y_offset: float = -0.125,
    delta_text_y_offset: float = 0.02,
    layout_engine_kwargs: dict[str, Any] = {"rect": (0, 0.2, 1, 0.8)},
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
            fov_crop_size,
        )
        processed_images.append(log_bf_std_dev)

    fig: plt.Figure = make_contact_sheet(
        processed_images,
        max_cols=len(processed_images),
        max_rows=1,
        fig_kwargs={"figsize": contact_figsize, "layout": "constrained"},
    )

    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**layout_engine_kwargs)

    ax_t = fig.axes[0]
    ax_t1 = fig.axes[1]
    for ax, label in [
        (ax_t, "t"),
        (ax_t1, "t+1"),
    ]:
        ax.set_frame_on(False)
        ax.set_title(label, fontsize=FONTSIZE_LARGE, x=axes_title_xloc)

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
            grid_crop_size,
            grid_crop_size,
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

    # Curved arrows to (theta,r,rho) labels and straight arrow between them
    fig.canvas.draw()

    bbox_t = ax_t.get_position()
    box_mid_x = grid_crop_position[0] + NATIVE_ZARR_RESOLUTION_CROP_SIZE / 2
    box_bottom_y = grid_crop_position[1] + NATIVE_ZARR_RESOLUTION_CROP_SIZE

    for ax, label, arrow_rad in [
        (ax_t, "t", map_arrow_rad),
        (ax_t1, "t+1", -map_arrow_rad),
    ]:
        _add_map_arrow_to_plot(
            fig,
            ax,
            box_position=(box_mid_x, box_bottom_y),
            rad=arrow_rad,
            text=f"({Unicode.THETA}, r, {Unicode.RHO}) at {label}",
            text_y_offset=text_y_offset,
            arrow_x_offset=map_arrow_x_offset,
            linewidth=map_arrow_linewidth,
            arrowstyle=map_arrow_arrowstyle,
        )

    # Horizontal arrow between the two (theta, r, rho) labels
    mid_y = bbox_t.y0 + text_y_offset + horizontal_arrow_y_offset
    _add_t_plus_1_arrow_to_plot(
        fig,
        ax_t,
        ax_t1,
        box_position=(box_mid_x, box_bottom_y),
        text_box_x_offset=map_arrow_x_offset,
        arrow_y_position=mid_y,
        arrow_text=f"({Unicode.DELTA}{Unicode.THETA}, {Unicode.DELTA}r, {Unicode.DELTA}{Unicode.RHO})",
        arrow_x_offset=horizontal_arrow_x_offset,
        linewidth=horizontal_arrow_linewidth,
        arrowstyle=horizontal_arrow_arrowstyle,
        delta_text_y_offset=delta_text_y_offset,
    )

    filename = "flow_field_example_t_to_tp1"
    save_plot_to_path(
        fig, savedir, filename, file_format=".svg", transparent=True, tight_layout=False
    )
    image_panel_path = savedir / f"{filename}.svg"

    return image_panel_path
