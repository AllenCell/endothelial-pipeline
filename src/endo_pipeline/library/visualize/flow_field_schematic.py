"""Methods for constructing schematics for the flow field supplementary figure."""

from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import join_sorted_strings, load_dataframe, load_image, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    _check_and_adjust_km_inputs,
    _evaluate_multivariate_product_kernel,
    _get_weighted_histogram_for_convolution,
    get_cartesian_product,
    get_kramers_moyal_coeffs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    crop_image,
    log_normalize_image,
    std_dev,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
)
from endo_pipeline.settings.autocorrelations import AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
    POLAR_ANGLE_PERIOD,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.examples import EXAMPLE_DATASET, FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import (
    FONTSIZE_LARGE,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
)
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_figure import (
    SUPP_FIG_TARGET_POINT,
    SUPP_FIG_ZOOM_FACTOR,
    XLABEL_KWARGS,
    YLABEL_KWARGS,
)
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    RANDOM_SEED,
)


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
    text_y_position: float,
    text_x_offset: float = 0.0,
    arrow_x_offset: float = 0.05,
    linewidth: float = 1.5,
    color: str = "black",
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
    text_y_position
        The y-coordinate for the text label in figure coordinate units.
    text_x_offset
        Horizontal offset for the text label from the arrow in figure coordinate
        units.
    arrow_x_offset
        Horizontal offset for the start of the arrow from the box in figure
        coordinate units.
    linewidth
        Line width for the arrow.
    color
        Color for the arrow.
    arrowstyle
        Arrow style string for the arrow.

    """
    # Align labels horizontally with the midpoint of each highlighted box
    arrow_start = _data_to_fig(fig, ax, box_position, x_offset=arrow_x_offset)
    text_x_position = arrow_start[0] + text_x_offset

    # Text labels
    fig.text(
        text_x_position,
        text_y_position,
        text,
        ha="center",
        va="top",
        fontsize=FONTSIZE_LARGE,
    )

    arrow = FancyArrowPatch(
        arrow_start,
        (text_x_position, text_y_position - 0.01),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        color=color,
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
    delta_text_x_offset: float = -0.01,
    delta_text_y_offset: float = 0.03,
) -> None:
    """Add a straight arrow between the (theta, r, rho) labels for t and t+1."""

    arrow_start_x = _data_to_fig(
        fig, ax_t, box_position, x_offset=text_box_x_offset + arrow_x_offset
    )[0]
    arrow_end_x = _data_to_fig(
        fig, ax_t1, box_position, x_offset=text_box_x_offset - arrow_x_offset - 0.02
    )[0]
    arrow_mid_x = (arrow_start_x + arrow_end_x) / 2

    fig.text(
        arrow_mid_x + delta_text_x_offset,
        arrow_y_position + delta_text_y_offset,
        arrow_text,
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_SMALL,
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
    axes_title_xloc: float = 0.25,
    map_arrow_x_offset: float = 0.065,
    map_arrow_rad: float = 0.3,
    map_arrow_linewidth: float = 1.5,
    horizontal_arrow_x_offset: float = 0.125,
    horizontal_arrow_y_offset: float = -0.025,
    horizontal_arrow_linewidth: float = 1.5,
    text_y_offset: float = -0.175,
    delta_text_y_offset: float = 0.025,
) -> Path:
    """Build the panel showing a grid crop from t to t+1 for a given example image."""

    contact_figsize = (2.5, 1.7)
    arrowstyle = "->,head_length=5,head_width=3"
    box_color = "deepskyblue"
    map_arrow_color = "deepskyblue"

    fov_crop_size = 2 * NATIVE_ZARR_RESOLUTION_CROP_SIZE
    scale_bar_um = 20
    grid_crop_position = (0, 0)
    grid_crop_size = NATIVE_ZARR_RESOLUTION_CROP_SIZE

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
        gridspec_kwargs={"hspace": 0.05},
    )

    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**{"rect": (0, 0.2, 1, 0.8)})

    ax_t = fig.axes[0]
    ax_t1 = fig.axes[1]
    for ax, label, include_label in [
        (ax_t, "t", False),
        (ax_t1, f"t+{Unicode.DELTA}t", True),
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
            include_label=include_label,
            label_xy=(0.95, 0.09),
            label_fontsize=FONTSIZE_MEDIUM,
        )

        # add highlighted box to show crop region used for flow field construction
        rect = plt.Rectangle(
            grid_crop_position,
            grid_crop_size,
            grid_crop_size,
            edgecolor=box_color,
            facecolor="none",
            linewidth=2,
            clip_on=False,
        )
        ax.add_patch(rect)

    # Curved arrows to (theta,r,rho) labels and straight arrow between them
    fig.canvas.draw()

    bbox_t = ax_t.get_position()
    box_mid_x = grid_crop_position[0] + NATIVE_ZARR_RESOLUTION_CROP_SIZE / 2
    box_bottom_y = grid_crop_position[1] + NATIVE_ZARR_RESOLUTION_CROP_SIZE

    for ax, label, text_x_offset, arrow_rad in [
        (ax_t, "t", -0.025, map_arrow_rad),
        (ax_t1, f"t+{Unicode.DELTA}t", 0.025, -map_arrow_rad),
    ]:
        _add_map_arrow_to_plot(
            fig,
            ax,
            box_position=(box_mid_x, box_bottom_y),
            rad=arrow_rad,
            text=f"({Unicode.THETA}, r, {Unicode.RHO}) at {label}",
            text_y_position=bbox_t.y0 + text_y_offset,
            text_x_offset=text_x_offset,
            arrow_x_offset=map_arrow_x_offset,
            linewidth=map_arrow_linewidth,
            color=map_arrow_color,
            arrowstyle=arrowstyle,
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
        arrowstyle=arrowstyle,
        delta_text_y_offset=delta_text_y_offset,
    )

    filename = "flow_field_example_t_to_tp1"
    save_plot_to_path(
        fig, savedir, filename, file_format=".svg", transparent=True, tight_layout=False
    )
    image_panel_path = savedir / f"{filename}.svg"

    return image_panel_path


def _compute_r_squared(data: np.ndarray, fit: np.ndarray) -> float:
    """Compute the R^2 value for a given fit to data."""
    sum_sq_res = np.sum((data - fit) ** 2)
    sum_sq_tot = np.sum((data - np.mean(data)) ** 2)
    r_squared = 1 - sum_sq_res / sum_sq_tot if sum_sq_tot > 0 else np.nan
    return r_squared


def _make_example_acf_plot(
    axes: plt.Axes,
    dataset_name: str,
    column_name: Column.DiffAEData,
    autocorrelation_manifest: DataframeManifest,
    y_labelpad: float = 0.5,
) -> None:
    """
    Make an example plot of the autocorrelation function with exponential fit
    for a given dataset and feature column.
    """
    acf_df = load_dataframe(autocorrelation_manifest.locations[dataset_name])

    # Keep only positive lags for plotting (ACF is symmetric around zero).
    acf_df_positive = acf_df[acf_df[Column.AutoCorrelation.LAG] > 0]

    # plot acf for polar radius
    column_label = COLUMN_METADATA[column_name].label
    acf_polar_r = acf_df_positive[acf_df_positive[Column.AutoCorrelation.FEATURE] == column_name]
    lag_hours = acf_polar_r[Column.AutoCorrelation.LAG].to_numpy() * TIME_STEP_IN_HOURS
    acf_mean = acf_polar_r[Column.AutoCorrelation.ACF_MEAN].to_numpy()
    acf_lb = acf_polar_r[Column.AutoCorrelation.ACF_LOWER_PERCENTILE].to_numpy()
    acf_ub = acf_polar_r[Column.AutoCorrelation.ACF_UPPER_PERCENTILE].to_numpy()

    # get exponential fit of acf_mean and compute R^2
    exponential_decay_curve = acf_polar_r[Column.AutoCorrelation.EXPONENTIAL_FIT].to_numpy()
    r_squared = _compute_r_squared(acf_mean, exponential_decay_curve)

    axes.plot(lag_hours, acf_mean, "k-", label="ACF (mean)")
    axes.fill_between(lag_hours, acf_lb, acf_ub, color="gray", alpha=0.2, label="ACF (P5-P95)")
    axes.plot(
        lag_hours,
        exponential_decay_curve,
        color="darkturquoise",
        linestyle="--",
        linewidth=1.25,
        label=f"Exp. fit\n(R{Unicode.SQUARED} = {r_squared:.2f})",
    )
    axes.set_xlabel("Lag (hours)", fontsize=FONTSIZE_SMALL, **XLABEL_KWARGS)
    axes.set_ylabel("ACF", fontsize=FONTSIZE_SMALL, labelpad=y_labelpad)
    axes.set_title(f"Autocorrelation\nfunction (ACF) in {column_label}", fontsize=FONTSIZE_SMALL)
    axes.legend(loc="upper right", fontsize=FONTSIZE_XSMALL)
    axes.set_box_aspect(1)


def _make_acf_r_squared_plot(
    axes: plt.Axes,
    column_names: list[Column.DiffAEData],
    autocorrelation_manifest: DataframeManifest,
    y_labelpad: float = 0.5,
    jitter_random_seed: int = RANDOM_SEED,
    jitter_radius: float = 0.1,
    axes_ylim: tuple[float, float] = (0.975, 1.005),
    scatter_kwargs: dict | None = None,
    include_ylabel: bool = False,
) -> None:
    """
    Plot the R^2 values for the exponential fits to the ACFs for all datasets
    and specified columns as a scatter plot with jitter on the column axis.
    """
    rng = np.random.default_rng(jitter_random_seed)
    r_squared_dict: dict[Column.DiffAEData, list[float]] = {key: [] for key in column_names}

    # loop over datasets and compute R^2 values for each specified column,
    # storing in r_squared_dict
    dataset_names = get_datasets_in_collection("timelapse")
    for dataset_name in dataset_names:
        # temporary until flow switch deprecation goes through
        if dataset_name not in autocorrelation_manifest.locations:
            continue
        acf_dataset = load_dataframe(autocorrelation_manifest.locations[dataset_name])
        # make sure FEATURE entries are sorted in the same order as column_names
        acf_dataset_sorted = acf_dataset.copy()
        acf_dataset_sorted[Column.AutoCorrelation.FEATURE] = pd.Categorical(
            acf_dataset_sorted[Column.AutoCorrelation.FEATURE],
            categories=column_names,
            ordered=True,
        )
        acf_dataset_sorted = acf_dataset_sorted.sort_values(
            by=[Column.AutoCorrelation.FEATURE, Column.AutoCorrelation.LAG]
        )
        for column_name_, acf_column in acf_dataset_sorted.groupby(
            Column.AutoCorrelation.FEATURE, observed=True
        ):
            column_name = cast(Column.DiffAEData, column_name_)
            acf_mean = acf_column[Column.AutoCorrelation.ACF_MEAN].to_numpy()
            exponential_decay_curve = acf_column[Column.AutoCorrelation.EXPONENTIAL_FIT].to_numpy()
            # get r_squared for exponential fit
            r_squared_dict[column_name].append(
                _compute_r_squared(acf_mean, exponential_decay_curve)
            )

    # plot R^2 values with jitter on x axis for better visualization of
    # overlapping points
    for i, column_name in enumerate(column_names):
        r2_values = r_squared_dict[column_name]
        jitter = rng.uniform(-jitter_radius, jitter_radius, size=len(r2_values))
        axes.scatter(
            i + jitter,
            r2_values,
            **(scatter_kwargs or {}),
        )
    axes.set_xticks(range(len(column_names)))
    axes.set_xticklabels(
        [COLUMN_METADATA[col].label or col for col in column_names], fontweight="bold"
    )
    if include_ylabel:
        axes.set_ylabel(
            f"R{Unicode.SQUARED}", fontsize=FONTSIZE_SMALL, labelpad=y_labelpad, rotation=0
        )
    axes.set_title(f"Exponential fit R{Unicode.SQUARED}\n(all datasets)", fontsize=FONTSIZE_SMALL)
    axes.set_ylim(axes_ylim)
    axes.set_box_aspect(1)


def make_autocorrelation_panel(save_dir: Path) -> Path:
    # example dataset for first subplot
    dataset_name = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]

    column_names = [
        Column.DiffAEData.POLAR_ANGLE,
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
    ]

    # load dataframe manifest for outputs of autocorrelation analysis workflow
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    columns_str = join_sorted_strings(cast(list[str], column_names))
    autocorrelation_manifest_name = (
        f"{AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX}_{columns_str}_{base_name}"
    )
    autocorrelation_manifest = load_dataframe_manifest(autocorrelation_manifest_name)

    # init figure for autocorrelation panel
    fig, ax = plt.subplots(1, 2, figsize=(3.75, 1.9), layout="constrained")

    # Suplot 1: example ACF plot for one dataset and one feature to illustrate
    # the exponential fit and R^2 value
    _make_example_acf_plot(
        ax[0],
        dataset_name,
        Column.DiffAEData.POLAR_RADIUS,
        autocorrelation_manifest,
        y_labelpad=0.5,
    )

    # Suplot 2: R^2 values for exponential fits to ACFs for all datasets and
    # specified features as a scatter plot
    _make_acf_r_squared_plot(
        ax[1],
        column_names,
        autocorrelation_manifest,
        y_labelpad=4,
        jitter_radius=0.15,
        scatter_kwargs={"color": "black", "alpha": 0.75, "linewidths": 0, "s": 3, "zorder": 3},
    )

    filename = "autocorrelation_analysis"
    save_plot_to_path(
        fig, save_dir, filename, file_format=".svg", transparent=True, tight_layout=False
    )
    return save_dir / f"{filename}.svg"


def _make_2d_pcolormesh(
    axes: plt.Axes,
    data_2d: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cmap: str = DRIFT_CONTOUR_COLORMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
    axes_xlim: tuple[float, float] | None = None,
    axes_ylim: tuple[float, float] | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_title: str | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
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
        axes.set_xlabel(axes_xlabel, **(xlabel_kwargs or {}))
    if axes_ylabel is not None:
        axes.set_ylabel(axes_ylabel, **(ylabel_kwargs or {}))
    if axes_xlim is not None:
        axes.set_xlim(axes_xlim)
    if axes_ylim is not None:
        axes.set_ylim(axes_ylim)
    if axes_aspect is not None:
        axes.set_aspect(axes_aspect)
    if axes_title is not None:
        axes.set_title(axes_title)
    return pcm


def _get_target_bin(
    target_point: tuple[float, float],
    bin_edges: list[np.ndarray],
) -> tuple[int, int]:
    """Return the (ix, iy) indices of the bin cell that contains target_point."""
    ix = int(
        np.clip(
            np.searchsorted(bin_edges[0], target_point[0], side="right") - 1,
            0,
            len(bin_edges[0]) - 2,
        )
    )
    iy = int(
        np.clip(
            np.searchsorted(bin_edges[1], target_point[1], side="right") - 1,
            0,
            len(bin_edges[1]) - 2,
        )
    )
    return ix, iy


def _add_target_bin_border(
    ax: plt.Axes,
    target_bin: tuple[int, int],
    bin_edges: list[np.ndarray],
    color: str = "magenta",
    linewidth: float = 1.5,
    label: str | None = "target bin",
) -> None:
    """Draw a square border around the target bin."""
    ix, iy = target_bin
    x_left = bin_edges[0][ix]
    y_bottom = bin_edges[1][iy]
    bin_width_x = bin_edges[0][ix + 1] - bin_edges[0][ix]
    bin_width_y = bin_edges[1][iy + 1] - bin_edges[1][iy]
    rect = Rectangle(
        (x_left, y_bottom),
        bin_width_x,
        bin_width_y,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label=label,
        zorder=5,
    )
    ax.add_patch(rect)


def _add_colorbar_for_quadmesh(
    fig: plt.Figure,
    axes: plt.Axes,
    quadmesh: QuadMesh,
    cax_position: str = "top",
    pad: float = 0.03,
    orientation: str = "horizontal",
    colorbar_label: str | None = None,
    colorbar_title_kwargs: dict | None = {"fontsize": FONTSIZE_MEDIUM, "pad": 2},
) -> None:
    """Add a colorbar for a given QuadMesh plot."""
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(cax_position, size="5%", pad=pad)

    # use title instead of label if orientation is horizontal to avoid issues
    # with label positioning on top of the colorbar
    label = None if orientation == "horizontal" else colorbar_label
    fig.colorbar(quadmesh, cax=cax, label=label, orientation=orientation)

    tick_axis = cax.xaxis if orientation == "horizontal" else cax.yaxis
    tick_axis.set_ticks_position(cax_position)
    tick_axis.set_label_position(cax_position)

    if colorbar_label is not None and orientation == "horizontal":
        cax.set_title(colorbar_label, **(colorbar_title_kwargs or {}))


def _make_weighted_displacement_histogram(
    fig: plt.Figure,
    axes: plt.Axes,
    dataframe_steady_state: pd.DataFrame,
    column_names: list[Column.DiffAEData],
    bin_edges: list[np.ndarray],
    target_bin: tuple[int, int],
    axes_xlim: tuple[float, float] | None = None,
    axes_ylim: tuple[float, float] | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_title: str | None = None,
    cmap: str = "RdBu_r",
    colorbar_label: str | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
) -> np.ndarray:
    """Compute and plot a 2D histogram of the data weighted by the sum of the displacements in each bin."""

    # Get trajectories and displacements for the specified columns. The
    # trajectory values are used for binning, and the displacements are used for
    # weighting the histogram counts.
    traj_list, disp_list = get_traj_and_diff(dataframe_steady_state, column_names)

    # we only need density and drift in the first input column for this
    # schematic, so the powers are set to select the appropriate displacements
    # for weighting the histogram counts
    powers = np.array([[0, 0], [1, 0]])
    traj_list_, disp_list_, powers_ = _check_and_adjust_km_inputs(traj_list, disp_list, powers)

    # Weighted_hist shape: (2, n_bins_x, n_bins_y), where the second entry along
    # the first axis corresponds to the x-displacement-weighted counts
    # (powers=[1,0])
    weighted_hist = _get_weighted_histogram_for_convolution(
        traj_list_, disp_list_, bin_edges, powers_
    )
    weighted_counts_delta_x = weighted_hist[1]

    # Set vmin and vmax for the colormap based on the 99th percentile of the
    # absolute values in the weighted histogram, to avoid outliers dominating
    # the color scale.
    vmax = np.nanpercentile(np.abs(weighted_counts_delta_x), 99)
    pcm = _make_2d_pcolormesh(
        axes,
        weighted_counts_delta_x,
        bin_edges[0],
        bin_edges[1],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        axes_xlabel=axes_xlabel,
        axes_ylabel=axes_ylabel,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_aspect=axes_aspect,
        axes_title=axes_title,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )
    _add_target_bin_border(
        axes,
        target_bin=target_bin,
        bin_edges=bin_edges,
    )
    _add_colorbar_for_quadmesh(
        fig,
        axes,
        pcm,
        colorbar_label=colorbar_label,
    )
    return weighted_counts_delta_x


def _plot_kernel_at_target_bin(
    fig: plt.Figure,
    axes: plt.Axes,
    kernels: list[KramersMoyalKernel],
    bin_edges: list[np.ndarray],
    bin_centers: list[np.ndarray],
    target_bin: tuple[int, int],
    axes_xlim: tuple[float, float] | None = None,
    axes_ylim: tuple[float, float] | None = None,
    axes_xlabel: str | None = None,
    axes_ylabel: str | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_title: str | None = None,
    cmap: str = "Purples",
    colorbar_label: str | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
) -> np.ndarray:
    """Plot the 2D product kernel weights centered at the target bin."""
    # evaluate 2D product kernel weights centered at the target bin
    ix, iy = target_bin
    target_bin_center = (
        (bin_edges[0][ix] + bin_edges[0][ix + 1]) / 2,
        (bin_edges[1][iy] + bin_edges[1][iy + 1]) / 2,
    )
    target_offsets = [
        bin_centers[0] - target_bin_center[0],
        bin_centers[1] - target_bin_center[1],
    ]
    offsets_grid = get_cartesian_product(target_offsets)
    kernel_weights_2d = _evaluate_multivariate_product_kernel(offsets_grid, kernels)
    pcm = _make_2d_pcolormesh(
        axes,
        kernel_weights_2d / kernel_weights_2d.max(),
        bin_edges[0],
        bin_edges[1],
        cmap=cmap,
        axes_xlabel=axes_xlabel,
        axes_ylabel=axes_ylabel,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_aspect=axes_aspect,
        axes_title=axes_title,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )
    _add_target_bin_border(
        axes,
        target_bin=target_bin,
        bin_edges=bin_edges,
    )
    _add_colorbar_for_quadmesh(
        fig,
        axes,
        pcm,
        colorbar_label=colorbar_label,
    )
    return kernel_weights_2d


def make_kernel_convolution_schematic(savedir: Path) -> Path:
    """
    Build the panel showing a schematic of the kernel convolution process for
    a single target bin in (r, rho) space.
    """
    dataset_name = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]

    column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]

    target_point = SUPP_FIG_TARGET_POINT
    r_half = SUPP_FIG_ZOOM_FACTOR * KERNEL_BANDWIDTHS_DYNAMICS[column_names[0]]
    rho_half = SUPP_FIG_ZOOM_FACTOR * KERNEL_BANDWIDTHS_DYNAMICS[column_names[1]]
    axes_xlim = (target_point[0] - r_half, target_point[0] + r_half)
    axes_ylim = (target_point[1] - rho_half, target_point[1] + rho_half)

    # Load feature data for provided dataset and filter to steady-state time
    # points and a single flow condition
    dataset_config = load_dataset_config(dataset_name)

    feature_manifest_name = GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME
    feature_manifest = load_dataframe_manifest(feature_manifest_name)

    dataset_location = get_dataframe_location_for_dataset(feature_manifest, dataset_name)

    columns_to_load = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]

    df_raw = load_dataframe(dataset_location, delay=True)
    df: pd.DataFrame = df_raw[columns_to_load].compute()
    dataframe_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    bin_widths = tuple(BIN_WIDTHS_DYNAMICS[col] for col in column_names)
    bin_edges, bin_centers = get_bins(bin_widths, df[column_names].to_numpy())
    target_bin = _get_target_bin(target_point, bin_edges)

    fig, ax = plt.subplots(
        1, 4, gridspec_kw={"wspace": 0.075}, figsize=(6.0, 1.65), layout="constrained"
    )
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    axes_xlabel = COLUMN_METADATA[column_names[0]].label
    axes_ylabel = COLUMN_METADATA[column_names[1]].label

    # panel 1 - r-displacement-weighted 2D histogram
    weighted_hist_delta_r = _make_weighted_displacement_histogram(
        fig=fig,
        axes=axes[0],
        dataframe_steady_state=dataframe_steady_state,
        column_names=column_names,
        bin_edges=bin_edges,
        target_bin=target_bin,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_xlabel=axes_xlabel,
        axes_ylabel=axes_ylabel,
        ylabel_kwargs=YLABEL_KWARGS,
        xlabel_kwargs=XLABEL_KWARGS,
        colorbar_label=f"Sum of {Unicode.DELTA} {axes_xlabel}",
    )

    # panel 2 - kernel weights centered at target bin
    kernels = [
        KramersMoyalKernel(
            name=KERNEL_NAMES_DYNAMICS[column_name],
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
            period=POLAR_ANGLE_PERIOD if column_name == Column.DiffAEData.POLAR_ANGLE else None,
        )
        for column_name in column_names
    ]
    kernel_weights_2d = _plot_kernel_at_target_bin(
        fig=fig,
        axes=axes[1],
        kernels=kernels,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        target_bin=target_bin,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_xlabel=axes_xlabel,
        xlabel_kwargs=XLABEL_KWARGS,
        colorbar_label="Kernel weight\n(normalized)",
    )

    # panel 3 - kernel-weighted histogram (i.e. numerator of KM estimator)
    kernel_weighted_hist_delta_r = kernel_weights_2d * weighted_hist_delta_r
    # use same vmin and vmax for colormap as panel 1 to allow direct comparison
    # of the effect of the kernel weighting on the histogram values
    vmax = np.nanpercentile(np.abs(weighted_hist_delta_r), 99)
    pcm = _make_2d_pcolormesh(
        axes[2],
        kernel_weighted_hist_delta_r,
        bin_edges[0],
        bin_edges[1],
        vmin=-vmax,
        vmax=vmax,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_xlabel=axes_xlabel,
        xlabel_kwargs=XLABEL_KWARGS,
    )
    _add_target_bin_border(
        axes[2],
        target_bin=target_bin,
        bin_edges=bin_edges,
    )
    _add_colorbar_for_quadmesh(
        fig,
        axes[2],
        pcm,
        colorbar_label=f"Kernel-weighted\nsum of {Unicode.DELTA} {axes_xlabel}",
    )

    # panel 4 - final KM coefficient estimate at target bin
    # with KM estimates at surrounding bins shown for context
    traj_list, disp_list = get_traj_and_diff(dataframe_steady_state, column_names)
    drift = get_kramers_moyal_coeffs(traj_list, disp_list, bin_edges, TIME_STEP_IN_HOURS, kernels)[
        0
    ]
    pcm = _make_2d_pcolormesh(
        axes[3],
        drift[..., 0],
        bin_edges[0],
        bin_edges[1],
        vmin=DRIFT_CONTOUR_VMIN,
        vmax=DRIFT_CONTOUR_VMAX,
        axes_xlabel=axes_xlabel,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        xlabel_kwargs=XLABEL_KWARGS,
    )
    _add_target_bin_border(
        axes[3],
        target_bin=target_bin,
        bin_edges=bin_edges,
    )
    _add_colorbar_for_quadmesh(
        fig,
        axes[3],
        pcm,
        colorbar_label=f"Drift in {axes_xlabel} (hr$^{{-1}}$)",
    )

    # only show y-axis tick labels on the first panel to avoid clutter, since
    # all panels share the same y-axis limits
    for ax in axes[1:]:
        ax.yaxis.set_tick_params(labelleft=False)

    filename = "kernel_convolution_schematic"
    save_plot_to_path(
        fig, savedir, filename, file_format=".svg", tight_layout=False, transparent=True
    )
    return savedir / f"{filename}.svg"
