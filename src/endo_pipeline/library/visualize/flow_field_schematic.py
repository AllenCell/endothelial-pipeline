"""Methods for constructing schematics for the flow field supplementary figure."""

from pathlib import Path
from typing import Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image, save_plot_to_path
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
    get_dataframe_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
)
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
from endo_pipeline.settings.examples import FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_XLARGE
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.image_data import NATIVE_ZARR_RESOLUTION_CROP_SIZE, PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
)


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
    horizontal_arrow_x_offset: float = 0.08,
    horizontal_arrow_y_offset: float = -0.02,
    horizontal_arrow_linewidth: float = 1.5,
    horizontal_arrow_arrowstyle: str = "->,head_length=5,head_width=3",
    text_y_offset: float = -0.15,
    delta_text_y_offset: float = 0.01,
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

    layout_engine: LayoutEngine = fig.get_layout_engine()
    layout_engine.set(rect=(0, 0.2, 1, 0.8))

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

    # ── Curved arrows to (theta,r,rho) labels and vertical arrow between them ──
    # Finalize layout so data→figure transforms are accurate
    fig.canvas.draw()

    bbox_t = ax_t.get_position()
    label_y = bbox_t.y0 + text_y_offset

    # Compute start point for each arrow from the bottom edge of the highlighted box
    def _data_to_fig(ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
        display = ax.transData.transform((x, y))
        fig_coords = fig.transFigure.inverted().transform(display)
        fig_coords[0] += map_arrow_x_offset
        return cast(tuple[float, float], tuple(fig_coords))

    box_mid_x = grid_crop_position[0] + NATIVE_ZARR_RESOLUTION_CROP_SIZE / 2
    box_bottom_y = grid_crop_position[1] + NATIVE_ZARR_RESOLUTION_CROP_SIZE
    arrow_start_t = _data_to_fig(ax_t, box_mid_x, box_bottom_y)
    arrow_start_t1 = _data_to_fig(ax_t1, box_mid_x, box_bottom_y)

    # Align labels horizontally with the midpoint of each highlighted box
    label_x_t = arrow_start_t[0]
    label_x_t1 = arrow_start_t1[0]

    # Text labels
    fig.text(
        label_x_t,
        label_y,
        f"({Unicode.THETA}, r, {Unicode.RHO}) at t",
        ha="center",
        va="top",
        fontsize=FONTSIZE_XLARGE,
    )
    fig.text(
        label_x_t1,
        label_y,
        f"({Unicode.THETA}, r, {Unicode.RHO}) at t+1",
        ha="center",
        va="top",
        fontsize=FONTSIZE_XLARGE,
    )

    # Curved arrows from bottom edge of highlighted box to its (theta, r, rho) label
    for start, lbl_x, rad in [
        (arrow_start_t, label_x_t, map_arrow_rad),
        (arrow_start_t1, label_x_t1, -map_arrow_rad),
    ]:
        arrow = FancyArrowPatch(
            start,
            (lbl_x, label_y - 0.01),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=map_arrow_arrowstyle,
            color="black",
            linewidth=map_arrow_linewidth,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.add_artist(arrow)

    # Horizontal arrow between the two (theta, r, rho) labels
    mid_y = label_y + horizontal_arrow_y_offset
    arrow_start_x = label_x_t + horizontal_arrow_x_offset
    arrow_end_x = label_x_t1 - horizontal_arrow_x_offset
    arrow_mid_x = (arrow_start_x + arrow_end_x) / 2
    fig.text(
        arrow_mid_x,
        mid_y + delta_text_y_offset,
        f"({Unicode.DELTA}{Unicode.THETA}, {Unicode.DELTA}r, {Unicode.DELTA}{Unicode.RHO})",
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_LARGE,
    )
    horizontal_arrow = FancyArrowPatch(
        (arrow_start_x, mid_y),
        (arrow_end_x, mid_y),
        arrowstyle=horizontal_arrow_arrowstyle,
        color="black",
        linewidth=horizontal_arrow_linewidth,
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(horizontal_arrow)

    filename = "flow_field_example_t_to_tp1"
    save_plot_to_path(
        fig, savedir, filename, file_format=".svg", transparent=True, tight_layout=False
    )
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
    label: str | None = None,
) -> None:
    """Add a colorbar for a given QuadMesh plot."""
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(quadmesh, cax=cax, label=label)


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
        label=colorbar_label,
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
        label=colorbar_label,
    )
    return kernel_weights_2d


def _plot_km_coeff_at_target_bin(
    fig: plt.Figure,
    axes: plt.Axes,
    dataframe_steady_state: pd.DataFrame,
    column_names: list[Column.DiffAEData],
    kernels: list[KramersMoyalKernel],
    bin_edges: list[np.ndarray],
    target_bin: tuple[int, int],
    time_step=TIME_STEP_IN_HOURS,
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
) -> None:
    """Plot the Kramers-Moyal coefficient estimate at the target bin."""
    traj_list, disp_list = get_traj_and_diff(dataframe_steady_state, column_names)
    drift = get_kramers_moyal_coeffs(traj_list, disp_list, bin_edges, time_step, kernels)[0]
    pcm = _make_2d_pcolormesh(
        axes,
        drift[..., 0],
        bin_edges[0],
        bin_edges[1],
        cmap=cmap,
        vmin=DRIFT_CONTOUR_VMIN,
        vmax=DRIFT_CONTOUR_VMAX,
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
        label=colorbar_label,
    )


def make_kernel_convolution_schematic(
    savedir: Path,
    dataset_name: str,
    column_names: list[Column.DiffAEData],
    target_point: tuple[float, float],
    axes_xlim: tuple[float, float] | None = None,
    axes_ylim: tuple[float, float] | None = None,
    n_rows: int = 1,
    n_cols: int = 4,
    cmap: str = DRIFT_CONTOUR_COLORMAP,
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
) -> Path:
    """
    Build the panel showing a schematic of the kernel convolution process for
    a single target bin in (r, rho) space.
    """
    # Load feature data for provided dataset and filter to steady-state time
    # points and a single flow condition
    dataset_config = load_dataset_config(dataset_name)

    feature_manifest_name = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
    feature_manifest = load_dataframe_manifest(feature_manifest_name)

    dataset_location = get_dataframe_location_for_dataset(feature_manifest, dataset_name)

    columns_to_load = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]

    df_raw = load_dataframe(dataset_location, delay=True)
    df: pd.DataFrame = df_raw[columns_to_load].compute()
    dataframe_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    bin_widths = tuple(BIN_WIDTHS_DYNAMICS[col] for col in column_names)
    bin_edges, bin_centers = get_bins(bin_widths, df[column_names].to_numpy())
    target_bin = _get_target_bin(target_point, bin_edges)

    fig, ax = plt.subplots(n_rows, n_cols, gridspec_kw=gridspec_kwargs, **(fig_kwargs or {}))
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
        axes_ylabel=axes_ylabel,
        ylabel_kwargs=ylabel_kwargs,
        cmap=cmap,
        colorbar_label=f"sum of $\\Delta$ {axes_xlabel}",
    )

    # get kernels for panel 2
    KernelName: TypeAlias = Literal["periodic", "gaussian", "epanechnikov"]
    kernels = [
        KramersMoyalKernel(
            name=cast(KernelName, KERNEL_NAMES_DYNAMICS[column_name]),
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
            period=POLAR_ANGLE_PERIOD if column_name == Column.DiffAEData.POLAR_ANGLE else None,
        )
        for column_name in column_names
    ]
    # panel 2 - kernel weights centered at target bin
    kernel_weights_2d = _plot_kernel_at_target_bin(
        fig=fig,
        axes=axes[1],
        kernels=kernels,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        target_bin=target_bin,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        colorbar_label="kernel weight (normalized)",
    )

    # panel 3 - kernel-weighted histogram (i.e. numerator of KM estimator)
    kernel_weighted_hist_delta_r = kernel_weights_2d * weighted_hist_delta_r
    vmax = np.nanpercentile(np.abs(kernel_weighted_hist_delta_r), 99)
    pcm = _make_2d_pcolormesh(
        axes[2],
        kernel_weighted_hist_delta_r,
        bin_edges[0],
        bin_edges[1],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_xlabel=axes_xlabel,
        axes_ylabel=axes_ylabel,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
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
        label=f"kernel-weighted sum of $\\Delta$ {axes_xlabel}",
    )

    # panel 4 - final KM coefficient estimate at target bin
    # with KM estimates at surrounding bins shown for context
    _plot_km_coeff_at_target_bin(
        fig=fig,
        axes=axes[3],
        dataframe_steady_state=dataframe_steady_state,
        column_names=column_names,
        kernels=kernels,
        bin_edges=bin_edges,
        target_bin=target_bin,
        axes_xlim=axes_xlim,
        axes_ylim=axes_ylim,
        axes_xlabel=axes_xlabel,
        xlabel_kwargs=xlabel_kwargs,
        colorbar_label=f"drift in {axes_xlabel} (hr$^{{-1}}$)",
    )

    filename = "kernel_convolution_schematic"
    save_plot_to_path(
        fig, savedir, filename, file_format=".svg", tight_layout=False, transparent=True
    )
    return savedir / f"{filename}.svg"
