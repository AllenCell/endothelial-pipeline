"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_1d,
    plot_drift_contours,
    plot_drift_quiver,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED


def _add_colorbar_to_contour_plot(
    fig: plt.Figure,
    axes: plt.Axes,
    vmin: float = DRIFT_CONTOUR_VMIN,
    vmax: float = DRIFT_CONTOUR_VMAX,
    ticks: np.ndarray | None = None,
    tick_label_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    cax_position: Literal["top", "bottom", "left", "right"] = "right",
    extend: Literal["neither", "both", "min", "max"] = "both",
    pad: float = 0.03,
) -> None:
    """
    Add a colorbar to a contour plot with specified formatting.

    Parameters
    ----------
    fig
        Matplotlib figure object containing the contour plot.
    axes
        Matplotlib axes object containing the contour plot.
    vmin
        Minimum value for the colorbar.
    vmax
        Maximum value for the colorbar.
    ticks
        Array of tick values for the colorbar. If None, ticks will be generated
        automatically based on `vmin`, `vmax`, and
        `DRIFT_CONTOUR_CBAR_NUM_TICKS`.
    tick_label_round
        Number of decimal places to round colorbar tick labels to.
    colormap
        Colormap to use for the colorbar.
    orientation
        Orientation of the colorbar, either "vertical" or "horizontal".
    cax_position
        Position of the colorbar axes relative to the main axes, one of "top",
        "bottom", "left", or "right".
    pad
        Padding between the main axes and the colorbar axes, in inches.

    """
    cax = inset_axes(
        axes,
        width="100%" if orientation == "horizontal" else "5%",
        height="5%" if orientation == "horizontal" else "100%",
        loc="lower left",
        bbox_to_anchor=(
            (0, 1.0 + pad, 1, 1) if orientation == "horizontal" else (1.0 + pad, 0, 1, 1)
        ),
        bbox_transform=axes.transAxes,
        borderpad=0,
    )

    color_mappable = ScalarMappable(
        norm=TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0), cmap=colormap
    )
    colorbar_ticks = (
        ticks
        if ticks is not None
        else np.linspace(
            np.round(vmin, tick_label_round),
            np.round(vmax, tick_label_round),
            DRIFT_CONTOUR_CBAR_NUM_TICKS,
        )
    )
    colorbar_ticks = np.round(colorbar_ticks, tick_label_round)

    fig.colorbar(
        color_mappable, cax=cax, orientation=orientation, ticks=colorbar_ticks, extend=extend
    )

    tick_axis = cax.xaxis if orientation == "horizontal" else cax.yaxis
    tick_axis.set_ticks_position(cax_position)
    tick_axis.set_label_position(cax_position)


def _get_nullcline_coords_from_contour_axes(
    axes: np.ndarray[plt.Axes, Any],
    column_names: list[Column.DiffAEData],
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
) -> dict[Column.DiffAEData, pd.DataFrame]:
    """
    Extract (r, rho) coordinates along nullclines from the contour collections
    in the given axes.

    Parameters
    ----------
    axes
        Axes objects containing the contour plots with nullclines.
    column_names
        List of column names corresponding to the nullclines, in the same order
        as they are plotted in the axes.
    r_lims
        Tuple specifying the limits of the r-axis, used to clip nullcline
        coordinates to the plotted range.
    rho_lims
        Tuple specifying the limits of the rho-axis, used to clip nullcline
        coordinates to the plotted range.

    Returns
    -------
    :
        Dictionary mapping each column name to a DataFrame containing the (r,
        rho) coordinates of the corresponding nullcline.

    """
    nullcline_coords = {col: pd.DataFrame() for col in column_names}
    for index, column in enumerate(column_names):
        # get coordinates from the contour collections corresponding to the
        # nullclines (these are generated by `plot_drift_contours` with
        # `include_nullclines=True`, last thing plotted on each subplot)
        nullcline_collection = axes[index].collections[-1]
        nullcline_paths = nullcline_collection.get_paths()
        for path in nullcline_paths:
            path_coords_ = path.vertices.T.copy()
            # clip to the axes limits to avoid including coordinates outside the
            # plotted range
            is_in_bounds = (
                (path_coords_[0] >= r_lims[0])
                & (path_coords_[0] <= r_lims[1])
                & (path_coords_[1] >= rho_lims[0])
                & (path_coords_[1] <= rho_lims[1])
            )
            path_coords = path_coords_[:, is_in_bounds]
            # sort so that coordinates are ordered by:
            #  - increasing rho for the r-nullcline (so that we can generate images by increasing rho)
            #  - increasing r for the rho-nullcline (so that we can generate images by increasing r)
            sort_arg = 1 if column == Column.DiffAEData.POLAR_RADIUS else 0
            sort_indices = np.argsort(path_coords[sort_arg])
            path_coords[0] = path_coords[0][sort_indices]
            path_coords[1] = path_coords[1][sort_indices]

            # take only 5 evenly spaced coordinates along the nullcline to avoid
            # generating too many points
            example_indices = np.round(np.linspace(0, len(path_coords[0]) - 1, 5)).astype(int)
            path_coords = path_coords[:, example_indices]

            # add points to plot
            axes[index].plot(
                path_coords[0],
                path_coords[1],
                "o",
                color="w",
                markeredgecolor="k",
                markeredgewidth=0.25,
                markersize=3,
            )

            # append the coordinates to the DataFrame for the corresponding column
            nullcline_coords[column] = pd.concat(
                [
                    nullcline_coords[column],
                    pd.DataFrame(
                        {
                            Column.DiffAEData.POLAR_RADIUS: path_coords[0],
                            Column.DiffAEData.PC3_FLIPPED: path_coords[1],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return nullcline_coords


def make_2d_contour_plot_panel(
    drift: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray],
    column_labels: list[str],
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
    r_ticks: list[float],
    rho_ticks: list[float],
    nullcline_r_style: str,
    nullcline_rho_style: str,
    nullcline_opacity: float,
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
    axes_title_kwargs: dict | None,
    include_colorbar: bool = False,
) -> tuple[Path, dict[Column.DiffAEData, pd.DataFrame]]:
    """
    Make and save plot of drift contours in (r, rho) space for a given dataset.
    """
    column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_labels = [COLUMN_METADATA[column].label or str(column) for column in column_names]
    # plot drift contours and save
    fig, axes_ = plot_drift_contours(
        meshgrid=meshgrid,
        drift=drift,
        variable_labels=column_labels,
        figsize=figsize,
        n_rows=1,
        n_cols=2,
        axes_limits=[r_lims, rho_lims],
        axes_aspect=None,
        axes_titles=(f"d{column_labels[0]}/dt", f"d{column_labels[1]}/dt"),
        include_colorbar=False,
        include_nullclines=True,
        nullcline_colors=("k", "k"),
        nullcline_styles=(nullcline_r_style, nullcline_rho_style),
        nullcline_opacity=nullcline_opacity,
        gridspec_kwargs=gridspec_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        axes_title_kwargs=axes_title_kwargs,
    )
    for ax_index, ax_ in enumerate(list(axes_)):
        # adjust label padding and drop tick labels on shared y axis
        ax_.set_box_aspect(1.0)
        ax_.set_xticks(r_ticks)
        ax_.set_yticks(rho_ticks)
        if ax_index == 1:
            ax_.tick_params(labelleft=False)

    # if indicated, add colorbar to the top of the first subplot with ticks and
    # label formatting
    if include_colorbar:
        _add_colorbar_to_contour_plot(fig, axes_[1])

    # get (r, rho) coordinates of r- and rho-nullclines for generating images
    axes = cast(np.ndarray[plt.Axes, Any], axes_)
    nullcline_coords = _get_nullcline_coords_from_contour_axes(axes, column_names, r_lims, rho_lims)

    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        pad_inches=0,
    )

    return fig_savedir / f"{filename}.svg", nullcline_coords


def make_2d_quiver_plot_panel(
    drift: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray],
    column_labels: list[str],
    stable_fixed_point: np.ndarray,
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    r_lims: tuple[float, float],
    rho_lims: tuple[float, float],
    r_ticks: list[float],
    rho_ticks: list[float],
    nullcline_r_style: str,
    nullcline_rho_style: str,
    nullcline_opacity: float,
    quiver_color: str,
    quiver_scale: float,
    quiver_downsample: int,
    vmin: float,
    vmax: float,
    include_legend: bool,
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
    quiver_legend_kwargs: dict | None,
) -> Path:
    fig, ax = plot_drift_quiver(
        drift=drift,
        meshgrid=meshgrid,
        quiver_scale=quiver_scale,
        quiver_color=quiver_color,
        quiver_downsample=quiver_downsample,
        vmin=vmin,
        vmax=vmax,
        variable_labels=column_labels,
        figsize=figsize,
        axes_limits=[r_lims, rho_lims],
        include_nullclines=True,
        nullcline_colors=("k", "k"),
        nullcline_styles=(nullcline_r_style, nullcline_rho_style),
        nullcline_opacity=nullcline_opacity,
        gridspec_kwargs=gridspec_kwargs,
        legend_kwargs=quiver_legend_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        plot_legend=include_legend,
    )

    ax.plot(
        stable_fixed_point[..., 0],
        stable_fixed_point[..., 1],
        FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=5,
        label="Stable fixed point",
    )
    if include_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=2,
            handletextpad=0.3,
        )

    # make room above axes for the legend
    fig.subplots_adjust(top=0.82)

    # set plot formatting args and save
    ax.set_box_aspect(1.0)
    ax.set_xticks(r_ticks)
    ax.set_yticks(rho_ticks)
    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )

    return fig_savedir / f"{filename}.svg"


def make_1d_drift_plot_panel(
    drift: np.ndarray,
    theta_values: np.ndarray,
    column_label: str,
    stable_fixed_point: float,
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    shear_stress_label: str,
    axes_xlim: tuple[float, float],
    axes_ylim: tuple[float, float],
    axes_xticks: list[float],
    axes_xtick_labels: list[str],
    axes_yticks: list[float],
    arrow_scale: float,
    drift_line_kwargs: dict | None,
    zero_line_kwargs: dict | None,
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
) -> Path:
    fig, ax = plot_drift_1d(
        drift=drift,
        x_values=theta_values,
        figsize=figsize,
        axes_limits=[axes_xlim, axes_ylim],
        axes_labels=[column_label, f"d{column_label}/dt"],
        add_flow_arrows=True,
        flow_arrow_kwargs={"color": "dimgrey", "scale": arrow_scale},
        flow_arrow_downsample=10,
        gridspec_kwargs=gridspec_kwargs,
        drift_line_kwargs=drift_line_kwargs,
        zero_line_kwargs=zero_line_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )
    # add stable fixed point in theta
    ax.plot(
        stable_fixed_point,
        np.zeros_like(stable_fixed_point),
        FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=5,
    )

    # reserve left margin for the vertical label
    fig.subplots_adjust(left=0.08)
    # add vertical title to the left of the contour plot spanning all rows
    fig.text(
        -0.5,
        0.5,
        shear_stress_label,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=FONTSIZE_MEDIUM,
        fontweight="bold",
    )

    # set plot formatting args
    ax.set_xticks(axes_xticks, labels=axes_xtick_labels)
    ax.set_yticks(axes_yticks)

    save_plot_to_path(
        fig, fig_savedir, filename, file_format=".svg", tight_layout=False, transparent=True
    )

    return fig_savedir / f"{filename}.svg"


def reconstruct_along_nullcline(
    nullcline_coords: dict[ColumnNameType, pd.DataFrame],
    theta_value: float,
    model: DiffusionAutoEncoder,
    fig_savedir: Path,
    num_gpus: int | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> tuple[Path, ...]:
    """
    Generate reconstructed images along a nullcline given the coordinates of the
    nullcline in feature space.

    Parameters
    ----------
    nullcline_coords
        Dictionary containing DataFrames with the coordinates of each nullcline.
    theta_value
        Value of the theta feature to use for generating the images along the
        nullcline (since the nullcline coordinates only contain r and rho).
    model
        DiffusionAutoEncoder model used to generate synthetic images.
    fig_savedir
        Directory where the generated figures will be saved.
    num_gpus
        Number of GPUs to use for image generation. If None, will use all
        available GPUs.
    random_seed
        Random seed for reproducibility of image generation. If None, will use a
        random seed.
    """
    column_names = cast(
        list[str],
        [
            Column.DiffAEData.POLAR_RADIUS,
            Column.DiffAEData.PC3_FLIPPED,
            Column.DiffAEData.POLAR_ANGLE,
        ],
    )
    output_paths: list[Path] = []
    for column, coords_dataframe in nullcline_coords.items():
        r_coords = coords_dataframe[Column.DiffAEData.POLAR_RADIUS]
        rho_coords = coords_dataframe[Column.DiffAEData.PC3_FLIPPED]
        full_coords_dataframe = pd.DataFrame(
            {
                Column.DiffAEData.POLAR_RADIUS: r_coords,
                Column.DiffAEData.PC3_FLIPPED: rho_coords,
                Column.DiffAEData.POLAR_ANGLE: theta_value * np.ones_like(r_coords),
            }
        )

        walk_array = generate_from_dataframe(
            full_coords_dataframe, column_names, model, num_gpus=num_gpus, random_seed=random_seed
        )
        # reverse order so that images are generated in the direction of
        # increasing r or rho (since nullcline coords are sorted by increasing r
        # or rho)
        walk_array = walk_array[::-1]
        fig_null_walk = make_contact_sheet(
            panels=[walk_array[i] for i in range(len(walk_array))],
            max_rows=len(walk_array),
            max_cols=1,
            fig_kwargs={"figsize": (0.45, 1.85), "layout": "constrained"},
            gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        )
        for i, ax in enumerate(fig_null_walk.axes):
            ax.text(
                0.98,
                0.98,
                f"r={r_coords[i]:.3f}\n{Unicode.RHO}={rho_coords[i]:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=2,
                color="white",
            )
        save_plot_to_path(
            fig_null_walk,
            fig_savedir,
            f"walk_array_null_{column}",
            file_format=".svg",
            tight_layout=False,
            pad_inches=0,
        )
        output_paths.append(fig_savedir / f"walk_array_null_{column}.svg")

    return tuple(output_paths)


def make_crop_example_contact_sheet(
    stable_fixed_point_dataframe: pd.DataFrame,
    feature_column_names: list[Column.DiffAEData],
    model: DiffusionAutoEncoder,
    n_crop_examples: int,
    fig_savedir: Path,
    fig_filename: str,
    file_format: Literal[".svg", ".png", ".pdf"] = ".svg",
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
    num_gpus: int | None = None,
    random_seed: int | None = RANDOM_SEED,
    scale_bar_um: int = 10,
    title: str | None = None,
) -> Path:
    """
    Make figure panel plot showing example crops at stable fixed points.

    Three types of image crops are shown:
        1. Synthetic "VE-cadherin" images generated by the DiffAE from the
           stable fixed point coordinates.
        2. Real VE-cadherin (max. intensity projection) images corresponding to
           the stable fixed point coordinates.
        3. Real brightfield (standard deviation projection) images corresponding
           to the stable fixed point coordinates.

    The examples are assembled as a contact sheet with 3 columns and
    n_crop_examples rows.

    Parameters
    ----------
    stable_fixed_point_dataframe
        DataFrame containing the coordinates of the stable fixed points.
    feature_column_names
        List of column names in the DataFrame corresponding to the stable fixed
        point coordinates.
    model
        DiffusionAutoEncoder model used to generate synthetic images.
    n_crop_examples
        Number of crop examples of each type to show in the figure panel.
    fig_savedir
        Directory to save the figure to.
    fig_filename
        Filename to save the figure as.
    file_format
        File format to save the figure as.
    gridspec_kwargs
        Additional keyword arguments for the gridspec layout of the contact
        sheet.
    fig_kwargs
        Additional keyword arguments for the figure layout of the contact sheet.
    bin_width_real_crops
        Bin width in feature space to use when filtering real crops to those
        near the stable fixed point coordinates.
    image_loading_resolution_level
        Resolution level to load the real images at.
    num_gpus
        Number of GPUs to use for image generation. If None, will use all
        available GPUs.
    random_seed
        Random seed for reproducibility of image generation. If None, will use a
        random seed.
    scale_bar_um
        Length of the scale bar in micrometers.

    Returns
    -------
    :
        Path to the saved figure.

    """
    if len(stable_fixed_point_dataframe) > 1:
        raise ValueError(
            "Expected stable_fixed_point_dataframe to contain only one row, but it contains "
            f"{len(stable_fixed_point_dataframe)} rows."
        )

    generated_images = generate_from_dataframe(
        stable_fixed_point_dataframe,
        cast(list[str], feature_column_names),
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
        n_noise_samples=n_crop_examples,
    )
    generated_image_list = [generated_images[i] for i in range(len(generated_images))]

    fig = make_contact_sheet(
        panels=generated_image_list,
        max_rows=n_crop_examples,
        max_cols=1,
        direction="top-down first",
        gridspec_kwargs=gridspec_kwargs,
        subplot_kwargs={"frame_on": False},
        fig_kwargs=fig_kwargs,
    )

    for i, ax in enumerate(fig.axes):
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="both", pad=2)

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=4,
            padding=6,
            include_label=True if i == 0 else False,
        )

    # if `title` is provided, add title above the first panel
    if title is not None:
        # expand the figure height to make room for the title so the axes
        # content is not squashed
        fig_width, fig_height = fig.get_size_inches()
        title_space_in = 0.25  # inches of extra height reserved for the title
        new_height = fig_height + title_space_in
        fig.set_size_inches(fig_width, new_height)

        # axes content occupies the bottom `content_top` fraction of the new height
        content_top = fig_height / new_height
        layout_engine = fig.get_layout_engine()
        if isinstance(layout_engine, ConstrainedLayoutEngine):
            layout_engine.set(rect=(0.0, 0, 1, content_top))

        fig.text(
            0.5,
            content_top + (1 - content_top) / 2,  # centred in the title band
            title,
            va="center",
            ha="center",
            rotation="horizontal",
            fontsize=FONTSIZE_XSMALL,
            fontweight="bold",
        )

    save_plot_to_path(
        fig,
        fig_savedir,
        fig_filename,
        file_format=file_format,
        tight_layout=False,
        pad_inches=0,
    )

    return fig_savedir / f"{fig_filename}{file_format}"
