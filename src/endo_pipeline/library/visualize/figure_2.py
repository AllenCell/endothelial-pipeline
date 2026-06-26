"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Literal, cast

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.track_integration import get_line_fit_and_filtered_df
from endo_pipeline.library.analyze.vector_field_estimation import load_drift_dataframe_for_dataset
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    make_legend_handles_for_fixed_pts,
    plot_drift_1d,
    plot_drift_3d,
    plot_drift_contours,
    process_3d_vector_field_for_visualization,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.first_passage_time import FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.flow_field_figure import (
    AXES_LIMITS_2D,
    GRIDSPEC_KWARGS,
    NULLCLINE_STYLES_2D,
    XLABEL_KWARGS,
    YLABEL_KWARGS,
)
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME


def add_colorbar_to_contour_plot(
    fig: plt.Figure,
    vmin: float = DRIFT_CONTOUR_VMIN,
    vmax: float = DRIFT_CONTOUR_VMAX,
    ticks: np.ndarray | None = None,
    tick_label_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    ticks_cax_position: Literal["top", "bottom", "left", "right"] = "right",
    label_cax_position: Literal["top", "bottom", "left", "right"] = "right",
    extend: Literal["neither", "both", "min", "max"] = "both",
    cax_rect: tuple[float, float, float, float] = (0.25, 0.9, 0.5, 0.02),
) -> None:
    """
    Add a colorbar to a contour plot with specified formatting.

    Parameters
    ----------
    fig
        Matplotlib figure object containing the contour plot.
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
    ticks_cax_position
        Which side of the colorbar axes to place the ticks, one of
        "top", "bottom", "left", or "right".
    label_cax_position
        Which side of the colorbar axes to place the label, one of
        "top", "bottom", "left", or "right".
    cax_rect
        Position of the colorbar axes in normalized figure coordinates
        ``(left, bottom, width, height)``. Added via ``fig.add_axes`` so its
        position is fully independent of the main axes layout.

    """
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

    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(
        color_mappable, cax=cax, orientation=orientation, ticks=colorbar_ticks, extend=extend
    )
    cax = cbar.ax

    if orientation == "horizontal":
        cax.xaxis.set_ticks_position(
            cast(Literal["top", "bottom", "both", "default", "none"], ticks_cax_position)
        )
        cax.xaxis.set_label_position(cast(Literal["top", "bottom"], label_cax_position))
    else:
        cax.yaxis.set_ticks_position(
            cast(Literal["left", "right", "both", "default", "none"], ticks_cax_position)
        )
        cax.yaxis.set_label_position(cast(Literal["left", "right"], label_cax_position))
    cbar.set_label("vector field \ncomponent value", fontsize=FONTSIZE_XSMALL, labelpad=2)


@figure_panel("Make panel of 2D contour plots of drift in (r, rho) space.")
def make_2d_contour_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    drift: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray],
    column_labels: list[str],
    stable_fixed_point: np.ndarray,
    filename: str,
    include_legend: bool = True,
    include_colorbar: bool = True,
) -> Path:
    """
    Make and save plot of drift contours in (r, rho) space for a given dataset.
    """
    column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_labels = [COLUMN_METADATA[column].label or str(column) for column in column_names]

    r_lims = AXES_LIMITS_2D[Column.DiffAEData.POLAR_RADIUS]
    rho_lims = AXES_LIMITS_2D[Column.DiffAEData.PC3_FLIPPED]
    r_ticks = [0.4, 1.0, 1.6]
    rho_ticks = [-0.75, 0.0, 0.75]
    nullcline_r_style = NULLCLINE_STYLES_2D[Column.DiffAEData.POLAR_RADIUS]
    nullcline_rho_style = NULLCLINE_STYLES_2D[Column.DiffAEData.PC3_FLIPPED]
    nullcline_opacity = 1.0
    gridspec_kwargs = GRIDSPEC_KWARGS
    xlabel_kwargs = XLABEL_KWARGS
    ylabel_kwargs = {**YLABEL_KWARGS, "rotation": 0}
    axes_title_kwargs = {
        "fontsize": FONTSIZE_SMALL,
        "x": 0.05,
        "y": 0.775,
        "rotation": 0,
        "ha": "left",
        "va": "center",
        "bbox": {
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    }

    # Use explicit axes rects so the subplot positions are fixed in figure
    # coordinates regardless of whether colorbar/legend are included.
    # Colorbar and legend are placed below the figure (negative y) and
    # captured by bbox_inches="tight" on save.
    subplot_rects = [(0.12, 0.58, 0.55, 0.35), (0.12, 0.18, 0.55, 0.35)]
    fig, axes_ = plot_drift_contours(
        meshgrid=meshgrid,
        drift=drift,
        variable_labels=column_labels,
        figsize=figure_size,
        n_rows=2,
        n_cols=1,
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
        axes_rects=subplot_rects,
    )

    for ax_index, ax_ in enumerate(list(axes_)):
        # add stable fixed point on top of the contour plot
        ax_.plot(
            stable_fixed_point[..., 0],
            stable_fixed_point[..., 1],
            FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
            color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
            markeredgecolor="k",
            markeredgewidth=0.5,
            markersize=5,
        )
        # adjust label padding and drop tick labels on shared x axis
        ax_.set_box_aspect(1.0)
        ax_.set_xticks(r_ticks)
        ax_.set_yticks(rho_ticks)
        ax_.yaxis.set_label_position("left")
        ax_.yaxis.tick_left()
        if ax_index == 0:
            ax_.tick_params(labelbottom=False)

    if include_colorbar:
        add_colorbar_to_contour_plot(
            fig,
            orientation="horizontal",
            ticks_cax_position="bottom",
            label_cax_position="top",
            cax_rect=(0.255, -0.02, 0.5, 0.025),
        )

    if include_legend:
        handles = []
        labels = []
        # plot_drift_contours draws nullclines via ax.contour(), which does not
        # produce labeled artists. Add proxy Line2D handles so the legend has
        # something to show.
        nullcline_styles = (nullcline_r_style, nullcline_rho_style)
        for col_idx, col in enumerate(column_names):
            label = COLUMN_METADATA[col].label or str(col)
            legend_label = f"{label}-nullcline (d{label}/dt=0)"
            handle = mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=nullcline_styles[col_idx],
                label=legend_label,
            )
            handles.append(handle)
            labels.append(legend_label)

        fixed_point_label = f"({column_labels[0]}$^*$, {column_labels[1]}$^*$)"
        fp_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=[StabilityLabel.STABLE],
            marker_size=4,
            labels={StabilityLabel.STABLE: fixed_point_label},
        )
        handles.extend(fp_handles)
        labels.append(fixed_point_label)
        fig.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="lower center",
            bbox_to_anchor=(0.25, -0.09),
            ncol=1,
            handletextpad=0.3,
            frameon=False,
        )

    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    return output_path / f"{filename}.svg"


@figure_panel("Make panel of 1D phase line plot of drift in theta space.")
def make_1d_drift_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    shear_stress_label: str,
    drift: np.ndarray,
    theta_values: np.ndarray,
    column_label: str,
    stable_fixed_point: float,
    filename: str,
    arrow_scale: float,
    arrow_width: float,
    include_legend: bool = True,
) -> Path:
    """Make and save plot of 1D drift as a function of theta for a given dataset."""
    axes_xlim = VECTOR_FIELD_THETA_RANGE
    axes_ylim = (-0.4, 0.4)

    # re-wrap theta values to be within the specified x-axis limits for better
    # visualization of the drift as a function of theta
    where_theta_below_xlim = theta_values < axes_xlim[0]
    where_theta_above_xlim = theta_values > axes_xlim[1]
    theta_values_wrapped = theta_values.copy()
    theta_values_wrapped[where_theta_below_xlim] += np.pi
    theta_values_wrapped[where_theta_above_xlim] -= np.pi
    arg_sorted_theta = np.argsort(theta_values_wrapped)
    theta_values_sorted = theta_values_wrapped[arg_sorted_theta]
    drift_sorted = drift[arg_sorted_theta]

    fig, ax = plot_drift_1d(
        drift=drift_sorted,
        x_values=theta_values_sorted,
        figsize=figure_size,
        axes_limits=[axes_xlim, axes_ylim],
        axes_labels=[column_label, ""],
        add_flow_arrows=True,
        flow_arrow_kwargs={"color": "dimgrey", "scale": arrow_scale, "width": arrow_width},
        flow_arrow_downsample=10,
        gridspec_kwargs=GRIDSPEC_KWARGS,
        axes_rect=(0.12, 0.08, 0.85, 0.76),
        drift_line_kwargs={"color": "k", "linewidth": 2, "label": f"d{column_label}/dt"},
        zero_line_kwargs={
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1,
            "alpha": 0.7,
            "label": f"d{column_label}/dt = 0",
        },
        xlabel_kwargs=XLABEL_KWARGS,
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

    if include_legend:
        handles, labels = ax.get_legend_handles_labels()
        fixed_point_label = f"{column_label}$^*$"
        fp_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=[StabilityLabel.STABLE],
            marker_size=4,
            labels={StabilityLabel.STABLE: fixed_point_label},
        )
        handles.extend(fp_handles)
        labels.append(fixed_point_label)
        fig.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper right",
            bbox_to_anchor=(0.97, 0.84),
            ncol=2,
            handletextpad=0.3,
            columnspacing=0.5,
        )

    # set plot formatting args
    ax.set_box_aspect(1.0)
    ax.set_xticks([0, np.pi / 2], labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"])
    ax.set_yticks([-0.3, 0.0, 0.3])

    # add shear stress label as title
    ax.set_title(shear_stress_label, fontsize=FONTSIZE_SMALL, fontweight="bold", pad=3)

    save_plot_to_path(
        fig, output_path, filename, file_format=".svg", tight_layout=False, transparent=True
    )

    return output_path / f"{filename}.svg"


@figure_panel("Reconstruct at fixed point.")
def reconstruct_fixed_points(
    fixed_point_df: pd.DataFrame,
    shear_stress_label: str,
    model: DiffusionAutoEncoder,
    figure_size: tuple[float, float],
    output_path: Path,
    num_gpus: int | None = None,
    random_seed: int | None = 4,
    include_row_label: bool = False,
) -> Path:
    """
    Reconstruct the fixed point coordinates from the polar angle, radius, and
    rho columns.

    Parameters
    ----------
    fixed_point_df
        DataFrame containing the fixed point coordinates, with columns for polar
        angle, polar radius, and flipped PC3 (rho).
    model
        The diffusion autoencoder model used for reconstruction.
    fig_savedir
        Directory to save the reconstructed figures.
    num_gpus
        Number of GPUs to use for reconstruction. If None, will use CPU.
    random_seed_start
        Starting random seed for reproducibility.
    num_examples
        Number of examples to generate for each fixed point coordinate (by
        varying the random seed).
    include_row_label
        If True, label the row of the contact sheet with "Reconstructed
        VE-cadherin MIP." Else, do not label the row of the contact sheet.
    """

    # Reconstruct images along at the fixed point coordinates and make a contact
    # sheet of the results
    column_names = cast(list[str], list(DYNAMICS_COLUMN_NAMES))

    # Column names in fixed point dataframe use the template "%s_fixed_point",
    # but the generate_from_dataframe function expects the base column names, so
    # remove the suffix here before passing to the generation function.
    column_rename_dict = {
        column: column.replace(ColumnTemplate.FIXED_POINT.replace("%s", ""), "")
        for column in fixed_point_df.columns
    }
    fixed_point_df_renamed = fixed_point_df.rename(columns=column_rename_dict)

    reconstructed_image = generate_from_dataframe(
        fixed_point_df_renamed,
        column_names,
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
    )

    fig_fixed_point_reconstructions = make_contact_sheet(
        panels=[reconstructed_image],
        max_rows=1,
        max_cols=1,
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        font_size=FONTSIZE_XSMALL,
    )

    # Add axes title ({feat_1}^*, {feat_2}^*, {feat_3}^*) labeling the
    # fixed point, using the same stable fixed point marker color.
    ax = fig_fixed_point_reconstructions.axes[0]
    ax.set_title(
        f"{shear_stress_label}:\n({Unicode.THETA}$^*$, r$^*$, {Unicode.RHO}$^*$)",
        color="black",
        fontsize=FONTSIZE_SMALL,
        fontweight="bold",
    )

    # Add row title (y label) as standalone text so that including vs. not
    # doesn't affect the size of the plot vis-a-vis the input figure size and
    # constrained layout.
    if include_row_label:
        fig_fixed_point_reconstructions.text(
            -0.1,
            0.35,
            "Reconstructed\nVE-cadherin patch",
            va="center",
            ha="center",
            fontsize=FONTSIZE_SMALL,
            fontweight="bold",
            rotation=90,
        )

    # add scalebars to each panel, only label the top left one to avoid
    # redundancy
    for i, ax in enumerate(fig_fixed_point_reconstructions.axes):
        add_scalebar(
            ax,
            scale_bar_um=20,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=5,
            padding=5,
            include_label=True if i == 0 else False,
            label_fontsize=FONTSIZE_XSMALL,
        )

    dataset_name = fixed_point_df[Column.DATASET].unique()[0]
    filename = f"{dataset_name}_fixed_point_reconstructions"
    save_plot_to_path(
        fig_fixed_point_reconstructions,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg"


@figure_panel("Make panel of 3D vector field plot with stable fixed point overlay.")
def make_3d_vector_field_plot_panel(
    figure_size: tuple[float, float],
    output_path: Path,
    dataset_name: str,
    shear_stress_label: str,
    include_colorbar: bool = True,
    include_legend: bool = True,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset, with
    the stable fixed point overlaid as a scatter marker.

    Parameters
    ----------
    figure_size
        Size of the figure to create.
    output_path
        Directory in which to save the figure panel.
    dataset_name
        Name of the dataset to visualize.
    shear_stress_label
        Label for the shear stress condition to include in the plot.
    include_colorbar
        Whether to include a colorbar indicating the magnitude of the drift
        vectors.
    include_legend
        Whether to include a legend indicating the stable fixed point marker.

    Returns
    -------
    :
        Path to the saved figure file.

    """
    drift_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]
    fixed_point_label = f"({col_labels[0]}$^*$, {col_labels[1]}$^*$, {col_labels[2]}$^*$)"

    theta_lims = VECTOR_FIELD_THETA_RANGE
    r_lims = (0, 1.75)
    rho_lims = (-1.5, 1.5)

    # Load, clip, and downsample drift vector field
    drift, meshgrid = process_3d_vector_field_for_visualization(
        drift_dataframe,
        feature_dataframe,
        column_names=column_names,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        mask_threshold=0.025,
    )

    fig, ax = plot_drift_3d(
        drift=drift,
        meshgrid=meshgrid,
        figsize=figure_size,
        include_colorbar=include_colorbar,
        include_legend=include_legend,
        fixed_point_legend_label=fixed_point_label,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        xticks=[0, np.pi / 2],
        xtick_labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"],
        yticks=[0.25, 0.75, 1.25],
        zticks=[-1.0, 0, 1.0],
        xlabel=col_labels[0],
        ylabel=col_labels[1],
        zlabel=col_labels[2],
        xlabel_kwargs={"labelpad": -8},
        ylabel_kwargs={"labelpad": -5},
        zlabel_kwargs={"labelpad": -8},
    )

    # Load and overlay stable fixed point
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    stable_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
    ]
    fpt_coords = stable_df[column_names].to_numpy()
    hex_color: str = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    ax.scatter(
        fpt_coords[:, 0],
        fpt_coords[:, 1],
        fpt_coords[:, 2],
        color=hex_color,
        s=15,
        zorder=5,
    )

    # add shear stress label as title
    ax.set_title(shear_stress_label, fontsize=FONTSIZE_SMALL, fontweight="bold", pad=0)

    # save as .svg file
    filename = f"3d_vector_field_{dataset_name}"
    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg"


@figure_panel("Make panel of histogram of first passage time correlation values across datasets.")
def make_first_passage_time_correlation_hist(
    figure_size: tuple[float, float], output_path: Path, dataset_names: list[str]
) -> Path:
    fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME)
    line_fit_df, _ = get_line_fit_and_filtered_df(fpt_manifest, dataset_names)

    pearson_correlations = []
    for _, df_dataset in line_fit_df.groupby(Column.DATASET):
        pearson_r = df_dataset[Column.VectorField.PEARSON_R].iloc[0]
        pearson_correlations.append(pearson_r)

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.hist(pearson_correlations, bins=list(np.linspace(-1, 1, 21)), edgecolor="k")
    column_label = COLUMN_METADATA[Column.VectorField.PEARSON_R].label or str(
        Column.VectorField.PEARSON_R
    )
    ax.set_xlabel(column_label)
    ax.set_ylabel("Count")
    # make sure y ticks are integers since this is a count histogram
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    filename = "fpt_hist"
    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )
    return output_path / f"{filename}.svg"


@figure_panel(
    "Make panel of histogram of first passage time distances from the fitted lines across datasets."
)
def make_first_passage_time_distance_to_linefit_hist(
    figure_size: tuple[float, float],
    output_path: Path,
    dataset_names: list[str],
    weighted: bool = True,
) -> Path:
    fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME)
    line_fit_df, fpt_stats_df_no_nan = get_line_fit_and_filtered_df(fpt_manifest, dataset_names)

    distances_all = []
    for dataset, df_dataset in line_fit_df.groupby(Column.DATASET):
        odr_result = df_dataset[Column.VectorField.ODR_RESULT].iloc[0]
        if weighted:
            # each (x,y) point that was passed to odr_fit to get the line fit has
            # an associated point on that line (that is affected by the weights
            # that are passed to odr_fit) and the distance between that point
            # and the original (x,y) point is delta and eps. We use these
            # distances as the values for a histogram
            weighted_distances = np.sqrt(odr_result.delta**2 + odr_result.eps**2)
            distances_all.extend(weighted_distances)
        else:
            # m, b = odr_result.beta  # slope and intercept of the fitted line
            # line_func = lambda x, m=m, b=b: m * x + b
            # get the MFPT points that are in the dataset and calculate their
            # distances from the fitted line
            fpt_stats_df_sub = fpt_stats_df_no_nan[fpt_stats_df_no_nan[Column.DATASET] == dataset]

            fpt_col_grid = f"mean{Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX}_grid_based"
            fpt_col_cell = f"mean{Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX}_cell_centered"
            mfpt_points = fpt_stats_df_sub[[fpt_col_grid, fpt_col_cell]]

            lob_points = np.stack([odr_result.xplusd, odr_result.yest], axis=-1)
            point1, point2 = lob_points[0:1], lob_points[-2:-1]

            vector_cross_prod = np.expand_dims(
                np.cross(point2 - point1, point1 - mfpt_points, axis=-1), axis=-1
            )
            p1p2_dist = np.linalg.norm(point2 - point1, axis=-1, keepdims=True)
            shortest_dist_to_line = np.linalg.norm(vector_cross_prod / p1p2_dist, axis=-1)
            distances_all.extend(shortest_dist_to_line)

    n_size = len(distances_all)

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    biggest_distance_as_int = int(max(np.ceil(distances_all)))
    bins: list[float] = np.arange(0, biggest_distance_as_int + 1, 0.5, dtype=float).tolist()
    ax.hist(distances_all, bins=bins, density=True, edgecolor="k")
    column_label = (
        "Deviation of MFPTs from linear\nfit for grid vs. cell-centered\ntrajectories (hrs)"
    )
    ax.set_xlabel(column_label)
    ax.set_ylabel("Probability\ndensity")
    ax.xaxis.set_major_locator(MultipleLocator(base=2))
    ax.xaxis.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(base=1))
    ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
    ax.yaxis.minorticks_on()
    ax.yaxis.set_minor_locator(MultipleLocator(base=0.1))
    ax.annotate(f"n = {n_size}", xy=(0.98, 0.72), xycoords="axes fraction", ha="right")

    filename = "fpt_hist"
    save_plot_to_path(
        fig,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )
    return output_path / f"{filename}.svg"
