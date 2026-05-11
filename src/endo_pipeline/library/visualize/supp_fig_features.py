"""Methods for building panels in the supplementary figure showing feature correlations and coordinate transformations."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_start_of_steady_state_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe, load_model, save_plot_to_path
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.pca import fit_pca
from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_latent_walk_images
from endo_pipeline.library.model.latent_walk_utils import get_latent_walk
from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.dynamics_workflows import TIME_STEP_IN_HOURS
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import (
    FONTSIZE_LARGE,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
    MAX_FIGURE_WIDTH,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    RANDOM_SEED,
)


def perform_latent_walk_along_top_pcs(
    save_path: Path,
    filename: str,
    figsize: tuple[float, float] = (MAX_FIGURE_WIDTH, 2.8),
    num_gpus: int | None = None,
) -> np.ndarray:
    """
    Perform a latent walk along the top principal 3 components of the data.

    This method acts as a wrapper for the latent walk generation and plotting
    functions. It uses the default model and dataset manifests to load the
    necessary model and data, performs the latent walk along the top 3 PCs,
    generates the corresponding images, and saves a contact sheet of the walk.

    Parameters
    ----------
    save_path
        Directory path to save the output figure.
    filename
        Name of the output figure file.
    figsize
        Figure size to use for the output figure.

    Returns
    -------
    :
        Array of shape (3, num_steps, h, w) containing the reconstructed image
        crops.

    """
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)
    model = load_model(model_manifest.locations[run_name], instantiate=True)
    if not isinstance(model, DiffusionAutoEncoder):
        raise ValueError(
            f"Model loaded from {model_manifest_name} with run name {run_name} is not a DiffusionAutoEncoder."
        )

    # set up output directory

    # load model configuration and reference dataset manifests
    dataframe_manifest_name = f"{model_manifest.name}_{run_name}_grid_pca_filtered"
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    num_pcs = 3
    walk_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    pca = fit_pca(num_pcs=num_pcs)
    dataframe_all_datasets = pd.concat(
        [
            load_dataframe(get_dataframe_location_for_dataset(dataframe_manifest, dataset_name))
            for dataset_name in dataset_names
        ]
    )
    data_for_walk = dataframe_all_datasets[walk_column_names]

    # get coordinate values for latent walk and the ranges of the walk for each
    # dimension
    walk, ranges = get_latent_walk(
        data_for_walk,
        walk_column_names,
        sigma=3,
        n_steps=7,
    )

    walk_latent = pca.inverse_transform(walk[walk_column_names].to_numpy())

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(
        model, walk_latent, ranges, num_gpus=num_gpus, random_seed=RANDOM_SEED
    )

    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        walk_column_names,
        save_path,
        filename,
        label_sigmas=True,
        figsize=figsize,
        file_format=".svg",
    )

    return walk_img_grid


def _add_axes_lines(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes, Any],
    center_index: int,
    n_steps: int,
    color: str = "blue",
    linewidth: float = 2.0,
    head_length: float = 0.75,
    head_width: float = 0.4,
    mutation_scale: float = 15,
    axes_extend: float = 0.14,
    elongation_label_offsets: tuple[float, float] = (0.2, 0.05),
) -> None:
    """Add horizontal and vertical axes lines with labels to the 2D latent walk plot."""
    ax_center = axes[center_index, center_index]
    center_bbox = ax_center.get_position()
    left_bbox = axes[center_index, 0].get_position()
    right_bbox = axes[center_index, n_steps - 1].get_position()
    top_bbox = axes[0, center_index].get_position()
    bottom_bbox = axes[n_steps - 1, center_index].get_position()

    cx = center_bbox.x0 + center_bbox.width / 2
    cy = center_bbox.y0 + center_bbox.height / 2

    arrowstyle = f"<->,head_length={head_length},head_width={head_width}"

    for posA, posB in [
        ((left_bbox.x0 - axes_extend, cy), (right_bbox.x1 + axes_extend, cy)),  # horizontal
        ((cx, bottom_bbox.y0 - axes_extend), (cx, top_bbox.y1 + axes_extend)),  # vertical
    ]:
        # mutation_scale must be set explicitly (default=1 leads to sub-pixel
        # arrowheads that render as empty paths in SVG).
        arrow = FancyArrowPatch(
            posA,
            posB,
            arrowstyle=arrowstyle,
            connectionstyle="arc3,rad=0",
            color=color,
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            transform=fig.transFigure,
            clip_on=False,
            zorder=4,
        )
        fig.add_artist(arrow)

    top_ax = axes[0, center_index]
    top_bbox = top_ax.get_position()
    pc1_label = cast(str, COLUMN_METADATA["pc_1"].label)
    pc2_label = cast(str, COLUMN_METADATA["pc_2"].label)
    # PC2 label: to the left of the top image in the PC2 column
    fig.text(
        top_bbox.x0 + 0.04,
        top_bbox.y1 + 0.1,
        pc2_label,
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        ha="right",
        va="top",
        transform=fig.transFigure,
    )
    # PC1 label: to the right of the rightmost image, vertically centred on the axis
    fig.text(
        right_bbox.x1 + axes_extend - 0.085,
        cy - 0.09,
        pc1_label,
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        ha="left",
        va="center",
        transform=fig.transFigure,
    )

    # "elongation" label: to the left of the bottom axis arrow tip
    fig.text(
        left_bbox.x0 + elongation_label_offsets[0],
        bottom_bbox.y0 - axes_extend + elongation_label_offsets[1],
        "elongation",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        ha="right",
        va="top",
        transform=fig.transFigure,
    )


def _add_orientation_arrow(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes, Any],
    center_index: int,
    n_steps: int,
    arc_rad: float = 0.5,
    head_length: float = 0.75,
    head_width: float = 0.4,
    color: str = "darkred",
    linewidth: float = 2.0,
    label_offset: tuple[float, float] = (0.075, 0.125),
) -> None:
    """Add an arced arrow and "orientation" label to the 2D latent walk plot."""
    overlay = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor="none", zorder=5)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    # arced arrow from rightmost image to topmost image with "orientation" label
    # at the midpoint of the arc
    rightmost_bbox = axes[center_index, n_steps - 1].get_position()
    topmost_bbox = axes[0, center_index].get_position()
    arrow_start = (
        rightmost_bbox.x1 - rightmost_bbox.width / 4,
        rightmost_bbox.y1,
    )
    arrow_end = (
        topmost_bbox.x1,
        topmost_bbox.y1 - topmost_bbox.height / 4,
    )
    arrowstyle = f"->,head_length={head_length},head_width={head_width}"
    connectionstyle = f"arc3,rad={arc_rad}"
    overlay.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": arrowstyle,
            "color": color,
            "lw": linewidth,
            "connectionstyle": connectionstyle,
        },
    )
    mid_x = (arrow_start[0] + arrow_end[0]) / 2
    mid_y = (arrow_start[1] + arrow_end[1]) / 2
    overlay.text(
        mid_x + label_offset[0],
        mid_y + label_offset[1],
        "orientation",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        ha="center",
        va="center",
        transform=overlay.transAxes,
    )


def plot_2d_latent_walk(
    images_pc1: np.ndarray,
    images_pc2: np.ndarray,
    save_path: Path,
    filename: str,
) -> Path:
    """
    Plot a "2D" latent walk along the first two principal components by
    arranging the images from the walks along PC 1 and PC 2 in a grid.

    The walk along PC 1 is arranged horizontally with PC 2 = 0, and the walk
    along PC 2 is arranged vertically with PC 1 = 0. The center image at the
    origin (0 sigma) is shared by both walks.

    Parameters
    ----------
    images_pc1
        Array of shape (num_steps, h, w) containing the reconstructed image
        crops for the walk along PC 1.
    images_pc2
        Array of shape (num_steps, h, w) containing the reconstructed image
        crops for the walk along PC 2.
    save_path
        Directory path to save the output figure.
    filename
        Name of the output figure file.

    Returns
    -------
    :
        Path to the saved figure file.

    """
    n_steps = images_pc1.shape[0]
    center = n_steps // 2  # index of the origin (0 sigma)

    fig, axes = plt.subplots(
        n_steps,
        n_steps,
        gridspec_kw={"wspace": 0.15, "hspace": 0.15},
        figsize=(2.1, 2.1),
        layout="constrained",
    )

    # Inset the subplot grid so the axis arrows that extend beyond the outermost
    # cells always land inside the figure canvas.  Setting rect here — before the
    # first draw — also prevents the constrained layout engine from shifting
    # subplot positions when savefig runs a second layout pass.
    # The bottom margin must be large enough to accommodate both the axis
    # arrow tips (which extend `axes_extend` below the grid) and the
    # "elongation" text label below that.
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(**{"rect": (0.10, 0.10, 0.78, 0.75)})

    for row in range(n_steps):
        for col in range(n_steps):
            ax: plt.Axes = axes[row, col]
            ax.axis("off")
            ax.set_zorder(5)
            ax.patch.set_visible(False)
            if row == center and col == center:
                # origin: use the center image (shared by both walks)
                ax.imshow(images_pc1[center], cmap="gray", zorder=1)
            elif row == center:
                # center row: PC1 walk (vary PC1, PC2 = 0)
                ax.imshow(images_pc1[col], cmap="gray", zorder=1)
            elif col == center:
                # center column: PC2 walk (vary PC2, PC1 = 0)
                # flip row index so PC2 increases upward
                ax.imshow(images_pc2[n_steps - 1 - row], cmap="gray", zorder=1)

    fig.canvas.draw()

    # Draw axis lines on a figure-level underlay so they appear behind all image axes.
    _add_axes_lines(
        fig,
        axes,
        center,
        n_steps,
        color="blue",
        linewidth=2.0,
        head_length=0.5,
        head_width=0.3,
        mutation_scale=15,
        axes_extend=0.14,
        elongation_label_offsets=(0.3, 0.075),
    )

    # Add arced arrow with label "orientation" going from PC1 to PC2
    _add_orientation_arrow(
        fig,
        axes,
        center,
        n_steps,
        arc_rad=0.5,
        head_length=0.75,
        head_width=0.4,
        color="darkred",
        linewidth=2.0,
        label_offset=(0.175, 0.115),
    )

    save_plot_to_path(
        fig, save_path, filename, file_format=".svg", transparent=True, tight_layout=False
    )
    return save_path / f"{filename}.svg"


def _make_feature_pair_histogram_panel(
    columns_to_plot: list[ColumnNameType],
    axes_ylim: tuple[float, float] | None = None,
    axes_yticks: list[float] | None = None,
    axes_ytick_labels: list[str] | None = None,
    shared_y_axis: bool = True,
    histogram_vmin: float = 0.0,
    histogram_vmax: float = 0.7,
    figsize: tuple[float, float] = (2.9, 2.45),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Build a 2x2 grid of 2-D histograms (time x feature) for two datasets and
    two feature columns.

    Rows correspond to low- and high-flow datasets; columns correspond to the
    two entries in *columns_to_plot*.

    Parameters
    ----------
    columns_to_plot
        Exactly two column names to compare side-by-side (left then right).
    axes_ylim
        Shared y-axis limits applied to every subplot.  ``None`` lets
        Matplotlib auto-scale each axis independently.
    axes_yticks
        Shared y-axis tick positions.  ``None`` uses Matplotlib defaults.
    axes_ytick_labels
        Tick label strings corresponding to *axes_yticks*.  ``None`` uses
        default numeric labels.
    shared_y_axis
        When ``True`` the two feature columns share the same y range, so tick
        labels are shown only on the left column.  When ``False`` each column
        keeps its own tick labels.
    histogram_vmin
        Lower colour-scale limit for the 2-D histogram density.
    histogram_vmax
        Upper colour-scale limit for the 2-D histogram density.
    figsize
        Figure size passed to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    :
        The :class:`~matplotlib.figure.Figure` and its 2x2 :class:`~numpy.ndarray`
        of :class:`~matplotlib.axes.Axes`, ready for further customisation or
        saving.
    """
    if len(columns_to_plot) != 2:
        raise ValueError(
            f"columns_to_plot must contain exactly 2 columns, got {len(columns_to_plot)}."
        )

    dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
    dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]

    dataframe_manifest = load_dataframe_manifest(
        DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED
    )

    start_imaging_line_color = "limegreen"
    steady_state_line_color = "darkturquoise"

    axes_xlim = (0, 50)  # in hours, after converting from frames
    axes_xticks = [0, 12, 24, 36, 48]
    axes_xtick_labels = [f"{x}" for x in axes_xticks]

    fig, ax = plt.subplots(2, 2, figsize=figsize, layout="constrained", gridspec_kw={"hspace": 0.1})

    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        # reserve left margin for the vertical label and top margin for the legend
        layout_engine.set(**{"rect": [0.08, 0, 1, 0.94]})

    time_column_label = "Time under flow (hours)"
    # convert frames to hours for better readability of x-axis
    time_conversion_factor = TIME_STEP_IN_HOURS

    columns_to_compute = [*columns_to_plot, Column.TIMEPOINT]

    for i, dataset in enumerate([dataset_low, dataset_high]):
        dataset_config = load_dataset_config(dataset)
        frames_before_imaging = abs(dataset_config.flow_conditions[0].start)
        shear_stress = np.ceil(max(fc.shear_stress for fc in dataset_config.flow_conditions))
        shear_stress_label = f"{shear_stress} dyn/cm{Unicode.SQUARED}"

        # use position 0 as a representative position for the dataset to get the
        # timepoint corresponding to the start of steady state, which we will
        # indicate with a vertical dashed line on the histogram
        start_steady_state_timepoint = (
            get_start_of_steady_state_for_position(dataset_config, position=0) or 0
        )
        # shift so that time = 0 corresponds to the start of flow, and convert
        # from frames to hours
        start_steady_state_timepoint += frames_before_imaging
        start_steady_state_timepoint_hrs = start_steady_state_timepoint * time_conversion_factor

        df_ = load_dataframe(
            get_dataframe_location_for_dataset(dataframe_manifest, dataset), delay=True
        )
        df: pd.DataFrame = df_[columns_to_compute].compute()
        # shift timepoints so that time = 0 corresponds to the start of flow
        df[Column.TIMEPOINT] = df[Column.TIMEPOINT] + frames_before_imaging

        time_bins = get_bins(bin_widths=(12,), data=df[Column.TIMEPOINT].to_numpy())[0][0]
        time_bins = time_bins * time_conversion_factor

        for j, column in enumerate(columns_to_plot):
            feature_column_label = COLUMN_METADATA[column].name or cast(str, column)
            # convert to sentence case for better readability as a plot title
            feature_column_label = feature_column_label.capitalize()

            ax_ij = cast(plt.Axes, ax[i, j])

            feature_bins = get_bins(bin_widths=(0.05,), data=df[column].to_numpy())[0][0]
            ax_ij.hist2d(
                df[Column.TIMEPOINT] * time_conversion_factor,
                df[column],
                bins=[time_bins, feature_bins],
                cmap="inferno",
                density=True,
                cmin=histogram_vmin,
                cmax=histogram_vmax,
            )

            # change the background color to grey
            ax_ij.set_facecolor("grey")

            # draw dashed line at start of imaging (time =
            # -frames_before_imaging)
            ax_ij.axvline(
                x=frames_before_imaging * time_conversion_factor,
                color=start_imaging_line_color,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                label="Start of imaging",
            )

            # draw dashed line at start of steady state
            ax_ij.axvline(
                x=start_steady_state_timepoint_hrs,
                color=steady_state_line_color,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                label="Start of steady state",
            )

            # set axes limits and ticks
            ax_ij.set_xlim(axes_xlim)
            ax_ij.set_xticks(axes_xticks)
            if axes_ylim is not None:
                ax_ij.set_ylim(axes_ylim)
            if axes_yticks is not None:
                ax_ij.set_yticks(axes_yticks)
            if j == 0:
                # add tick labels for the left column
                if axes_ytick_labels is not None:
                    ax_ij.set_yticklabels(axes_ytick_labels, fontsize=FONTSIZE_XSMALL)
                else:
                    ax_ij.tick_params(axis="y", labelsize=FONTSIZE_XSMALL)
            elif j == 1:
                if shared_y_axis:
                    # omit redundant y tick labels when both columns share the same range
                    ax_ij.set_yticklabels([])
                else:
                    ax_ij.tick_params(axis="y", labelsize=FONTSIZE_XSMALL)
            if i == 0:
                # put label as column title for top row
                ax_ij.set_title(feature_column_label, fontsize=FONTSIZE_XSMALL)
                ax_ij.set_xticklabels([])  # no x tick labels for top row
            elif i == 1:
                # only set x-axis tick labels and label for bottom row
                ax_ij.set_xticklabels(axes_xtick_labels, fontsize=FONTSIZE_XSMALL)
                ax_ij.set_xlabel(time_column_label, labelpad=1, fontsize=FONTSIZE_XSMALL)

        # add vertical label for shear stress to the left of the contour plot
        # (y positions reflect the constrained-layout rect top of 0.94)
        y_position = 0.72 if i == 0 else 0.31
        fig.text(
            0.05,
            y_position,
            shear_stress_label,
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=FONTSIZE_MEDIUM,
            fontweight="bold",
        )

    # add legend for the vertical dashed line indicating the start of steady state
    # and grey background indicating cutoff for cell piling (no data);
    # placed in the reserved top margin using figure coordinates so that
    # constrained layout is not affected
    handles, labels = ax_ij.get_legend_handles_labels()
    handles.append(Patch(facecolor="grey", edgecolor="none"))
    labels.append("No data")
    fig.legend(
        handles,
        labels,
        fontsize="xx-small",
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        ncol=3,
        handletextpad=0.3,
        borderpad=0.3,
        columnspacing=0.8,
        frameon=False,
    )

    # set the color limits to be the same across all histograms
    # plot adjacent to the right of the rightmost histogram, spanning both rows
    cbar_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=histogram_vmin, vmax=histogram_vmax), cmap="inferno"
    )
    cbar = fig.colorbar(cbar_mappable, ax=ax[:, 1], location="right", pad=0.1)
    cbar.set_label("Histogram", labelpad=3, fontsize=FONTSIZE_SMALL)

    return fig, ax


def make_theta_orientation_histogram_panel(output_path: Path) -> Path:
    """
    Make the panel showing the histogram over time of theta (patch-based ML
    feature) side by side with orientation (cell-based segmentation feature).
    """
    axes_ylim = (0, np.pi)
    axes_yticks = [0, np.pi / 2, np.pi]
    axes_ytick_labels = [f"0={Unicode.PI}", f"{Unicode.PI}/2", f"{Unicode.PI}=0"]

    fig, _ = _make_feature_pair_histogram_panel(
        columns_to_plot=[Column.DiffAEData.POLAR_ANGLE, Column.SegData.ORIENTATION],
        axes_ylim=axes_ylim,
        axes_yticks=axes_yticks,
        axes_ytick_labels=axes_ytick_labels,
        shared_y_axis=True,
    )
    filename = "theta_orientation_histograms"
    save_plot_to_path(
        fig, output_path, filename, file_format=".svg", tight_layout=False, transparent=False
    )
    return output_path / f"{filename}.svg"


def make_r_aspect_ratio_histogram_panel(output_path: Path) -> Path:
    """
    Make the panel showing the histogram over time of polar radius r
    (patch-based ML feature) side by side with cell aspect ratio (cell-based
    segmentation feature).

    Both features capture cell elongation: *r* encodes the magnitude of the
    alignment signal in the DiffAE latent space while aspect ratio is a direct
    morphological measurement.  Placing them side by side reveals how the two
    representations co-vary across the time course and across shear-stress
    conditions.

    The two features have different natural ranges, so each column (feature)
    is given its own consistent y-axis limits derived from the union of both
    dataset rows rather than using a shared range.
    """
    fig, ax = _make_feature_pair_histogram_panel(
        columns_to_plot=[Column.DiffAEData.POLAR_RADIUS, Column.SegData.ASPECT_RATIO],
        axes_ylim=None,
        axes_yticks=None,
        axes_ytick_labels=None,
        shared_y_axis=False,
    )

    # datasets have different natural ranges for r and aspect ratio, so set
    # y-limits per dataset row rather than per feature column;
    r_ylim_dataset_1 = (0.0, 2.3)
    aspect_ratio_ylim_dataset_1 = (1.0, 5.95)
    r_ylim_dataset_2 = (0.0, 2.8)
    aspect_ratio_ylim_dataset_2 = (1.0, 7.25)
    # Reconcile y-limits per feature column so that both dataset rows (low-flow
    # and high-flow) share the same scale for that feature.
    num_columns = ax.shape[1]
    for j in range(num_columns):
        for i in range(ax.shape[0]):
            ax_ij = cast(plt.Axes, ax[i, j])
            if j == 0:
                # r column
                if i == 0:
                    ax_ij.set_ylim(r_ylim_dataset_1)
                elif i == 1:
                    ax_ij.set_ylim(r_ylim_dataset_2)
            elif j == 1:
                # aspect ratio column
                if i == 0:
                    ax_ij.set_ylim(aspect_ratio_ylim_dataset_1)
                elif i == 1:
                    ax_ij.set_ylim(aspect_ratio_ylim_dataset_2)

    filename = "r_aspect_ratio_histograms"
    save_plot_to_path(
        fig, output_path, filename, file_format=".svg", tight_layout=False, transparent=False
    )
    return output_path / f"{filename}.svg"
