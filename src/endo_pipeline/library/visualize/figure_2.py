"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_1d,
    plot_drift_contours,
)
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_3d import (
    NORMALIZE_QUIVER_VECTORS,
    QUIVER_COLORMAP,
    QUIVER_DOWNSAMPLE_FACTOR,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
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


def _get_example_points_along_nullcline(
    nullcline_coords: dict[Column.DiffAEData, pd.DataFrame],
    stable_fixed_point: np.ndarray,
    num_points: int = 5,
) -> dict[Column.DiffAEData, np.ndarray]:
    """
    Get coordinates of example points along each nullcline, selected to be
    approximately equally spaced by arc length along the nullcline.

    Parameters
    ----------
    nullcline_coords
        Dictionary mapping each column name to a DataFrame containing the (r,
        rho) coordinates of the corresponding nullcline.
    stable_fixed_point
        Coordinates of the stable fixed point, used as a reference point for
        selecting example points along the nullcline.
    num_points
        Number of example points to select along each nullcline.

    Returns
    -------
    :
        Dictionary mapping each column name to an array of shape (2, num_points)
        containing the (r, rho) coordinates of the selected example points
        along the corresponding nullcline.

    """
    example_points = {}
    for column, coords_dataframe in nullcline_coords.items():
        path_coords = (
            coords_dataframe[[Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]]
            .to_numpy()
            .T
        )

        # calculate arc length along the nullcline path, inserting a 0 at the
        # beginning for the first point
        arc_length = np.cumsum(np.sqrt(np.sum(np.diff(path_coords, axis=1) ** 2, axis=0)))
        arc_length = np.insert(arc_length, 0, 0)

        # starting with the fixed point as reference, find 4 additional points
        # (on either side of the fixed point if possible) along the nullcline
        # that are equally spaced by arc length
        fixed_point_index = np.argmin(
            np.sqrt(
                np.sum((path_coords - stable_fixed_point.flatten()[:, np.newaxis]) ** 2, axis=0)
            )
        )
        total_length = arc_length[-1]
        example_arc_lengths = np.linspace(
            max(0, arc_length[fixed_point_index] - total_length / 2),
            min(total_length, arc_length[fixed_point_index] + total_length / 2),
            num_points,
        )
        example_indices = [
            np.argmin(np.abs(arc_length - example_arc_length))
            for example_arc_length in example_arc_lengths
        ]
        # get coordinates of the selected example points
        example_points[column] = path_coords[:, example_indices]

        # replace the point closest to the stable fixed point with the stable
        # fixed point
        closest_index = np.argmin(
            np.sqrt(
                np.sum(
                    (example_points[column] - stable_fixed_point.flatten()[:, np.newaxis]) ** 2,
                    axis=0,
                )
            )
        )
        example_points[column][:, closest_index] = stable_fixed_point.flatten()

    return example_points


def make_2d_contour_plot_panel(
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
    gridspec_kwargs: dict | None,
    xlabel_kwargs: dict | None,
    ylabel_kwargs: dict | None,
    axes_title_kwargs: dict | None,
    include_colorbar: bool = True,
    include_legend: bool = True,
) -> tuple[Path, dict[Column.DiffAEData, np.ndarray]]:
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
    # get (r, rho) coordinates of r- and rho-nullclines for generating images
    axes = cast(np.ndarray[plt.Axes, Any], axes_)
    nullcline_coords_ = _get_nullcline_coords_from_contour_axes(
        axes, column_names, r_lims, rho_lims
    )

    # get 5 equally space (by arc length) coordinates along the nullcline to
    # plot as points on top of the contour, including the stable fixed point and
    # 2 points on either side if possible
    nullcline_coords = _get_example_points_along_nullcline(nullcline_coords_, stable_fixed_point)

    for ax_index, ax_ in enumerate(list(axes_)):
        # add nullcline points on top of the contour plot
        ax_.plot(
            nullcline_coords[column_names[ax_index]][0],
            nullcline_coords[column_names[ax_index]][1],
            "o",
            color="w",
            markeredgecolor="k",
            markeredgewidth=0.25,
            markersize=3,
        )
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
        # shrink the constrained-layout region so the inset colorbar axes
        # (which lives outside the main axes boundary) is not clipped on save

        layout_engine = fig.get_layout_engine()
        if isinstance(layout_engine, ConstrainedLayoutEngine):
            layout_engine.set(rect=(0, 0, 0.9, 1))

    handles = []
    labels = []
    if include_legend:
        # plot_drift_contours draws nullclines via ax.contour(), which does not
        # produce labeled artists. Add proxy Line2D handles so the legend has
        # something to show.
        nullcline_styles = (nullcline_r_style, nullcline_rho_style)
        for col_idx, col in enumerate(column_names):
            label = COLUMN_METADATA[col].label or str(col)
            handle = mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=nullcline_styles[col_idx],
                label=f"Nullcline d{label}/dt=0",
            )
            handles.append(handle)
            labels.append(f"Nullcline d{label}/dt=0")
        fig.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            handletextpad=0.3,
        )

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


def make_1d_drift_plot_panel(
    drift: np.ndarray,
    theta_values: np.ndarray,
    column_label: str,
    stable_fixed_point: float,
    figsize: tuple[float, float],
    fig_savedir: Path,
    filename: str,
    axes_xlim: tuple[float, float],
    axes_ylim: tuple[float, float],
    axes_xticks: list[float],
    axes_xtick_labels: list[str],
    axes_yticks: list[float],
    arrow_scale: float,
    arrow_width: float,
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
        flow_arrow_kwargs={"color": "dimgrey", "scale": arrow_scale, "width": arrow_width},
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

    # set plot formatting args
    ax.set_box_aspect(1.0)
    ax.set_xticks(axes_xticks, labels=axes_xtick_labels)
    ax.set_yticks(axes_yticks)

    save_plot_to_path(
        fig, fig_savedir, filename, file_format=".svg", tight_layout=False, transparent=True
    )

    return fig_savedir / f"{filename}.svg"


def reconstruct_along_nullcline(
    nullcline_coords: dict[ColumnNameType, np.ndarray],
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
        r_coords = coords_dataframe[0]
        rho_coords = coords_dataframe[1]
        full_coords_dataframe = pd.DataFrame(
            {
                Column.DiffAEData.POLAR_RADIUS: r_coords,
                Column.DiffAEData.PC3_FLIPPED: rho_coords,
                Column.DiffAEData.POLAR_ANGLE: theta_value * np.ones_like(r_coords),
            }
        )

        walk_array = generate_from_dataframe(
            full_coords_dataframe,
            column_names,
            model,
            num_gpus=num_gpus,
            random_seed=5,
        )
        fig_null_walk = make_contact_sheet(
            panels=[walk_array[i] for i in range(len(walk_array))],
            max_rows=1,
            max_cols=len(walk_array),
            fig_kwargs={"figsize": (3.125, 0.625), "layout": "constrained"},
            gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
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


def make_3d_vector_field_plot_panel(
    dataset_name: str,
    fig_savedir: Path,
    filename: str,
    downsample_factor: int = QUIVER_DOWNSAMPLE_FACTOR,
    normalize_vectors: bool = NORMALIZE_QUIVER_VECTORS,
    colorscale: str = QUIVER_COLORMAP,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset using
    Plotly, with the stable fixed point overlaid as a scatter marker.

    The drift vector field is loaded via
    :func:`~endo_pipeline.library.analyze.vector_field_estimation.load_drift_dataframe_for_dataset`
    and the stable fixed point is loaded from the bootstrapped fixed-point
    dataframe manifest (``bootstrapped_fixed_points_grid``).

    Parameters
    ----------
    dataset_name
        Name of the dataset to visualize.
    fig_savedir
        Directory in which to save the figure as an interactive HTML file and a static SVG file.
    filename
        Filename stem (without extension) for the saved HTML and SVG files.
    downsample_factor
        Factor by which to downsample the grid before plotting, to keep the
        interactive figure responsive.
    normalize_vectors
        Whether to normalize each cone vector to unit length before plotting.
        When ``True`` all arrows have the same length and are coloured by the
        original vector magnitude instead.
    colorscale
        Plotly-compatible colorscale name used to colour the cones by vector
        magnitude.

    Returns
    -------
    :
        Path to the saved SVG file containing the 3D vector field and the stable
        fixed point.

    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]

    # ------------------------------------------------------------------
    # Load drift vector field
    # ------------------------------------------------------------------
    drift_df = load_drift_dataframe_for_dataset(dataset_name)
    if drift_df.empty:
        raise ValueError(
            f"No drift dataframe found for dataset '{dataset_name}'. "
            "Cannot create 3D vector field plot."
        )
    flow_field_dict = get_vector_field_as_dict_from_dataframe(drift_df, column_names)

    # grids and vectors are 3-D arrays shaped (n_theta, n_r, n_rho)
    x_grid, y_grid, z_grid = flow_field_dict["grid"]
    u_field, v_field, w_field = flow_field_dict["vectors"]

    # ------------------------------------------------------------------
    # Downsample uniformly along every axis
    # ------------------------------------------------------------------
    d = downsample_factor
    x_ds = x_grid[::d, ::d, ::d]
    y_ds = y_grid[::d, ::d, ::d]
    z_ds = z_grid[::d, ::d, ::d]
    u_ds = u_field[::d, ::d, ::d]
    v_ds = v_field[::d, ::d, ::d]
    w_ds = w_field[::d, ::d, ::d]

    # ------------------------------------------------------------------
    # Compute vector magnitudes for colouring
    # ------------------------------------------------------------------
    magnitude = np.sqrt(u_ds**2 + v_ds**2 + w_ds**2)

    if normalize_vectors:
        # avoid division by zero at zero-magnitude grid points
        safe_mag = np.where(magnitude == 0, 1.0, magnitude)
        u_plot = u_ds / safe_mag
        v_plot = v_ds / safe_mag
        w_plot = w_ds / safe_mag
    else:
        u_plot = u_ds
        v_plot = v_ds
        w_plot = w_ds

    # flatten for Plotly
    x_flat = x_ds.ravel()
    y_flat = y_ds.ravel()
    z_flat = z_ds.ravel()
    u_flat = u_plot.ravel()
    v_flat = v_plot.ravel()
    w_flat = w_plot.ravel()
    mag_flat = magnitude.ravel()

    # ------------------------------------------------------------------
    # Build Plotly traces
    # ------------------------------------------------------------------
    cone_trace = go.Cone(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        u=u_flat,
        v=v_flat,
        w=w_flat,
        colorscale=colorscale,
        cmin=float(mag_flat.min()),
        cmax=float(mag_flat.max()),
        colorbar={"title": "drift magnitude"},
        sizemode="scaled",
        sizeref=0.5,
        showscale=True,
        opacity=0.8,
        name="drift field",
    )

    traces: list = [cone_trace]

    # ------------------------------------------------------------------
    # Load and overlay stable fixed point
    # ------------------------------------------------------------------
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    if not fixed_points_df.empty and Column.VectorField.STABILITY in fixed_points_df.columns:
        stable_df = fixed_points_df[
            fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.STABLE
        ]
        if not stable_df.empty:
            fpt_coords = stable_df[column_names].to_numpy()
            stable_trace = go.Scatter3d(
                x=fpt_coords[:, 0],
                y=fpt_coords[:, 1],
                z=fpt_coords[:, 2],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "size": 8,
                    "color": FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
                    "line": {"color": "black", "width": 1},
                },
                name="stable fixed point",
            )
            traces.append(stable_trace)

    # ------------------------------------------------------------------
    # Compose figure
    # ------------------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3D drift vector field - {dataset_name}",
        scene={
            "xaxis_title": col_labels[0],
            "yaxis_title": col_labels[1],
            "zaxis_title": col_labels[2],
        },
        legend={"x": 0.01, "y": 0.99},
    )

    # ------------------------------------------------------------------
    # Save as interactive HTML and static SVG
    # ------------------------------------------------------------------
    out_path = Path(fig_savedir) / f"{filename}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))

    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )

    return fig_savedir / f"{filename}.svg"
