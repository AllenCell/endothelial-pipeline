"""Methods for visualizing the outputs of the DiffAE feature analysis workflows."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_histogram,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
)
from endo_pipeline.library.visualize.figure_utils import set_axes_properties
from endo_pipeline.library.visualize.fixed_points import StabilityLegendHandle
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    KERNEL_PERIODS_DYNAMICS,
    POLAR_ANGLE_PERIOD,
)
from endo_pipeline.settings.figures import FONTSIZE_XSMALL
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_LEVELS,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE


def plot_drift_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, Sequence[plt.Axes]] | None = None,
    figsize: tuple[float, float] = (7, 12),
    n_rows: int = 2,
    n_cols: int = 1,
    axes_limits: list[tuple[float, float]] | None = None,
    axes_aspect: Literal["auto", "equal"] | float | None = "equal",
    axes_titles: tuple[str, str] | None = None,
    colormap: str = DRIFT_CONTOUR_COLORMAP,
    vmin: float | None = DRIFT_CONTOUR_VMIN,
    vmax: float | None = DRIFT_CONTOUR_VMAX,
    num_levels: int = DRIFT_CONTOUR_LEVELS,
    include_colorbar: bool = True,
    cbar_num_ticks: int = DRIFT_CONTOUR_CBAR_NUM_TICKS,
    cbar_tick_round: int = DRIFT_CONTOUR_CBAR_ROUND,
    include_nullclines: bool = True,
    nullcline_styles: tuple = ("dashed", "dashdot"),
    nullcline_colors: tuple = ("k", "k"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.7,
    gridspec_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
    axes_title_kwargs: dict | None = None,
    axes_rects: list[tuple[float, float, float, float]] | None = None,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Make and save contour plot of each component of the drift vector field over
    the 2D state space.

    The contour lines are colored according to the value of the drift component,
    using a diverging colormap centered at zero to visualize the direction and
    magnitude of the drift.

    **Axes specification**

    When ``axes_rects`` is provided, the figure is built without a layout engine
    via ``plt.figure`` and each subplot is added with ``fig.add_axes``. Callers
    can then place additional axes (colorbar, legend) at fixed figure
    coordinates without risk of clipping.  When ``None`` (default) and
    ``fig_ax`` is also ``None``, ``plt.subplots`` with ``layout="constrained"``
    is used instead.

    Parameters
    ----------
    meshgrid
        Meshgrid on which the drift is evaluated, typically obtained from
        np.meshgrid(..., indexing="ij").
    drift
        Drift vector field evaluated on the meshgrid, with shape (nx, ny, ndim).
    variable_labels
        Labels for axes corresponding to the state space variables, e.g.,
        ["$x_1$", "$x_2$"].
    fig_ax
        Optional tuple of figure and axes objects to plot on. If None, a new
        figure and axes will be created; if provided, the contour plots will be
        made on the provided axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    axes_aspect
        Aspect ratio for the axes, e.g., "equal" to make x and y have equal
        scaling.
    axes_titles
        Optional tuple of titles for each subplot.
    colormap
        Colormap to use for the contour plots.
    vmin
        Optional, minimum colorbar value for the contour plots.
    vmax
        Optional, maximum colorbar value for the contour plots.
    num_levels
        Number of contour levels to use in the plot.
    include_colorbar
        Whether to include a colorbar for each contour plot.
    cbar_num_ticks
        Number of ticks to use in the colorbar for each contour plot.
    cbar_tick_round
        Number of decimal places to round colorbar ticks to in the contour
        plots.
    nullcline_styles
        Tuple of line styles for the nullclines of each variable.
    nullcline_colors
        Tuple of colors for the nullclines of each variable.
    nullcline_linewidth
        Line width for the nullcline lines.
    nullcline_opacity
        Opacity for the nullcline lines (between 0 and 1).
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label
        padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label
        padding.
    axes_title_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_title for
        customizing the subplot titles, e.g., to specify a font size.
    axes_rects
        Optional list of subplot positions in normalized figure coordinates
        ``(left, bottom, width, height)``, one entry per subplot in
        top-to-bottom order.

    """
    if fig_ax is not None:
        fig, ax = fig_ax
        ax = cast(Sequence[plt.Axes], ax)
    elif axes_rects is not None:
        # Build figure with explicitly positioned axes to ensure consistent
        # placement regardless of whether colorbar/legend are included —
        # same pattern as plot_drift_3d (line ~950).
        fig = plt.figure(figsize=figsize)
        ax = cast(Sequence[plt.Axes], [fig.add_axes(rect) for rect in axes_rects])
    else:
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=figsize, layout="constrained", gridspec_kw=gridspec_kwargs
        )
        ax = cast(Sequence[plt.Axes], ax)

    for var_index, var_name in enumerate(variable_labels):
        vmin_ = vmin or np.nanmin(drift[..., var_index])
        vmax_ = vmax or np.nanmax(drift[..., var_index])
        contour_levels = np.linspace(vmin_, vmax_, num_levels)
        # center colormap at zero to visualize sign and magnitude of drift
        colormap_norm = TwoSlopeNorm(vmin=vmin_, vmax=vmax_, vcenter=0)

        contour = ax[var_index].contourf(
            meshgrid[0],
            meshgrid[1],
            drift[..., var_index],
            levels=contour_levels,
            cmap=colormap,
            norm=colormap_norm,
            extend="both",
        )
        if include_nullclines:
            # add dashed line for nullcline
            ax[var_index].contour(
                meshgrid[0],
                meshgrid[1],
                drift[..., var_index],
                levels=[0],
                colors=nullcline_colors[var_index],
                linestyles=[nullcline_styles[var_index]],
                linewidths=nullcline_linewidth,
                alpha=nullcline_opacity,
            )
        if include_colorbar:
            colorbar_ticks = np.linspace(vmin_, vmax_, cbar_num_ticks)
            colorbar_ticks = np.round(colorbar_ticks, cbar_tick_round)
            fig.colorbar(contour, ax=ax[var_index], label=f"d{var_name}/dt", ticks=colorbar_ticks)

        # set axis properties, only including label for edge plot of shared axes
        # (e.g., only xlabel for left column and only ylabel for bottom row if
        # multiple rows/columns of subplots)
        xlabel: str | None
        ylabel: str | None
        if n_rows > n_cols:
            # if more rows than columns, only set xlabel for bottom row
            xlabel = variable_labels[0] if var_index == n_rows - 1 else None
            ylabel = variable_labels[1]
        elif n_cols >= n_rows:
            # if more columns than rows, only set ylabel for left column
            xlabel = variable_labels[0]
            ylabel = variable_labels[1] if var_index == 0 else None
        set_axes_properties(
            ax[var_index],
            xlim=axes_limits[0] if axes_limits else None,
            ylim=axes_limits[1] if axes_limits else None,
            xlabel=xlabel,
            ylabel=ylabel,
            title=axes_titles[var_index] if axes_titles else None,
            aspect=axes_aspect,
            xlabel_kwargs=xlabel_kwargs,
            ylabel_kwargs=ylabel_kwargs,
            title_kwargs=axes_title_kwargs,
        )

    return fig, ax


def plot_drift_quiver(
    meshgrid: tuple[np.ndarray, np.ndarray],
    drift: np.ndarray,
    variable_labels: list[str],
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (4, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    quiver_scale: float = 4,
    quiver_color: str = "dimgrey",
    quiver_downsample: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    include_nullclines: bool = True,
    nullcline_styles: tuple = ("dashed", "dashdot"),
    nullcline_colors: tuple = ("k", "k"),
    nullcline_linewidth: float = 1.5,
    nullcline_opacity: float = 0.9,
    gridspec_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
    plot_legend: bool = True,
):
    """
    Make and save quiver plot of the drift vector field over the 2D state space.

    Parameters
    ----------
    meshgrid
        Meshgrid on which the drift is evaluated, typically obtained from
        np.meshgrid(..., indexing="ij").
    drift
        Drift vector field evaluated on the meshgrid, with shape (nx, ny, ndim).
    variable_labels
        Labels for axes corresponding to the state space variables, e.g.,
        ["$x_1$", "$x_2$"].
    fig_ax
        Optional tuple of (Figure, Axes) to plot on. If None, a new figure and
        axes will be created; if provided, the quiver plot will be made on the
        provided axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    quiver_scale
        Scale factor for the quiver plot (smaller values make arrows longer).
    quiver_color
        Color for the quiver arrows.
    quiver_downsample
        Factor by which to downsample the quiver arrows for visualization.
    include_nullclines
        Whether to include nullclines (where drift components are zero).
    nullcline_styles
        Tuple of line styles for the nullclines of each variable.
    nullcline_colors
        Tuple of colors for the nullclines of each variable.
    nullcline_linewidth
        Line width for the nullcline lines.
    nullcline_opacity
        Opacity for the nullcline lines (between 0 and 1).
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    legend_kwargs
        Optional dictionary of keyword arguments to pass to ax.legend for
        customizing the legend, e.g., to specify a title or font size.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label padding.
    plot_legend
        Whether to plot the legend for the nullclines.
    """

    # if vmin and vmax are provided, rescale components of the drift to be
    # between vmin and vmax for visualization purposes (e.g., to make arrows
    # more visible if drift magnitudes are very small or very large)
    if vmin is not None and vmax is not None:
        for component in range(drift.shape[-1]):
            drift[..., component] = np.clip(drift[..., component], vmin, vmax)

    fig, ax = fig_ax or plt.subplots(figsize=figsize, gridspec_kw=gridspec_kwargs)
    ax.quiver(
        meshgrid[0][::quiver_downsample, ::quiver_downsample],
        meshgrid[1][::quiver_downsample, ::quiver_downsample],
        drift[::quiver_downsample, ::quiver_downsample, 0],
        drift[::quiver_downsample, ::quiver_downsample, 1],
        color=quiver_color,
        pivot="tail",
        scale=quiver_scale,
    )
    if include_nullclines:
        for var_index, var_name in enumerate(variable_labels):
            # add dashed line for nullcline
            ax.contour(
                meshgrid[0],
                meshgrid[1],
                drift[..., var_index],
                levels=[0],
                colors=nullcline_colors[var_index],
                linestyles=[nullcline_styles[var_index]],
                linewidths=nullcline_linewidth,
                alpha=nullcline_opacity,
            )
            # add legend for nullclines
            ax.plot(
                [],
                [],
                color=nullcline_colors[var_index],
                linestyle=nullcline_styles[var_index],
                label=f"Nullcline d{var_name}/dt=0",
            )
        if plot_legend:
            ax.legend(**(legend_kwargs or {}))

    set_axes_properties(
        ax,
        xlim=axes_limits[0] if axes_limits else None,
        ylim=axes_limits[1] if axes_limits else None,
        xlabel=variable_labels[0],
        ylabel=variable_labels[1],
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )

    return fig, ax


def plot_drift_1d(
    drift: np.ndarray,
    x_values: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[float, float] = (4, 4),
    axes_limits: list[tuple[float, float]] | None = None,
    axes_labels: list[str] | None = None,
    add_flow_arrows: bool = True,
    flow_arrow_downsample: int = 5,
    flow_arrow_kwargs: dict | None = {"color": "dimgrey"},
    gridspec_kwargs: dict | None = None,
    axes_rect: tuple[float, float, float, float] | None = None,
    drift_line_kwargs: dict | None = None,
    zero_line_kwargs: dict | None = None,
    xlabel_kwargs: dict | None = None,
    ylabel_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D drift as a function of the state variable.

    Parameters
    ----------
    drift
        1D array of the drift component evaluated at the input `x_values`.
    x_values
        1D array of state variable values corresponding to the drift values.
    fig_ax
        Optional tuple of (Figure, Axes) to plot on. If None, a new figure and
        axes will be created; if provided, the plot will be made on the provided
        axes.
    figsize
        Size of the figure, specified as a tuple (width, height).
    axes_limits
        Optional limits for the axes, specified as a list of tuples.
    axes_labels
        Optional list of labels for the x and y axes, specified as a list of
        strings.
    add_flow_arrows
        If true, draw arrows along y = 0 to indicate the direction of flow,
        pointing right if drift is positive and left if drift is negative.
    flow_arrow_downsample
        Integer specifying the downsampling factor for the flow arrows. Arrows
        will be drawn at every nth center, where n is the downsampling factor.
    flow_arrow_kwargs
        Optional dictionary of keyword arguments to pass to ax.arrow for
        customizing the appearance of the flow arrows, e.g., to specify color or
        line width.
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to plt.subplots for
        creating the figure and axes, e.g., to specify a GridSpec layout.
    drift_line_kwargs
        Dictionary of keyword arguments to pass to ax.plot for customizing the
        line representing the drift, e.g., to specify color or line width.
    zero_line_kwargs
        Dictionary of keyword arguments to pass to ax.plot for customizing the
        line representing the zero drift level, e.g., to specify color, line
        style, line width, or opacity.
    xlabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_xlabel for
        customizing the x-axis label, e.g., to specify a font size or label
        padding.
    ylabel_kwargs
        Optional dictionary of keyword arguments to pass to ax.set_ylabel for
        customizing the y-axis label, e.g., to specify a font size or label
        padding.
    axes_rect
        Optional axes position in normalized figure coordinates
        ``(left, bottom, width, height)``.  When provided the figure is built
        without a layout engine via ``plt.figure`` and the axes is added with
        ``fig.add_axes``, mirroring the explicit-positioning pattern used in
        :func:`plot_drift_3d`.  When ``None`` (default) and ``fig_ax`` is also
        ``None``, ``plt.subplots`` with ``layout="constrained"`` is used
        instead.

    Returns
    -------
    :
        Tuple of figure and axes objects containing the plot of the 1D drift as
        a function of the state variable.
    """
    if fig_ax is not None:
        fig, ax = fig_ax
    elif axes_rect is not None:
        # Build figure with explicitly positioned axes — same pattern as
        # plot_drift_3d so callers can safely add further axes (legend) at
        # fixed figure coordinates without clipping.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(axes_rect)
    else:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained", gridspec_kw=gridspec_kwargs)
    ax.plot(x_values, drift, **(drift_line_kwargs or {}))
    ax.plot(x_values, np.zeros_like(x_values), **(zero_line_kwargs or {}))

    # add arrows to indicate flow direction, pointing right if drift is
    # positive and left if drift is negative (downsampled for vis)
    if add_flow_arrows:
        # make stand-in y values and drift in y direction for quiver plot: plot
        # arrows along y=0, with length and direction determined by drift values
        # in x (with "drift" in y = 0)
        y_values = np.zeros_like(x_values)
        drift_y = np.zeros_like(drift)

        # if scale is not specified in flow_arrow_kwargs, set it automatically
        # based on the maximum absolute value of the drift and the space between
        # arrows, to make arrow lengths visually informative without being too
        # small or too large
        if flow_arrow_kwargs is None or "scale" not in flow_arrow_kwargs:
            max_drift = np.max(np.abs(drift))
            downsample_spacing = np.mean(np.diff(x_values[::flow_arrow_downsample]))
            if max_drift > 0:
                flow_arrow_kwargs = flow_arrow_kwargs or {}
                flow_arrow_kwargs["scale"] = max_drift / downsample_spacing * 0.75
            else:
                flow_arrow_kwargs = flow_arrow_kwargs or {}
                flow_arrow_kwargs["scale"] = 1.0

        ax.quiver(
            x_values[::flow_arrow_downsample],
            y_values[::flow_arrow_downsample],
            drift[::flow_arrow_downsample],
            drift_y[::flow_arrow_downsample],
            **(flow_arrow_kwargs or {}),
        )

    set_axes_properties(
        ax,
        xlim=axes_limits[0] if axes_limits else None,
        ylim=axes_limits[1] if axes_limits else None,
        xlabel=axes_labels[0] if axes_labels else None,
        ylabel=axes_labels[1] if axes_labels else None,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )

    return fig, ax


def _plot_quiver_3d_cones(
    ax: Axes3D,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    length: float,
    colors: np.ndarray,
    alpha: float = 0.8,
    cone_fraction: float = 0.30,
    cone_radius_ratio: float = 0.4,
    n_facets: int = 8,
) -> None:
    """
    Draw 3-D quiver arrows with cone-shaped arrowheads using
    :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection` for the shafts and
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` for the cone faces.

    Unlike :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`, every arrowhead
    is a proper closed cone so arrows look volumetric from any viewing angle.

    Each arrow is decomposed into:

        - **shaft**: a single line segment from the tail to the cone base.
        - **cone side faces**: ``n_facets`` triangles between the base circle
          and the apex.
        - **cone base disc**: ``n_facets`` triangles that cap the open base of
          the cone so it appears solid when viewed from behind.

    All geometry is batched into two
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` objects (sides and
    base) for efficient rendering.

    Parameters
    ----------
    ax
        The 3-D axes on which to draw.
    x, y, z
        Flat arrays of arrow tail positions.
    u, v, w
        Flat arrays of arrow direction components.  They are normalised
        internally, so only the direction matters; overall arrow length is
        controlled by ``length``.
    length
        Total arrow length in data units (shaft + cone).
    colors
        RGBA colour array of shape ``(N, 4)`` — one colour per arrow.
    alpha
        Overall opacity applied to both shaft lines and cone faces.
    cone_fraction
        Fraction of ``length`` occupied by the cone head.  The shaft fills the
        remaining ``1 - cone_fraction`` portion.
    cone_radius_ratio
        Cone base radius expressed as a fraction of the cone height.  Larger
        values produce stubbier, more visible heads.
    n_facets
        Number of triangular side faces on each cone.  8 gives a smooth
        appearance without excessive vertex count.

    """
    eps = np.finfo(float).eps
    mag = np.sqrt(u**2 + v**2 + w**2)

    # Drop zero-magnitude vectors — they have no defined direction to draw.
    valid = mag > eps
    x, y, z, u, v, w, colors, mag = (arr[valid] for arr in (x, y, z, u, v, w, colors, mag))

    ud, vd, wd = u / mag, v / mag, w / mag

    shaft_length = length * (1.0 - cone_fraction)
    cone_height = length * cone_fraction
    cone_radius = cone_height * cone_radius_ratio

    # tip (apex) and cone-base centre for every arrow
    tip_x = x + ud * length
    tip_y = y + vd * length
    tip_z = z + wd * length
    base_x = x + ud * shaft_length
    base_y = y + vd * shaft_length
    base_z = z + wd * shaft_length

    # ------------------------------------------------------------------ shafts
    shaft_segs = np.stack(
        [
            np.column_stack([x, y, z]),
            np.column_stack([base_x, base_y, base_z]),
        ],
        axis=1,
    )  # (N, 2, 3)
    shaft_col = Line3DCollection(shaft_segs, colors=colors, alpha=alpha, linewidths=0.8)
    ax.add_collection3d(shaft_col)

    # ------------------------------------------------------------------ cones
    # Build two orthonormal vectors perpendicular to each direction
    # to parameterise the base circle.
    arbitrary = np.where(
        (np.abs(wd) < 0.9)[:, np.newaxis],
        np.tile([0.0, 0.0, 1.0], (len(ud), 1)),
        np.tile([1.0, 0.0, 0.0], (len(ud), 1)),
    )  # (N, 3)
    d_vec = np.column_stack([ud, vd, wd])  # (N, 3)
    perp1 = np.cross(d_vec, arbitrary)
    perp1 /= np.linalg.norm(perp1, axis=1, keepdims=True) + eps
    perp2 = np.cross(d_vec, perp1)
    perp2 /= np.linalg.norm(perp2, axis=1, keepdims=True) + eps

    # angles for base-circle vertices
    angles = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)
    cos_a = np.cos(angles)  # (n_facets,)
    sin_a = np.sin(angles)  # (n_facets,)

    # base circle points: shape (N, n_facets, 3)
    circle = (
        np.array([base_x, base_y, base_z]).T[:, np.newaxis, :]  # (N,1,3)
        + cone_radius * cos_a[np.newaxis, :, np.newaxis] * perp1[:, np.newaxis, :]
        + cone_radius * sin_a[np.newaxis, :, np.newaxis] * perp2[:, np.newaxis, :]
    )

    apex = np.column_stack([tip_x, tip_y, tip_z])  # (N, 3)

    # side faces: triangles (apex, circle[i], circle[i+1])
    next_i = (np.arange(n_facets) + 1) % n_facets
    side_verts = np.stack(
        [
            apex[:, np.newaxis, :].repeat(n_facets, axis=1),  # (N, n_facets, 3)
            circle,  # (N, n_facets, 3)
            circle[:, next_i, :],  # (N, n_facets, 3)
        ],
        axis=2,
    )  # (N, n_facets, 3-verts, 3-coords)
    N = len(x)
    side_verts_list = side_verts.reshape(N * n_facets, 3, 3).tolist()
    side_colors = np.repeat(colors, n_facets, axis=0)

    side_col = Poly3DCollection(
        side_verts_list,
        facecolors=side_colors,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection3d(side_col)

    # base disc: triangles (base_centre, circle[i], circle[i+1])
    base_centre = np.column_stack([base_x, base_y, base_z])
    base_verts = np.stack(
        [
            base_centre[:, np.newaxis, :].repeat(n_facets, axis=1),
            circle,
            circle[:, next_i, :],
        ],
        axis=2,
    )  # (N, n_facets, 3, 3)
    base_verts_list = base_verts.reshape(N * n_facets, 3, 3).tolist()

    base_col = Poly3DCollection(
        base_verts_list,
        facecolors=side_colors,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection3d(base_col)


class _HandlerConeArrow(HandlerBase):
    """Legend handler that draws a shaft + filled triangular cone head."""

    def __init__(
        self, color: ColorType, cone_fraction: float = 0.45, cone_radius_ratio: float = 0.7
    ) -> None:
        self._color = color
        self._cone_fraction = cone_fraction
        self._cone_radius_ratio = cone_radius_ratio
        super().__init__()

    def create_artists(
        self,
        _legend: Legend,
        _orig_handle: Artist,
        xdescent: float,
        _ydescent: float,
        width: float,
        height: float,
        _fontsize: float,
        trans: Any,
    ) -> list[Artist]:
        """
        Create a shaft + filled triangular cone head to represent a quiver arrow
        in the legend.

        Parameters
        ----------
        _legend
            The Legend object to which the handler is being applied (not used),
            kept for API compatibility with HandlerBase.
        _orig_handle
            The original handle (the object being represented in the legend, not
            used), kept for API compatibility with HandlerBase.
        xdescent
            The horizontal space to reserve for the handle.
        _ydescent
            The vertical space to reserve for the handle (not used), kept for
            API compatibility with HandlerBase.
        width
            The total width of the area allocated for the handle.
        height
            The total height of the area allocated for the handle.
        _fontsize
            The font size of the legend text (not used), kept for API
            compatibility with HandlerBase.
        trans
            The transformation to apply to the created artists to position them
            correctly in the legend.

        Returns
        -------
        :
            List of artists (arrow shaft and cone head) to be added to the legend.
        """
        shaft_y = height / 2
        cone_base_x = width * (1.0 - self._cone_fraction) - xdescent
        tip_x = width - xdescent
        cone_half_h = height * self._cone_fraction * self._cone_radius_ratio

        shaft = Line2D(
            [0, cone_base_x],
            [shaft_y, shaft_y],
            color=self._color,
            linewidth=0.8,
            transform=trans,
        )
        cone = MplPolygon(
            [
                [cone_base_x, shaft_y - cone_half_h],
                [tip_x, shaft_y],
                [cone_base_x, shaft_y + cone_half_h],
            ],
            closed=True,
            facecolor=self._color,
            edgecolor="none",
            transform=trans,
        )
        return [shaft, cone]


def process_3d_vector_field_for_visualization(
    vector_field_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    column_names: list[Column.DiffAEData],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    mask_threshold: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process a 3D vector field over a 3D meshgrid for visualization as a quiver
    plot with arrows colored by magnitude.

    Processing steps include: - masking grid points with low data density (based
    on a kernel density estimate
      of the feature data) by setting the vectors to NaN at those points,
    - wrapping the grid in the periodic variable theta (assumed to be the x
      coordinate) to be within the specified limits for better visualization
    - masking the vector field to be within the specified limits (taking only
      grid points within limits) and reshaping accordingly as 3D arrays of
      updated number of points within limits.

    Parameters
    ----------
    vector_field_dataframe
        DataFrame containing the vector field evaluated on a meshgrid, with
        columns corresponding to the grid coordinates and vector components.
    feature_dataframe
        DataFrame containing the feature data used to estimate the data density
        for masking low-density grid points.
    column_names
        List of column names corresponding to the grid coordinates.
    xlim
        Tuple specifying the plot limits for the x coordinate.
    ylim
        Tuple specifying the plot limits for the y coordinate.
    zlim
        Tuple specifying the plot limits for the z coordinate.
    mask_threshold
        Threshold for masking low-density grid points based on the kernel
        density estimate.

    Returns
    -------
    :
        Processed 3D grid coordinates and vector components, each as a tuple
        of 3D arrays.
    """
    vector_field_dict = get_vector_field_as_dict_from_dataframe(
        vector_field_dataframe, column_names
    )

    # grids and vectors are 3-D arrays shaped (n_theta, n_r, n_rho)
    x_grid_, y_grid_, z_grid_ = vector_field_dict["grid"]
    u_field_, v_field_, w_field_ = vector_field_dict["vectors"]

    # mask grid points with low data density before clipping/downsampling
    grid_points_1d = [
        np.unique(x_grid_[:, 0, 0]),
        np.unique(y_grid_[0, :, 0]),
        np.unique(z_grid_[0, 0, :]),
    ]
    bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]
    bin_limits = [
        (pts[0] - bw / 2, pts[-1] + bw / 2)
        for pts, bw in zip(grid_points_1d, bin_widths, strict=True)
    ]
    bins_3d = get_bins(bin_widths=tuple(bin_widths), bin_limits=bin_limits, pad=0)[0]
    kernels = [
        KramersMoyalKernel(
            name=KERNEL_NAMES_DYNAMICS[col],
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[col],
            period=KERNEL_PERIODS_DYNAMICS[col],
        )
        for col in column_names
    ]
    hist = np.histogramdd(feature_dataframe[column_names].to_numpy(), bins=bins_3d)[0]
    hist_kde = get_kernel_density_estimate_from_histogram(
        hist[None, ...], bins=bins_3d, kernel=kernels
    )
    low_density_mask = hist_kde < mask_threshold
    u_field_ = u_field_.copy()
    v_field_ = v_field_.copy()
    w_field_ = w_field_.copy()
    u_field_[low_density_mask] = np.nan
    v_field_[low_density_mask] = np.nan
    w_field_[low_density_mask] = np.nan

    # Wrap theta grid to be within the specified limits for better visualization
    # of the vector field (default is (0, pi), but we want to shift the limits
    # so that the stable fixed point is not at the boundary). Note that this
    # method hard-codes the assumption that theta is the x coordinate, which is
    # true for our current use case of this method.
    where_theta_below_lims = x_grid_ < xlim[0]
    where_theta_above_lims = x_grid_ > xlim[1]
    x_grid_[where_theta_below_lims] += POLAR_ANGLE_PERIOD
    x_grid_[where_theta_above_lims] -= POLAR_ANGLE_PERIOD
    arg_sorted_theta = np.argsort(x_grid_[:, 0, 0])
    x_grid_ = x_grid_[arg_sorted_theta, :, :]
    y_grid_ = y_grid_[arg_sorted_theta, :, :]
    z_grid_ = z_grid_[arg_sorted_theta, :, :]
    u_field_ = u_field_[arg_sorted_theta, :, :]
    v_field_ = v_field_[arg_sorted_theta, :, :]
    w_field_ = w_field_[arg_sorted_theta, :, :]

    # mask vector field to be within the specified limits (take only grid points
    # within limits), reshaping accordingly as 3D arrays of updated number of
    # points within limits
    x_in_bounds = (x_grid_ >= xlim[0]) & (x_grid_ <= xlim[1])
    num_x_in_bounds = np.unique(np.sum(x_in_bounds, axis=0))[-1]
    y_in_bounds = (y_grid_ >= ylim[0]) & (y_grid_ <= ylim[1])
    num_y_in_bounds = np.unique(np.sum(y_in_bounds, axis=1))[-1]
    z_in_bounds = (z_grid_ >= zlim[0]) & (z_grid_ <= zlim[1])
    num_z_in_bounds = np.unique(np.sum(z_in_bounds, axis=2))[-1]
    in_bounds_mask = x_in_bounds & y_in_bounds & z_in_bounds

    x_grid = x_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    y_grid = y_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    z_grid = z_grid_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    u_field = u_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    v_field = v_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)
    w_field = w_field_[in_bounds_mask].reshape(num_x_in_bounds, num_y_in_bounds, num_z_in_bounds)

    drift = (u_field, v_field, w_field)
    meshgrid = (x_grid, y_grid, z_grid)

    return drift, meshgrid


def plot_drift_3d(
    drift: tuple[np.ndarray, np.ndarray, np.ndarray],
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray],
    figsize: tuple[float, float] = (4.0, 5.0),
    downsample_factor: int = 6,
    colormap: str = "viridis_r",
    magnitude_limits: tuple[float, float] = (5e-2, 1.5),
    arrow_alpha: float = 0.6,
    include_colorbar: bool = True,
    include_legend: bool = True,
    include_stable_fixed_point_legend: bool = True,
    include_saddle_point_legend: bool = False,
    fixed_point_legend_label: str | None = None,
    saddle_point_legend_label: str | None = None,
    colorbar_rect: tuple[float, float, float, float] = (0.45, 0.12, 0.5, 0.02),
    **axes_kwargs: Any,
) -> tuple[plt.Figure, Axes3D]:
    """
    Render a 3D drift coefficient vector field as a quiver plot with arrows
    coloured by magnitude.

    Parameters
    ----------
    drift
        Tuple of 3D arrays (u, v, w) representing the components of the drift
        vector field evaluated on the meshgrid.
    meshgrid
        Tuple of 3D arrays (x, y, z) representing the coordinates of the
        meshgrid on which the drift is evaluated, typically obtained from
        np.meshgrid(..., indexing="ij").
    figsize
        Size of the figure, specified as a tuple (width, height).
    downsample_factor
        Factor by which to downsample the vector field for visualization. Arrows
        will be plotted at every nth grid point in each dimension, where n is
        the downsample factor.
    colormap
        Colormap to use for colouring the arrows by their magnitude.
    magnitude_limits
        Tuple specifying the (min, max) limits for the arrow magnitudes when
        mapping to colours. Magnitudes outside this range will be clipped to the
        limits for colouring purposes.
    arrow_alpha
        Opacity for the arrows (between 0 and 1).
    include_colorbar
        Whether to include a colorbar indicating the mapping from arrow colour
        to magnitude.
    include_legend
        Whether to include a legend for the arrows (and stable fixed point, if
        indicated).
    include_stable_fixed_point_legend
        Whether to include a legend entry for the stable fixed point (if True, a
        proxy artist with the same marker and color as the stable fixed point in
        the plot will be added to the legend).
    include_saddle_point_legend
        Whether to include a legend entry for the saddle point (if True, a proxy
        artist with the same marker and color as the saddle point in the plot will
        be added to the legend).
    fixed_point_legend_label
        Optional custom label for the stable fixed point legend entry.
    saddle_point_legend_label
        Optional custom label for the saddle point legend entry.
    colorbar_rect
        Rectangle specifying the position of the colorbar in normalized figure
        coordinates (left, bottom, width, height).
    axes_kwargs
        Additional keyword arguments to pass to set_axes_properties for
        customizing the axes, e.g., to specify axis limits, labels, title, or
        aspect ratio.


    Returns
    -------
    :
        Matplotlib figure and 3D axes objects.

    """

    # downsample vector field for plotting
    x_grid, y_grid, z_grid = meshgrid
    u_field, v_field, w_field = drift
    x_ds = x_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    y_ds = y_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    z_ds = z_grid[::downsample_factor, ::downsample_factor, ::downsample_factor]
    u_ds = u_field[::downsample_factor, ::downsample_factor, ::downsample_factor]
    v_ds = v_field[::downsample_factor, ::downsample_factor, ::downsample_factor]
    w_ds = w_field[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Compute vector magnitudes for colouring before normalizing to unit vectors
    # for plotting
    x_flat = x_ds.ravel()
    y_flat = y_ds.ravel()
    z_flat = z_ds.ravel()
    u_flat = u_ds.ravel()
    v_flat = v_ds.ravel()
    w_flat = w_ds.ravel()
    mag_flat = np.sqrt(u_flat**2 + v_flat**2 + w_flat**2)

    # Remove grid points that were masked upstream by density filtering (NaN
    # vectors)
    valid = ~np.isnan(mag_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    z_flat = z_flat[valid]
    u_flat = u_flat[valid]
    v_flat = v_flat[valid]
    w_flat = w_flat[valid]
    mag_flat = mag_flat[valid]

    # Map magnitudes to colours
    mag_eps = 1e-10
    cmap = plt.get_cmap(colormap)
    safe_cmin = max(magnitude_limits[0], mag_eps)
    safe_cmax = max(magnitude_limits[1], safe_cmin + mag_eps)
    norm_log = LogNorm(vmin=safe_cmin, vmax=safe_cmax)
    colors = cmap(norm_log(np.clip(mag_flat, safe_cmin, safe_cmax)))
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm_log)

    # Build matplotlib 3D figure with explicitly positioned axes to ensure
    # consistent size regardless of whether colorbar/legend are included
    fig = plt.figure(figsize=figsize)
    # Position: [left, bottom, width, height] in figure coordinates
    ax: Axes3D = fig.add_axes((0.10, 0.25, 0.80, 0.67), projection="3d")
    figsize_ratio = figsize[1] / figsize[0]
    ax.set_box_aspect((1.1 * figsize_ratio, 0.98 * figsize_ratio, 1.05 * figsize_ratio))

    # Render all arrows at the same absolute size (so visual clutter from
    # large-magnitude outliers is reduced) while still colouring by magnitude.
    avg_spacing = np.mean(np.diff(np.unique(x_flat)))
    arrow_length = avg_spacing * 0.8
    u_plot = u_flat / (mag_flat + mag_eps)
    v_plot = v_flat / (mag_flat + mag_eps)
    w_plot = w_flat / (mag_flat + mag_eps)
    _plot_quiver_3d_cones(
        ax,
        x_flat,
        y_flat,
        z_flat,
        u_plot,
        v_plot,
        w_plot,
        length=arrow_length,
        colors=colors,
        alpha=arrow_alpha,
    )

    if include_colorbar:
        # Colorbar - horizontal strip below the axes, centred in the right half
        # of the figure (x=0.50..0.94, centre at 0.72).  The legend is stacked
        # just above and shares the same centre so both have identical left-to-
        # right spacing.
        scalar_mappable.set_array([])
        cbar_ax = fig.add_axes(colorbar_rect)
        cbar = fig.colorbar(
            scalar_mappable,
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL, pad=2)
        cbar.set_label("vector magnitude", fontsize=FONTSIZE_XSMALL, labelpad=2)
        cbar_ax.xaxis.set_label_position("top")
        cbar_ax.xaxis.tick_bottom()

    if include_legend:
        # Legend below the axes, stacked above the colorbar and sharing the
        # same horizontal centre (x=0.72) so both have identical left-to-right
        # spacing.  Draw the vector arrow handle as a
        # shaft + filled triangular cone head (matching the plot style) coloured at
        # a value in the center of the colormap, and add a proxy artist for the
        # stable fixed point using the same marker and color as in the plot.
        arrow_color = cmap(0.5)
        arrow_handle = Line2D(
            [],
            [],
            label="$d\\mathbf{x}/dt=\\mathbf{f}(\\mathbf{x})$",
        )
        handles = [arrow_handle]
        if include_stable_fixed_point_legend:
            fp_labels: dict[str, str] | None = None
            if fixed_point_legend_label is not None:
                fp_labels = {StabilityLabel.STABLE: fixed_point_legend_label}
            fp_handles = make_legend_handles_for_fixed_pts(
                fpt_stabilities=[StabilityLabel.STABLE],
                marker_size=4,
                labels=fp_labels,
            )
            handles.extend(fp_handles)
        if include_saddle_point_legend:
            fp_labels = None
            if saddle_point_legend_label is not None:
                fp_labels = {StabilityLabel.SADDLE: saddle_point_legend_label}
            saddle_fp_handles = make_legend_handles_for_fixed_pts(
                fpt_stabilities=[StabilityLabel.SADDLE],
                marker_size=3,
                labels=fp_labels,
            )
            handles.extend(saddle_fp_handles)
        fig.legend(
            handles=handles,
            fontsize=FONTSIZE_XSMALL,
            loc="lower center",
            bbox_to_anchor=(0.27, 0.04),
            frameon=False,
            handletextpad=0.3,
            labelspacing=0.4,
            handler_map={arrow_handle: _HandlerConeArrow(color=arrow_color)},
        )

    # set axes labels and ticks with custom formatting
    ax.tick_params(axis="both", pad=-3)
    set_axes_properties(ax, **axes_kwargs)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_ha("right")
        tick.set_va("center")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_ha("left")
        tick.set_va("center")
    ax.zaxis.set_rotate_label(False)
    # Move z-axis spine to the left vertical edge. 'lower' forces the spine
    # onto the min-x/min-y corner of the bounding box, which projects to the
    # left side in the default view angle.
    ax.zaxis.set_ticks_position("lower")
    ax.zaxis.set_label_position("lower")
    for tick in ax.zaxis.get_majorticklabels():
        tick.set_ha("right")
        tick.set_va("center")

    return fig, ax


def make_legend_handles_for_fixed_pts(
    fpt_stabilities: list[str],
    marker_size: int = 10,
    edge_color: str = "black",
    labels: dict[str, str] | None = None,
) -> list[StabilityLegendHandle]:
    """Make a custom legend for the fixed point types, nullclines and trajectories.

    Purpose of this method is to create a legend that only includes the fixed
    point types that are present in the plot, since the number and type of fixed
    points can vary across parameter space. That is, we want to avoid having
    duplicate labels where we have multiple fixed points of the same type, but
    we also want to avoid having labels for types that are not present.

    Parameters
    ----------
    fpt_stabilities
        List of stability labels for the fixed points.
    marker_size
        Size of the markers for the legend handles.
    edge_color
        Color of the marker edges.
    labels
        Optional dictionary mapping stability labels to custom legend labels. If
        None, default labels of the form "{stability_type} fixed point" will be
        used.

    Returns
    -------
    :
        List of StabilityLegendHandle objects representing the legend handles
        for the fixed point types.

    """
    labels_ = labels or {}
    my_handles = []
    # get legend handles for the fixed point types that are present in given
    # list of fixed point stabilities, in the order given by StabilityLabel enum
    for stability_type in StabilityLabel:
        if stability_type in fpt_stabilities:
            my_handles.append(
                StabilityLegendHandle(
                    stability_label=stability_type,
                    legend_label=labels_.get(stability_type, f"{stability_type} fixed point"),
                    marker=FIXED_POINT_PLOT_STYLE[stability_type].marker,
                    face_color=FIXED_POINT_PLOT_STYLE[stability_type].color,
                    edge_color=edge_color,
                    marker_size=marker_size,
                )
            )

    return my_handles
