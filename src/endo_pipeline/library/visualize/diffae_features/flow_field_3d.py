"""Methods used for visualization the the 3D Diff AE feature flow fields."""

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.polar_coords import rewrap_polar_angle
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.dynamics_workflows import POLAR_ANGLE_RANGE
from endo_pipeline.settings.figures import (
    FONT_FAMILY,
    FONTSIZE_LARGE,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
)
from endo_pipeline.settings.flow_field_3d import (
    CLIP_MAGNITUDES,
    CLIP_MAX_MAGNITUDE_PERCENTILE,
    CLIP_MIN_MAGNITUDE_PERCENTILE,
    FIGSIZE_2D_FLOW_FIELD,
    FIGSIZE_FLOW_FIELD_STACK,
    FLOW_FIELD_X_AXIS_LABEL,
    FLOW_FIELD_Y_AXIS_LABELS,
    KDE_CONTOUR_COLORMAP,
    KDE_CONTOUR_LEVELS,
    KDE_CONTOUR_OPACITY,
    LOG_NORM_MAGNITUDES,
    NCOLS_2D_FLOW_FIELD,
    NORMALIZE_QUIVER_VECTORS,
    NROWS_2D_FLOW_FIELD,
    QUIVER_COLORMAP,
    QUIVER_DOWNSAMPLE_FACTOR,
    QUIVER_VECTOR_SCALE,
)

logger = logging.getLogger(__name__)


def interpolate_on_curve(traj: np.ndarray, n_points: int = 5) -> np.ndarray:
    """
    Obtain points along a curve equally spaced by arc length.

    Parameters
    ----------
    traj
        Curve in n-dimensional space, shape (num_t, num_dimensions).
    n_points
        Number of equally spaced points to interpolate along the curve.

    Returns
    -------
    :
        Interpolated points along the curve, shape (n_points, num_dimensions).

    """
    ndim = traj.shape[1]  # number of dimensions

    # compute cumulative distance of
    # each point from the first point
    # along the trajectory
    distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc_length = np.cumsum(np.concatenate(([0], distances)))

    # interpolate to by these distances to
    #  get n_points evenly spaced points
    arc_length_new = np.linspace(0, arc_length[-1], n_points)

    # initialize array interpolated points
    interpolated_points = np.zeros((n_points, 3))
    for i in range(ndim):  # loop over dimensions
        interpolated_points[:, i] = np.interp(arc_length_new, arc_length, traj[:, i])

    return interpolated_points


def set_slice_plot_bounds_and_labels(
    axs: np.ndarray[plt.Axes, Any],
    bounds: list[np.ndarray] | list[tuple[float, float]],
    x_label: str = FLOW_FIELD_X_AXIS_LABEL,
    y_labels: tuple[str, ...] = FLOW_FIELD_Y_AXIS_LABELS,
) -> np.ndarray[plt.Axes, Any]:
    """Set the axis limits and labels for the 2D slice plots of the flow field.

    Parameters
    ----------
    axs
        Array of Matplotlib Axes to set the bounds and labels for.
    bounds
        List of arrays or tuples specifying the plot bounds for each principal
        component.
    x_label
        Label for the x-axis.
    y_labels
        Tuple of labels for the y-axes of each subplot. The length of y_labels
        must match the number of axes in axs.

    Returns
    -------
    :
        The input array of Matplotlib Axes with the bounds and labels set.

    """
    if len(y_labels) != len(axs):
        logger.error("Number of y_labels must match number of axes.")
        raise ValueError("Number of y_labels must match number of axes.")

    xmin, xmax = bounds[0][0], bounds[0][1]

    for i, ax in enumerate(axs):
        qmin, qmax = bounds[i + 1][0], bounds[i + 1][1]
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(x_label, fontsize=FONTSIZE_MEDIUM)
        ax.set_ylabel(y_labels[i], fontsize=FONTSIZE_MEDIUM)
        ax.set_ylim(qmin, qmax)
        ax.set_aspect("equal")
        # set number of x ticks = number of y ticks = 5
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        # set aspect
        ax.set_aspect("auto", adjustable="box")
    return axs


def get_slice_indexes(
    sliced_variable_grid: np.ndarray, sliced_variable_val: float
) -> tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...]:
    """Get the slice indexes of the grid that are closest to the prescribed value.

    This function is used to slice the 3D grid in 2D for plotting.

    **Input variable grid**

    The input variable grid ``sliced_variable_grid`` is a 3D array representing
    the values of the variable along the grid points in the 3D space. For
    example, if we are slicing along the z-axis (e.g., PC3), then
    ``sliced_variable_grid`` would be the 3D array of z-values at each grid
    point. This is, e.g., obtained as an element of the tuple output by
    `np.meshgrid`.

    Parameters
    ----------
    sliced_variable_grid
        The grid of the variable to be sliced.
    sliced_variable_val
        The value of the variable at which to slice the grid.

    Returns
    -------
    :
        Tuple of arrays representing the indices of the slice in the grid.

    """
    # get slice closest to the prescribed value
    # first, get the absolute distance to the prescribed value
    dist_to_point = np.abs(sliced_variable_grid - sliced_variable_val)
    # get indexes of points where this distance is minimized
    slice_indexes_ = np.where(dist_to_point.ravel() == dist_to_point.min())[0]
    # unravel the grid to get the indices of the points in the grid
    # that are within the z-range of interest
    slice_indexes = np.unravel_index(slice_indexes_, sliced_variable_grid.shape)
    return slice_indexes


def _get_colormap_norm(
    color_metric: np.ndarray,
    log_normalize: bool = True,
) -> Normalize | LogNorm:
    """Get the colormap normalization object for the given color metric.

    Parameters
    ----------
    color_metric
        The metric to be used for coloring.
    log_normalize
        Whether to log normalize the color mapping.

    Returns
    -------
    :
        The colormap normalization object.

    """
    if log_normalize:
        return LogNorm(
            vmin=np.nanmin(color_metric),
            vmax=np.nanmax(color_metric),
        )
    else:
        return Normalize(
            vmin=np.nanmin(color_metric),
            vmax=np.nanmax(color_metric),
        )


def _get_colormap_values(
    colormap_name: str,
    color_metric: np.ndarray,
    log_normalize: bool = LOG_NORM_MAGNITUDES,
    clip_metric: bool = CLIP_MAGNITUDES,
    clip_min_percentile: float | None = CLIP_MIN_MAGNITUDE_PERCENTILE,
    clip_max_percentile: float | None = CLIP_MAX_MAGNITUDE_PERCENTILE,
) -> np.ndarray:
    """Get the colormap values for the given color metric.

    Parameters
    ----------
    colormap_name
        The name of the colormap to be used.
    color_metric
        The metric to be used for coloring (e.g., vector magnitudes).
    log_normalize
        Log normalize the color mapping if True, else linear normalize.
    clip_metric
        Whether to clip the color metric to avoid outliers.
    clip_min_percentile
        The minimum percentile for clipping the color metric. If None, no clipping is applied.
    clip_max_percentile
        The maximum percentile for clipping the color metric. If None, no clipping is applied.

    Returns
    -------
    :
        The normalized colormap values for the given color metric.

    """
    colormap_object = get_cmap(colormap_name)
    if clip_metric:
        a_min = (
            np.nanpercentile(color_metric, clip_min_percentile)
            if clip_min_percentile is not None
            else None
        )
        a_max = (
            np.nanpercentile(color_metric, clip_max_percentile)
            if clip_max_percentile is not None
            else None
        )
        color_metric = np.clip(
            color_metric,
            a_min=a_min,
            a_max=a_max,
        )
    # get colormap normalization
    norm_colors = _get_colormap_norm(color_metric, log_normalize=log_normalize)
    color_values = colormap_object(norm_colors(color_metric.ravel())).reshape(
        (*color_metric.shape, 4)
    )  # reshape to get RGBA colors
    return color_values


def plot_flow_field_stack(
    flow_field_dict: dict,
    plot_axes_indicies: tuple[int, int],
    slice_axis_index: int,
    plot_bounds: list[np.ndarray],
    slice_steps: np.ndarray,
    fig_savedir: Path,
    colormap_name: str = QUIVER_COLORMAP,
    clip_metric: bool = CLIP_MAGNITUDES,
    log_normalize: bool = LOG_NORM_MAGNITUDES,
    feature_labels: list[str] | None = None,
) -> None:
    """Make and save plot of the 3D flow field in 2D over a stack of slices in the 3rd variable.

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    plot_axes_indicies
        Tuple (i,j) of indices specifying which features to plot.
    slice_axis_index
        Index of the feature variable to slice over.
    plot_bounds
        List of arrays specifying the plot bounds for the x and y axes of the 2D
        plots.
    slice_steps
        List of arrays specifying the slice steps for the slicing axis.
    fig_savedir
        Directory to save the figures.
    colormap_name
        Name of the colormap to use for the flow field magnitude.
    clip_metric
        Whether to clip the color metric to avoid outliers.
    log_normalize
        Whether to log normalize the color mapping.
    feature_labels
        List of labels for each feature variable in the plot. If None, default
        labels are used.

    """
    if feature_labels is None:
        feature_labels = [
            get_label_for_column(DIFFAE_PC_COLUMN_NAMES[idx]) for idx in range(NUM_PCS_TO_ANALYZE)
        ]

    # unpack plot axes
    i, j = plot_axes_indicies

    # get flow field
    v_i = flow_field_dict["vectors"][i]
    v_j = flow_field_dict["vectors"][j]
    v_k = flow_field_dict["vectors"][slice_axis_index]

    # color by magnitude of the flow field (log normalized or not, clipped or not)
    vector_magnitude = np.sqrt(v_i**2 + v_j**2 + v_k**2)
    color_array = _get_colormap_values(
        colormap_name,
        vector_magnitude,
        log_normalize=log_normalize,
        clip_metric=clip_metric,
    )

    # get grid and grid spacing
    x_i_grid = flow_field_dict["grid"][i]
    x_j_grid = flow_field_dict["grid"][j]
    x_k_grid = flow_field_dict["grid"][slice_axis_index]

    ax_list = []
    for n, slice_value in enumerate(slice_steps):
        # set up figure
        fig, ax = plt.subplots(figsize=FIGSIZE_FLOW_FIELD_STACK)

        # get meshgrid indexes for the current slice value
        x_k_valids = get_slice_indexes(x_k_grid, slice_value)

        # plot quiver plots for the specified slice
        ax = plot_one_slice_quiver(
            (v_i, v_j),
            (x_i_grid, x_j_grid),
            x_k_valids,
            ax=ax,
            color=color_array,
        )
        # set the axis limits and labels
        ax = set_slice_plot_bounds_and_labels(
            np.array([ax]),
            plot_bounds,
            x_label=feature_labels[i],
            y_labels=(feature_labels[j],),
        )[0]
        # add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=colormap_name, norm=_get_colormap_norm(vector_magnitude, log_normalize)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Flow Field Magnitude", fontsize=FONTSIZE_SMALL)
        # set title with slice value
        ax.set_title(f"{feature_labels[slice_axis_index]} = {slice_value:.4f}")
        plt.tight_layout()
        ax_list.append(ax)
        save_plot_to_path(
            fig,
            fig_savedir,
            f"flow_field_stack_{n}",
        )


def plot_one_slice_quiver(
    velocities: tuple,
    grid: tuple,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ax: plt.Axes,
    color: str | np.ndarray,
    norm: bool = NORMALIZE_QUIVER_VECTORS,
    downsample_factor: int = QUIVER_DOWNSAMPLE_FACTOR,
    scale: int | float = QUIVER_VECTOR_SCALE,
) -> plt.Axes:
    """Plot one slice of the flow field (quiver plot) for a given slice of the grid.

    Parameters
    ----------
    velocities
        Tuple of 2D arrays representing the velocity components in the slice.
    grid
        Tuple of 2D arrays representing the grid coordinates in the slice.
    slice_indexes
        Tuple of arrays specifying the slice indexes for the 2D slice.
    ax
        Matplotlib Axes to plot on.
    color
        Array of RGBA colors for coloring the quiver arrows or single color string.
    norm
        Whether to normalize the quiver plot arrows.
    downsample_factor
        Factor by which to downsample the quiver plot grid.
    scale
        Scale factor for the quiver arrows.

    Returns
    -------
    :
        The Matplotlib Axes with the quiver plot of the flow field slice.

    """
    # slice the grid to get the points in the slice
    # and reshape to 2d array
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    x1_grid = grid[0][slice_indexes].reshape(my_shape)
    x2_grid = grid[1][slice_indexes].reshape(my_shape)
    dx1 = velocities[0][slice_indexes].reshape(my_shape)
    dx2 = velocities[1][slice_indexes].reshape(my_shape)

    # flatten down to 2D depending on which axis has shape == 1
    which_idx = np.where(np.array(my_shape) == 1)[0][0]
    # get xi_grid[... 0 ...] where 0 is taken from the axis with shape == 1
    # and same for dx1 and dx2
    x1_grid = np.take(x1_grid, 0, axis=which_idx)
    x2_grid = np.take(x2_grid, 0, axis=which_idx)
    dx1 = np.take(dx1, 0, axis=which_idx)
    dx2 = np.take(dx2, 0, axis=which_idx)

    if norm:  # normalize vectors in 2D
        dx1_ = dx1 / np.sqrt(dx1**2 + dx2**2)
        dx2_ = dx2 / np.sqrt(dx1**2 + dx2**2)
    else:
        dx1_ = dx1.copy()
        dx2_ = dx2.copy()

    # downsample the grid for quiver plot
    # and transpose (meshgrid generated via indexing ij)
    x1_grid_ = x1_grid[::downsample_factor, ::downsample_factor].T
    x2_grid_ = x2_grid[::downsample_factor, ::downsample_factor].T
    dx1_ = dx1_[::downsample_factor, ::downsample_factor].T
    dx2_ = dx2_[::downsample_factor, ::downsample_factor].T

    # if coloring arrows by some metric, slice, reshape, and downsample color array
    if isinstance(color, np.ndarray):
        color_sliced = color[slice_indexes].reshape((*my_shape, 4))  # RGBA colors
        color_squeezed = np.take(color_sliced, 0, axis=which_idx)
        color_ = (
            color_squeezed[::downsample_factor, ::downsample_factor].swapaxes(0, 1).reshape(-1, 4)
        )
    else:
        color_ = color  # single color string

    # plot quiver
    ax.quiver(x1_grid_, x2_grid_, dx1_, dx2_, color=color_, scale=scale)

    return ax


def plot_quiver_slices(
    flow_field_dict: dict,
    slice_indexes: tuple[
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ],
    colormap_name: str = QUIVER_COLORMAP,
    norm: bool = NORMALIZE_QUIVER_VECTORS,
    log_norm_colormap: bool = True,
    fig_ax: tuple | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Plot quiver plots of the 3D flow field for the specified 2D slices.

    **Input dictionary flow_field_dict:**

    The method input ``flow_field_dict`` should have the following key/value pairs:
        - "vectors": tuple of 3D arrays (v1,v2,v3)
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    slice_indexes
        Tuple of tuples specifying the slice indexes for the 2D slices.
    colormap_name
        Name of the colormap to use for the quiver plot arrows.
    norm
        Whether to normalize the quiver plot arrows.
    log_norm_colormap
        Whether to use a logarithmic normalization for the colormap.
    fig_ax
        Tuple of (fig, ax) to plot on. If None, a new figure and axes are created.

    Returns
    -------
    :
        Matplotlib Figure and array of Axes with the quiver plots of the flow field slices.

    """
    # get flow field
    v1, v2, v3 = flow_field_dict["vectors"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    vector_magnitude = np.sqrt(v1**2 + v2**2 + v3**2)
    color_array = _get_colormap_values(
        colormap_name,
        vector_magnitude,
        log_normalize=log_norm_colormap,
        clip_metric=True,
    )

    # plot quiver plots for the specified slices
    if fig_ax is None:
        fig, ax = plt.subplots(
            NROWS_2D_FLOW_FIELD, NCOLS_2D_FLOW_FIELD, figsize=FIGSIZE_2D_FLOW_FIELD
        )
    else:
        fig, ax = fig_ax
    ax[0] = plot_one_slice_quiver(
        (v1, v2), (xgrid, ygrid), slice_indexes[0], ax=ax[0], color=color_array, norm=norm
    )
    ax[1] = plot_one_slice_quiver(
        (v1, v3), (xgrid, zgrid), slice_indexes[1], ax=ax[1], color=color_array, norm=norm
    )
    # add colorbar to bottom axis
    sm = plt.cm.ScalarMappable(
        cmap=colormap_name,
        norm=_get_colormap_norm(vector_magnitude, log_normalize=log_norm_colormap),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.set_label("flow field magnitude (3D)", fontsize=FONTSIZE_SMALL)

    # set axes aspect to be square
    for ax_ in ax:
        ax_.set_aspect(1.0 / ax_.get_data_ratio())

    return fig, ax


def plot_flow_field_slices(
    flow_field_dict: dict,
    dataset_name: str,
    plot_bounds: list[np.ndarray],
    fig_savedir: Path | None,
    feature_vals: tuple[Any, Any],
    colormap_name: str = QUIVER_COLORMAP,
    norm: bool = NORMALIZE_QUIVER_VECTORS,
    prob_kde: np.ndarray | None = None,
    log_norm_colormap: bool = True,
    column_names: list[str] | None = None,
    fig_title: str | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Plot 2D slices of the 3D flow field for the specified 2D slices.

    Also overlays a KDE contour plot of the data in the 2D slice if `prob_kde`
    is not None.

    **Input dictionary flow_field_dict:**

    The method input ``flow_field_dict`` should have the following key/value
    pairs:
        - "vectors": tuple of 3D arrays (v1,v2,v3)
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    dataset_name
        Name of the dataset being plotted.
    plot_bounds
        List of arrays specifying the plot bounds for each principal component.
    fig_savedir
        Optional, directory to save the figure.
    feature_vals
        Values at which to slice the data of the variables that make up the 3rd
        and 2nd axes of the 3D space (e.g., PC3 and PC2) for plotting the 2D
        slices.
    colormap_name
        Name of the colormap to use for the quiver plot arrows.
    norm
        Whether to normalize the quiver plot arrows.
    prob_kde
        Optional, 3D array representing the probability density estimate for the
        data in the slices. If None, no KDE contours are plotted.
    log_norm_colormap
        Whether to use a logarithmic normalization for the colormap.
    column_names
        Optional, list of column names corresponding to features being used for
        the analysis (e.g. the top 3 PCs). Used for labeling the slice values in
        the plot titles and logging.

    Returns
    -------
    :
        Matplotlib Figure and array of Axes with the quiver plots of the flow field slices.

    """
    column_names_ = column_names or DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
    column_labels = [get_label_for_column(col) for col in column_names_]
    # get grid and grid spacing
    meshgrid_tuple = flow_field_dict["grid"]

    # for plotting in 2D, we need to slice the data in feature space at the
    # specified values of the 2nd and 3rd variables (e.g., PC2 and PC3)
    feature_z_val = feature_vals[0]
    feature_y_val = feature_vals[1]

    # get z-slice closest to z = feature_z_val
    zvalids = get_slice_indexes(meshgrid_tuple[-1], feature_z_val)
    # get y-slice closest to y = feature_y_val
    yvalids = get_slice_indexes(meshgrid_tuple[1], feature_y_val)

    # plot quiver plots of these y and z slices, with optional KDE contours of
    # the data in these slices
    fig, ax = plt.subplots(NROWS_2D_FLOW_FIELD, NCOLS_2D_FLOW_FIELD, figsize=FIGSIZE_2D_FLOW_FIELD)

    # plot KDE contours of data in x-y and x-z planes, if specified
    if prob_kde is not None:
        for subplot_index, plot_axis_index_pair, slice_axis_index in [
            (0, (0, 1), 2),  # feature 1 vs feature 2 over feature 3 slices
            (1, (0, 2), 1),  # feature 1 vs feature 3 over feature 2 slices
        ]:
            # get a 2D meshgrid for the current slice
            # take the slice along the z-axis for x-y plane and y-axis for x-z plane
            mesh_dim_1 = np.take(meshgrid_tuple[plot_axis_index_pair[0]], 0, axis=slice_axis_index)
            mesh_dim_2 = np.take(meshgrid_tuple[plot_axis_index_pair[1]], 0, axis=slice_axis_index)

            # marginalize probability density over the variable that is sliced (e.g., z or y)
            dx_along_sliced_axis = np.unique(
                np.diff(meshgrid_tuple[slice_axis_index], axis=slice_axis_index)
            )[-1]
            # replace NaNs with 0 for integration, since we want to integrate
            # over the entire slice and NaNs represent points where the density
            # is not defined (e.g., outside the bounds of the data)
            prob_kde_no_nan = np.nan_to_num(prob_kde, nan=0.0)
            marginal_prob_kde = np.trapz(
                prob_kde_no_nan, dx=dx_along_sliced_axis, axis=slice_axis_index
            )
            # plot contourf of the density
            ax[subplot_index].contourf(
                mesh_dim_1,
                mesh_dim_2,
                marginal_prob_kde,
                levels=KDE_CONTOUR_LEVELS,
                cmap=KDE_CONTOUR_COLORMAP,
                alpha=KDE_CONTOUR_OPACITY,
            )
        # add contour plot colorbar
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=KDE_CONTOUR_COLORMAP),
            ax=ax[0],
        )
        cbar.set_label("marginal probability density", fontsize=FONTSIZE_SMALL)

    # plot quiver plots for the specified slices
    fig, ax = plot_quiver_slices(
        flow_field_dict,
        (zvalids, yvalids),
        colormap_name=colormap_name,
        norm=norm,
        log_norm_colormap=log_norm_colormap,
        fig_ax=(fig, ax),
    )

    # set the axis limits and labels
    ax = set_slice_plot_bounds_and_labels(
        ax, plot_bounds, x_label=column_labels[0], y_labels=(column_labels[1], column_labels[2])
    )
    # set titles with slice values
    ax[0].set_title(f"{column_labels[2]} = {feature_z_val:.2f}")
    ax[1].set_title(f"{column_labels[1]} = {feature_y_val:.2f}")
    plt.tight_layout()

    if fig_title is not None:
        fig.suptitle(
            fig_title,
            fontsize=FONTSIZE_LARGE,
            y=1.02,
            fontfamily=FONT_FAMILY,
        )

    if fig_savedir is not None:
        file_name = filename or f"flow_field_{dataset_name}"
        save_plot_to_path(fig, fig_savedir, file_name)

    return fig, ax


def plot_stable_fixed_points_together(
    stable_fixed_points_df: pd.DataFrame,
    plot_bounds: list[np.ndarray],
    fig_savedir: Path,
    column_names: list[str],
) -> None:
    """Make and save plot of stable fixed points from multiple datasets together.

    **Input DataFrame**

    The method input ``stable_fixed_points_df`` should have the following
    columns:
        - ColumnName.DATASET
        - column_names[0] (e.g., "pc_1")
        - column_names[1] (e.g., "pc_2")
        - column_names[2] (e.g., "pc_3")

    Parameters
    ----------
    stable_fixed_points_df
        DataFrame containing stable fixed points from multiple datasets.
    plot_bounds
        List of arrays specifying the plot bounds for each principal component.
    fig_savedir
        Directory to save the figure.
    column_names
        List of column names corresponding to features being used for the
        analysis (e.g. the top 3 PCs). Used for indexing the columns in the DataFrame.

    """
    # check that required columns are present
    fp_column_names = [ColumnTemplate.FIXED_POINT % column for column in column_names]
    required_columns = [Column.DATASET, *fp_column_names]
    check_required_columns_in_dataframe(stable_fixed_points_df, required_columns)

    column_labels = [get_label_for_column(col) for col in column_names]

    # initialize plots
    fig, ax = plt.subplots(NROWS_2D_FLOW_FIELD, NCOLS_2D_FLOW_FIELD, figsize=FIGSIZE_2D_FLOW_FIELD)

    # loop over datasets and plot their stable fixed points
    patch_list_for_legend = []
    for dataset_name, dataset_df in stable_fixed_points_df.groupby(Column.DATASET):
        dataset_name_ = cast(str, dataset_name)
        scatter_color = get_dataset_color(dataset_name_)
        patch_list_for_legend.append(Patch(color=scatter_color, label=dataset_name_))
        fpts = dataset_df[fp_column_names].values
        for fpt in fpts:
            # plot fixed point
            # x-y, x-z
            ax[0].scatter(fpt[0], fpt[1], s=100, color=scatter_color, edgecolor="black")
            ax[1].scatter(fpt[0], fpt[2], s=100, color=scatter_color, edgecolor="black")

    # set the axis limits and labels
    ax = set_slice_plot_bounds_and_labels(
        ax, plot_bounds, x_label=column_labels[0], y_labels=(column_labels[1], column_labels[2])
    )

    # add legend
    ax[0].legend(bbox_to_anchor=(1.02, 1.02), title="Datasets", handles=patch_list_for_legend)

    plt.tight_layout()

    # save the figure
    save_plot_to_path(fig, fig_savedir, "fixed_points_plot")


def visualize_3d_flow_field_for_one_dataset(
    flow_field_dict: dict,
    df: pd.DataFrame,
    column_names: list[str],
    traj: np.ndarray,
    stable_fixed_points: list[np.ndarray],
    prob_kde: np.ndarray | None,
    plot_bounds: list[np.ndarray],
    plot_stack: bool,
    fig_savedir: Path,
    fig_title: str | None,
    filename: str | None,
) -> None:
    """Make and save 2D summary plots for the computed 3D flow fields.

    **Input dictionary flow_field_dict:**

    The method input `flow_field_dict` should have the following key/value
    pairs:
        - "vectors": tuple of 3D arrays (v1,v2,v3)
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    df
        DataFrame containing the data to be plotted (from one
        dataset/experimental condition).
    column_names
        List of column names corresponding to features being used for the
        analysis (e.g. the top 3 PCs).
    traj
        Trajectory of the system in the flow field.
    stable_fixed_points
        List of stable fixed points in the flow field.
    prob_kde
        Optional, 3D array representing the probability density estimate for the
        data in the slices. If None, no KDE contours are plotted.
    plot_bounds
        List of arrays specifying the plot bounds for each principal component.
    plot_stack
        Whether to plot stacks of flow field slices.
    fig_savedir
        Directory to save the figures.
    fig_title
        Optional, title for the figure. If None, no title is set.
    filename
        Optional, filename for saving the figure. If None, a default name is used.

    """
    # dataset flow condition for saving the figures
    dataset_name = df[Column.DATASET].unique()[0]

    ###### additional plots for visualization of flow field #######
    # 1) plot stacks of flow field slices
    # 2) last point of trajectory over flow field
    # 3) entire trajectory over flow field
    # 4) trajectory with equally spaced interpolated points

    # 1) plot stacks of flow field slices
    # get feature axis slices from the
    # meshgrid in the flow field dictionary
    if plot_stack:
        plot_axes_indicies = [
            (0, 1),  # feature 1 vs feature 2 over feature 3 slices
            (0, 2),  # feature 1 vs feature 3 over feature 2 slices
            (1, 2),  # feature 2 vs feature 3 over feature 1 slices
        ]
        slice_axis_indices = [2, 1, 0]  # feature 3, feature 2, feature 1
        pc_slices = [
            flow_field_dict["grid"][0][:, 0, 0],  # feature 1
            flow_field_dict["grid"][1][0, :, 0],  # feature 2
            flow_field_dict["grid"][2][0, 0, :],  # feature 3
        ]
        column_labels = [get_label_for_column(col) for col in column_names]

        for i, slice_axis in enumerate(slice_axis_indices):
            column_name = column_names[slice_axis]
            logger.info("Plotting flow field stack for slice axis [ %s ].", column_name)
            plot_axes = plot_axes_indicies[i]
            slice_steps = pc_slices[slice_axis]
            plot_bounds_2d = [
                plot_bounds[plot_axes[0]],
                plot_bounds[plot_axes[1]],
            ]
            # save to subdirectory of fig_savedir
            stack_savedir = fig_savedir / f"{filename}_{column_name}_stack"
            stack_savedir.mkdir(parents=True, exist_ok=True)
            plot_flow_field_stack(
                flow_field_dict,
                plot_axes_indicies=plot_axes,
                slice_axis_index=slice_axis,
                plot_bounds=plot_bounds_2d,
                slice_steps=slice_steps,
                fig_savedir=stack_savedir,
                feature_labels=column_labels,
            )

    if len(stable_fixed_points) == 0:
        logger.warning(
            "No stable fixed points found for dataset [ %s ]; plotting slices at mean of data.",
            dataset_name,
        )
        # plot slices at mean of data at last time point
        mean_at_last_timepoint = df[df[Column.TIMEPOINT] == df[Column.TIMEPOINT].max()].mean(
            numeric_only=True
        )
        feature_vals = (
            mean_at_last_timepoint[column_names[2]],
            mean_at_last_timepoint[column_names[1]],
        )  # feature 3, feature 2
        fig, ax = plot_flow_field_slices(
            flow_field_dict,
            dataset_name,
            plot_bounds,
            None,
            prob_kde=prob_kde,
            feature_vals=feature_vals,
            column_names=column_names,
            fig_title=fig_title,
        )
    else:
        for k, fpt in enumerate(stable_fixed_points):
            # plot flow field slices at this stable fixed point
            feature_vals = (fpt[2], fpt[1])  # feature 3, feature 2
            fig, ax = plot_flow_field_slices(
                flow_field_dict,
                dataset_name,
                plot_bounds,
                None,
                prob_kde=prob_kde,
                feature_vals=feature_vals,
                column_names=column_names,
                fig_title=fig_title,
            )

            for j, ax_ in enumerate(ax):  # feature 1 vs feature 2, feature 1 vs feature 3
                ax_.scatter(fpt[0], fpt[j + 1], s=75, color="black")
            # save the figure
            save_plot_to_path(fig, fig_savedir, f"{filename}_fpt_{k}")

    # 2) plot entire trajectory over flow field feature 1 vs feature 2, feature
    # 1 vs feature 3
    # 3) same plot with equally spaced interpolated points along the trajectory
    # overlaid in red
    interpolated_points = interpolate_on_curve(traj)

    # need to account for possible wrap-around in the trajectory due
    # to periodic boundary conditions along circular features (e.g., PC angles)
    # when plotting the trajectory over the flow field slices
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        polar_angle_index = column_names.index(Column.DiffAEData.POLAR_ANGLE)
        polar_angle_range = POLAR_ANGLE_RANGE
        traj[:, polar_angle_index] = rewrap_polar_angle(
            traj[:, polar_angle_index], polar_angle_range
        )
        interpolated_points[:, polar_angle_index] = rewrap_polar_angle(
            interpolated_points[:, polar_angle_index], polar_angle_range
        )

    for j, ax_ in enumerate(ax):
        if Column.DiffAEData.POLAR_ANGLE in column_names:
            # identify where the trajectory wraps around by looking for large
            # jumps in the circular feature (large enough that they exceed the
            # threshold for half the range of the circular feature)
            abs_diff_along_circular_feature = np.abs(np.diff(traj[:, polar_angle_index]))
            diff_threshold = (polar_angle_range[1] - polar_angle_range[0]) / 2
            wrap_around_mask = abs_diff_along_circular_feature > diff_threshold
            # get the indices where the trajectory wraps around, and add 1 to
            # get the index of the point after the wrap (where the trajectory
            # reappears on the other side of the plot)
            wrap_around_indices = np.where(wrap_around_mask)[0] + 1
            # split the trajectory into segments at these indices
            traj_segments = np.split(traj, wrap_around_indices, axis=0)
            for segment in traj_segments:
                ax_.plot(segment[:, 0], segment[:, j + 1], linewidth=2.5, color="navy")
        else:
            ax_.plot(traj[:, 0], traj[:, j + 1], linewidth=2.5, color="navy")

    # save the figure
    save_plot_to_path(fig, fig_savedir, f"{filename}_traj")

    # 3) trajectory with equally spaced interpolated points
    for j, ax_ in enumerate(ax):
        ax_.scatter(
            interpolated_points[:, 0],
            interpolated_points[:, j + 1],
            s=10,
            color="red",
        )

    # save the figure
    save_plot_to_path(fig, fig_savedir, f"{filename}_traj_interpolated")
