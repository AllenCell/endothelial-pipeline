import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.noise_amplitude import (
    extrapolate_noise_amplitude_tensor,
    get_reshaped_noise_amplitude_and_grid,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.diffae_features.flow_field_3d import get_slice_indexes
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import TIME_STEP_IN_HOURS
from endo_pipeline.settings.figures import (
    FONT_FAMILY,
    FONTSIZE_LARGE,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)

DIFFUSION_TENSOR_CONTOUR_COLORMAP: str = "viridis"
"""Colormap for diffusion tensor entry contour plots in 3D flow field visualization."""

DIFFUSION_TENSOR_CONTOUR_VMIN: float = 0.0
"""Fixed minimum colorbar value for diffusion tensor entry contour plots."""

DIFFUSION_TENSOR_CONTOUR_VMAX: float = 1.0
"""Fixed maximum colorbar value for diffusion tensor entry contour plots."""

DIFFUSION_TENSOR_CONTOUR_LEVELS: int = 25
"""Number of contour levels for diffusion tensor entry contour plots."""

DIFFUSION_TENSOR_CONTOUR_CBAR_NUM_TICKS: int = 5
"""Number of colorbar ticks for diffusion tensor entry contour plots."""

NROWS_DIFFUSION_TENSOR_CONTOURS: int = 2
"""Number of rows of entry panels for the diffusion tensor contour figure (3D state space)."""

NCOLS_DIFFUSION_TENSOR_CONTOURS: int = 3
"""Number of columns of entry panels for the diffusion tensor contour figure (3D state space)."""

FIGSIZE_DIFFUSION_TENSOR_CONTOURS: tuple[int, int] = (13, 8)
"""Figure size for the diffusion tensor entry contour figure."""


def plot_normalized_two_timepoint_cross_correlations(
    cross_correlations: np.ndarray,
    sigma_t: np.ndarray,
    timepoints_range: list[Any],
    column_names: list[str],
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> None:
    # rho_ij(t, t') = R_ij(t, t') / sqrt(R_ii(t, t) * R_jj(t', t')), which
    # rescales to [-1, 1] and divides out any drift in the noise amplitude
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_corr = cross_correlations / (
            sigma_t[:, np.newaxis, :, np.newaxis] * sigma_t[np.newaxis, :, np.newaxis, :]
        )

    # R_ij(t, t') = R_ji(t', t), so only the unique feature pairs are plotted
    n_dim = len(column_names)
    if cross_corr.shape[2] != n_dim or cross_corr.shape[3] != n_dim:
        logger.error(
            "Mismatch between cross-correlation matrix dimensions and number of column names."
        )
        return

    for i in range(n_dim):
        for j in range(i, n_dim):
            # Colormap: plot the normalized cross-correlation matrix:
            # rho_ij(t, t') where the x axis is t and the y axis is t'
            corr_matrix = cross_corr[:, :, i, j]
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.pcolormesh(
                corr_matrix.T,
                cmap="coolwarm",
                norm=colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
                shading="auto",
            )
            fig.colorbar(cax)
            ax.set_xlabel("Timepoint $t$")
            ax.set_xticks(
                np.arange(len(timepoints_range))[::25] + 0.5,
                labels=timepoints_range[::25],
            )
            ax.set_ylabel("Timepoint $t'$")
            ax.set_yticks(
                np.arange(len(timepoints_range))[::25] + 0.5,
                labels=timepoints_range[::25],
            )
            column_label_i = COLUMN_METADATA[column_names[i]].label or column_names[i]
            column_label_j = COLUMN_METADATA[column_names[j]].label or column_names[j]
            ax.set_title(
                f"{plot_title}\n{Unicode.RHO}$_{{ij}}(t, t')$ for $(i,j)$ = ({column_label_i}, {column_label_j})"
            )
            figure_name = file_name or "noise_correlation_matrix"
            figure_name = f"{figure_name}_{column_names[i]}_{column_names[j]}"
            save_plot_to_path(fig, output_path, figure_name)


def plot_cross_correlations_against_lag(
    cross_correlations: np.ndarray,
    column_names: list[str],
    timepoints_array: np.ndarray,
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> None:
    # R_ij(t, t') = R_ji(t', t), so only the unique feature pairs are plotted
    n_dim = len(column_names)
    if cross_correlations.shape[2] != n_dim or cross_correlations.shape[3] != n_dim:
        logger.error(
            "Mismatch between cross-correlation matrix dimensions and number of column names."
        )
        return

    for i in range(n_dim):
        for j in range(i, n_dim):
            corr_matrix = cross_correlations[:, :, i, j]
            # Compute the time lag matrix tau = t' - t for all pairs of timepoints
            tau = timepoints_array[np.newaxis, :] - timepoints_array[:, np.newaxis]
            # R_ii is symmetric in tau, so keep only non-negative lags
            keep = tau >= 0 if i == j else np.ones_like(tau, dtype=bool)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(tau[keep], corr_matrix[keep], alpha=0.2, s=10, zorder=1)
            ax.set_xlabel(f"{Unicode.TAU}$= t' - t$")
            ax.set_ylabel("$R_{{ij}}(t, t')$")
            column_label_i = COLUMN_METADATA[column_names[i]].label or column_names[i]
            column_label_j = COLUMN_METADATA[column_names[j]].label or column_names[j]
            ax.set_title(
                f"{plot_title}\nR_{{ij}}(t, t') for (i,j) = ({column_label_i}, {column_label_j})"
            )
            figure_name = file_name or "noise_correlation_vs_tau"
            figure_name = f"{figure_name}_{column_names[i]}_{column_names[j]}"
            save_plot_to_path(fig, output_path, figure_name)


def plot_noise_correlation_vs_lag(
    noise_statistics_dataframe: pd.DataFrame,
    column_names: list[str],
    output_path: Path,
    file_format: Literal[".png", ".pdf", ".svg"] = ".png",
    plot_title: str | None = None,
    file_name: str | None = None,
    single_dataset: bool = True,
    figure_size: tuple[float, float] = (6.25, 3.25),
) -> Path:
    """
    Plot the noise correlation as a function of lag directly from the noise
    statistics dataframe (see `compute_noise_statistics_dataframe`), rather than
    from the full two-timepoint noise correlation function.

    # Multi-dataset summary

    If `single_dataset` is False, the function will compute and plot the mean
    noise correlation over all datasets. It will also include the mean
    measurement error variance, with its (5, 95) percentiles over all
    datasets, as text on the plot. The plots also show the (5, 95) percentiles
    of the noise correlation over all datasets.

    Parameters
    ----------
    noise_statistics_dataframe
        Dataframe of noise correlation statistics, as produced by
        `compute_noise_statistics_dataframe`.
    column_names
        List of feature column names to plot.
    output_path
        Directory where the plot will be saved.
    plot_title
        Optional, title for the plot.
    file_name
        Optional, specific name of the file to save the plot as.
    single_dataset
        True if the noise statistics dataframe corresponds to a single dataset,
        False otherwise.

    Returns
    -------
    :
        Path to the saved plot.
    """
    if not single_dataset:
        noise_statistics_dataframe = (
            noise_statistics_dataframe.groupby(
                [
                    Column.NoiseStatistics.FEATURE_1,
                    Column.NoiseStatistics.FEATURE_2,
                    Column.NoiseStatistics.LAG,
                ]
            )
            .agg(
                CORRELATION_MEAN=(Column.NoiseStatistics.CORRELATION, "mean"),
                CORRELATION_5=(Column.NoiseStatistics.CORRELATION, lambda x: x.quantile(0.05)),
                CORRELATION_95=(Column.NoiseStatistics.CORRELATION, lambda x: x.quantile(0.95)),
                MEASUREMENT_NOISE_COVARIANCE_MEAN=(
                    Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE,
                    "mean",
                ),
                MEASUREMENT_NOISE_COVARIANCE_5=(
                    Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE,
                    lambda x: x.quantile(0.05),
                ),
                MEASUREMENT_NOISE_COVARIANCE_95=(
                    Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE,
                    lambda x: x.quantile(0.95),
                ),
            )
            .reset_index()
        )
    correlation_column = (
        "CORRELATION_MEAN" if not single_dataset else Column.NoiseStatistics.CORRELATION
    )

    n_dim = len(column_names)
    n_off_diagonal = n_dim * (n_dim - 1) // 2
    n_cols = max(n_dim, n_off_diagonal)
    fig, ax = plt.subplots(
        2, n_cols, figsize=figure_size, squeeze=False, sharex=True, layout="constrained"
    )
    diagonal_index = 0
    off_diagonal_index = 0
    for i in range(n_dim):
        for j in range(i, n_dim):
            # top row holds i=j (diagonal) panels, bottom row holds off-diagonal panels
            if i == j:
                axis = ax[0, diagonal_index]
                diagonal_index += 1
            else:
                axis = ax[1, off_diagonal_index]
                off_diagonal_index += 1
            pair_dataframe = noise_statistics_dataframe[
                (noise_statistics_dataframe[Column.NoiseStatistics.FEATURE_1] == column_names[i])
                & (noise_statistics_dataframe[Column.NoiseStatistics.FEATURE_2] == column_names[j])
            ].sort_values(Column.NoiseStatistics.LAG)

            axis.plot(
                pair_dataframe[Column.NoiseStatistics.LAG] * TIME_STEP_IN_HOURS,
                pair_dataframe[correlation_column],
                color="b",
                marker=".",
                linestyle="-",
                linewidth=1.0,
            )
            if not single_dataset:
                axis.fill_between(
                    pair_dataframe[Column.NoiseStatistics.LAG] * TIME_STEP_IN_HOURS,
                    pair_dataframe["CORRELATION_5"],
                    pair_dataframe["CORRELATION_95"],
                    color="b",
                    alpha=0.35,
                )
            axis.axhline(0, color="k", linewidth=0.5, alpha=0.75, linestyle="--")
            column_label_i = COLUMN_METADATA[column_names[i]].label or column_names[i]
            column_label_j = COLUMN_METADATA[column_names[j]].label or column_names[j]
            axis.set_ylabel(
                f"{Unicode.KAPPA}$_\mathregular{{{column_label_i},{column_label_j}}}$({Unicode.TAU})"
            )
            # measurement error variance is only defined for diagonal (i == j) terms
            measurement_error_variance = pair_dataframe[
                (
                    "MEASUREMENT_NOISE_COVARIANCE_MEAN"
                    if not single_dataset
                    else Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE
                )
            ].iloc[0]
            if i == j and not pd.isna(measurement_error_variance):
                sigma_varepsilon_text = (
                    f"{Unicode.SIGMA}$_\\varepsilon$: {measurement_error_variance:.2e}"
                )
                if not single_dataset:
                    measurement_error_variance_5 = pair_dataframe[
                        "MEASUREMENT_NOISE_COVARIANCE_5"
                    ].iloc[0]
                    measurement_error_variance_95 = pair_dataframe[
                        "MEASUREMENT_NOISE_COVARIANCE_95"
                    ].iloc[0]
                    sigma_varepsilon_text += f" ({measurement_error_variance_5:.2e}, {measurement_error_variance_95:.2e})"
                axis.text(
                    0.95,
                    0.95,
                    sigma_varepsilon_text,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    size=FONTSIZE_XSMALL,
                )
            axis.set_ylim(-1.05, 1.05)
    for ax_ in ax[0, diagonal_index:]:
        ax_.axis("off")
    for ax_ in ax[1, off_diagonal_index:]:
        ax_.axis("off")
    # Add x axis label to the bottom row of panels
    for ax_ in ax[1, :]:
        ax_.set_xlabel(f"Lag {Unicode.TAU} (hours)", fontsize=FONTSIZE_SMALL)
    if plot_title is not None:
        fig.suptitle(plot_title)
    figure_name = file_name or "noise_correlation_vs_lag"
    return save_plot_to_path(
        fig, output_path, figure_name, tight_layout=False, file_format=file_format
    )


def plot_noise_tensor_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    tensor_values: np.ndarray,
    entry_labels: list[str],
    axes_limits: tuple[tuple[float, float], tuple[float, float]],
    x_label: str,
    y_label: str,
    x_ticks: list[float] | None = None,
    x_tick_labels: list[str] | None = None,
    y_ticks: list[float] | None = None,
    y_tick_labels: list[str] | None = None,
    n_rows: int = NROWS_DIFFUSION_TENSOR_CONTOURS,
    n_cols: int = NCOLS_DIFFUSION_TENSOR_CONTOURS,
    figsize: tuple[float, float] = FIGSIZE_DIFFUSION_TENSOR_CONTOURS,
    colormap: str = DIFFUSION_TENSOR_CONTOUR_COLORMAP,
    num_levels: int = DIFFUSION_TENSOR_CONTOUR_LEVELS,
    cbar_num_ticks: int = DIFFUSION_TENSOR_CONTOUR_CBAR_NUM_TICKS,
    vmin: float = DIFFUSION_TENSOR_CONTOUR_VMIN,
    vmax: float = DIFFUSION_TENSOR_CONTOUR_VMAX,
    axis: plt.Axes | None = None,
    show_colorbar: bool = True,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Make contour plots of each unique entry of a diffusion/noise amplitude
    tensor over a single 2D slice of the 3D state space.

    Parameters
    ----------
    meshgrid
        Tuple (x_grid, y_grid) of 2D arrays with the grid coordinates of the
        slice.
    tensor_values
        Tensor entry values evaluated on the slice, with shape
        ``(nx, ny, len(entry_labels))``.
    entry_labels
        Title for each tensor entry panel, e.g. ``"{Unicode.SIGMA}(r, r)"``.
    axes_limits
        Tuple of (x_limits, y_limits) shared across all panels.
    x_label
        Label for the x-axis, set on the bottom row of panels only.
    y_label
        Label for the y-axis, set on the left column of panels only.
    x_ticks
        Optional, override the default x-axis tick positions, applied to
        every panel.
    x_tick_labels
        Optional, override the default x-axis tick labels for `x_ticks`.
    y_ticks
        Optional, override the default y-axis tick positions, applied to
        every panel.
    y_tick_labels
        Optional, override the default y-axis tick labels for `y_ticks`.
    n_rows
        Number of rows of entry panels.
    n_cols
        Number of columns of entry panels.
    figsize
        Size of the figure, specified as a tuple (width, height).
    colormap
        Colormap to use for the contour plots.
    num_levels
        Number of contour levels to use in each panel.
    cbar_num_ticks
        Number of ticks to use in the colorbar for each panel.
    vmin
        Fixed minimum colorbar value, shared across all panels.
    vmax
        Fixed maximum colorbar value, shared across all panels.
    axis
        Optional, existing Axes to draw the single tensor entry on instead of
        creating a new figure. Only valid when `entry_labels` has one entry.
    show_colorbar
        False to skip drawing a colorbar for every panel in this call, e.g.
        when a shared colorbar is drawn separately for only one panel.

    Returns
    -------
    :
        Matplotlib Figure and 2D array of Axes with the contour plots of each
        tensor entry.

    """
    if axis is not None:
        if len(entry_labels) != 1:
            raise ValueError("`axis` can only be provided when plotting a single tensor entry.")
        n_rows, n_cols = 1, 1
        fig = axis.get_figure()
        ax = np.array([[axis]])
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, layout="constrained")
    ax_flat = ax.ravel()

    for entry_index, entry_label in enumerate(entry_labels):
        values = tensor_values[..., entry_index]
        if np.all(np.isnan(values)):
            logger.warning(
                "Tensor entry '%s' is all-NaN in this slice; skipping panel.", entry_label
            )
            ax_flat[entry_index].set_title(entry_label, fontsize=FONTSIZE_MEDIUM)
            continue

        contour_levels = np.linspace(vmin, vmax, num_levels)
        contour = ax_flat[entry_index].contourf(
            meshgrid[0], meshgrid[1], values, levels=contour_levels, cmap=colormap, extend="both"
        )
        if show_colorbar:
            colorbar_ticks = np.linspace(vmin, vmax, cbar_num_ticks)
            fig.colorbar(contour, ax=ax_flat[entry_index], ticks=colorbar_ticks, format="%.2f")
        ax_flat[entry_index].set_title(entry_label, fontsize=FONTSIZE_SMALL)

    # hide any unused panels (e.g. fewer entries than n_rows * n_cols)
    for ax_ in ax_flat[len(entry_labels) :]:
        ax_.axis("off")

    ax_grid = ax.reshape(n_rows, n_cols)
    for ax_ in ax_grid[-1, :]:
        ax_.set_xlabel(x_label, fontsize=FONTSIZE_SMALL)
    for ax_ in ax_grid[:, 0]:
        ax_.set_ylabel(y_label, fontsize=FONTSIZE_SMALL, rotation=0)
    for ax_ in ax_flat[: len(entry_labels)]:
        ax_.set_xlim(*axes_limits[0])
        ax_.set_ylim(*axes_limits[1])
        ax_.set_box_aspect(1)
        if x_ticks is not None:
            ax_.set_xticks(x_ticks, labels=x_tick_labels)
        if y_ticks is not None:
            ax_.set_yticks(y_ticks, labels=y_tick_labels)

    return fig, ax


def plot_noise_tensor_contours_at_fixed_point(
    noise_amplitude_dataframe: pd.DataFrame,
    column_names: list[str],
    fixed_point: tuple[float, float, float],
    output_path: Path,
    file_name: str = "noise_tensor_contour_fixed_point",
    plot_title: str | None = None,
    figure_size: tuple[float, float] = (5.0, 2.5),
) -> Path:
    """Plot contours of the sigma_11 noise tensor entry over two 2D slices of
    state space through a fixed point, in a single figure with separate axes.

    `column_names` is assumed to be ordered `(theta, r, rho)`, matching
    `DYNAMICS_COLUMN_NAMES`. The two panels are:

    - sigma_11(theta, r) at rho = rho*
    - sigma_11(r, rho) at theta = theta*

    Parameters
    ----------
    noise_amplitude_dataframe
        Dataframe of the kernel-smoothed noise amplitude tensor field, as
        produced by `create_noise_amplitude_vector_field_df`.
    column_names
        List of the three dynamics column names `(theta, r, rho)` used to
        compute the noise amplitude tensor.
    fixed_point
        Tuple `(theta_star, r_star, rho_star)` with the coordinates of the
        fixed point to slice through, matching the order of `column_names`.
    output_path
        Directory where the plot will be saved.
    file_name
        Name of the output file.
    figsize
        Size of the combined figure.

    Returns
    -------
    :
        Path to the saved plot.
    """
    theta_star, _, rho_star = fixed_point

    sigma, grid_points_1d = get_reshaped_noise_amplitude_and_grid(
        noise_amplitude_dataframe, column_names
    )
    # fill NaN bins (too few samples) before slicing, so sparse per-bin
    # estimates don't leave the slices blank
    sigma = extrapolate_noise_amplitude_tensor(sigma, grid_points_1d)
    grid = np.meshgrid(*grid_points_1d, indexing="ij")

    column_labels = [get_label_for_column(column) for column in column_names]
    sigma_11_label = f"{Unicode.SIGMA}$_\mathregular{{{column_labels[0]},{column_labels[0]}}}$({','.join(column_labels)})"
    axes_bounds = [(points[0], points[-1]) for points in grid_points_1d]
    theta_metadata = COLUMN_METADATA[column_names[0]]

    fig, axs = plt.subplots(1, 2, figsize=figure_size, layout="constrained")

    for (
        ax,
        (x_index, y_index),
        slice_axis_index,
        slice_value,
        show_colorbar,
        x_ticks,
        x_tick_labels,
        y_ticks,
        y_tick_labels,
    ) in [
        (
            axs[0],
            (0, 1),
            2,
            rho_star,
            False,
            theta_metadata.ticks[::2],
            theta_metadata.tick_labels[::2],
            None,
            None,
        ),
        (axs[1], (1, 2), 0, theta_star, True, None, None, None, None),
    ]:
        slice_indexes = get_slice_indexes(grid[slice_axis_index], slice_value)
        slice_shape = [len(np.unique(slice_indexes[d])) for d in range(len(slice_indexes))]
        collapse_axis = np.where(np.array(slice_shape) == 1)[0][0]

        x_grid = np.take(grid[x_index][slice_indexes].reshape(slice_shape), 0, axis=collapse_axis)
        y_grid = np.take(grid[y_index][slice_indexes].reshape(slice_shape), 0, axis=collapse_axis)
        sigma_11_slice = np.take(
            sigma[..., 0, 0][slice_indexes].reshape(slice_shape), 0, axis=collapse_axis
        )[..., np.newaxis]

        axes_title_text = f"{sigma_11_label}\n{column_labels[slice_axis_index]} = {slice_value:.3f}"

        fig, _ = plot_noise_tensor_contours(
            (x_grid, y_grid),
            sigma_11_slice,
            [axes_title_text],
            axes_limits=(axes_bounds[x_index], axes_bounds[y_index]),
            x_label=column_labels[x_index],
            y_label=column_labels[y_index],
            x_ticks=x_ticks,
            x_tick_labels=x_tick_labels,
            y_ticks=y_ticks,
            y_tick_labels=y_tick_labels,
            axis=ax,
            show_colorbar=show_colorbar,
        )

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=FONTSIZE_MEDIUM, fontweight="bold")

    return save_plot_to_path(fig, output_path, file_name, tight_layout=False, file_format=".svg")


def plot_noise_tensor_stack(
    tensor_field: np.ndarray,
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray],
    plot_axes_indices: tuple[int, int],
    slice_axis_index: int,
    plot_bounds: list[tuple[float, float]],
    slice_steps: np.ndarray,
    entry_pairs: list[tuple[int, int]],
    entry_labels: list[str],
    fig_savedir: Path,
    feature_labels: list[str],
    fig_title: str | None = None,
) -> None:
    """Make and save contour plots of diffusion tensor entries over a stack of
    2D slices taken along one axis of the 3D state space.

    Parameters
    ----------
    tensor_field
        Diffusion/noise amplitude tensor field, with shape
        ``(*grid_shape, ndim, ndim)``.
    meshgrid
        Tuple of 3D arrays (xgrid, ygrid, zgrid) for the full 3D state space,
        typically obtained from `np.meshgrid(..., indexing="ij")`.
    plot_axes_indices
        Tuple (i, j) of feature indices to plot on the x and y axes of each
        slice.
    slice_axis_index
        Index of the feature variable to slice over.
    plot_bounds
        List of (min, max) tuples specifying the plot bounds for each feature.
    slice_steps
        Grid values along the slicing axis at which to take 2D slices.
    entry_pairs
        List of (i, j) index pairs identifying each unique tensor entry, e.g.
        from `itertools.combinations_with_replacement`.
    entry_labels
        Title for each tensor entry panel, matching the order of `entry_pairs`.
    fig_savedir
        Directory to save the figures.
    feature_labels
        List of labels for each feature variable in the plot.
    fig_title
        Optional, title for the figure. If None, no title is set.

    """
    i, j = plot_axes_indices
    x_i_grid = meshgrid[i]
    x_j_grid = meshgrid[j]
    x_k_grid = meshgrid[slice_axis_index]

    for n, slice_value in enumerate(slice_steps):
        slice_indexes = get_slice_indexes(x_k_grid, slice_value)
        my_shape = [len(np.unique(slice_indexes[d])) for d in range(len(slice_indexes))]
        which_idx = np.where(np.array(my_shape) == 1)[0][0]

        x1_grid = np.take(x_i_grid[slice_indexes].reshape(my_shape), 0, axis=which_idx)
        x2_grid = np.take(x_j_grid[slice_indexes].reshape(my_shape), 0, axis=which_idx)

        tensor_slice = np.stack(
            [
                np.take(tensor_field[..., a, b][slice_indexes].reshape(my_shape), 0, axis=which_idx)
                for a, b in entry_pairs
            ],
            axis=-1,
        )

        fig, _ = plot_noise_tensor_contours(
            (x1_grid, x2_grid),
            tensor_slice,
            entry_labels,
            axes_limits=(plot_bounds[i], plot_bounds[j]),
            x_label=feature_labels[i],
            y_label=feature_labels[j],
        )
        title = f"{feature_labels[slice_axis_index]} = {slice_value:.4f}"
        if fig_title is not None:
            title = f"{fig_title}\n{title}"
        fig.suptitle(title, fontsize=FONTSIZE_LARGE, fontfamily=FONT_FAMILY)
        # figure uses the constrained layout engine (see plot_diffusion_tensor_contours),
        # which conflicts with save_plot_to_path's default tight_layout call
        save_plot_to_path(fig, fig_savedir, f"noise_tensor_stack_{n}", tight_layout=False)


def plot_polar_theta_noise_versus_polar_r(
    tensor_field: np.ndarray,
    meshgrid: tuple[np.ndarray, ...],
    fig_title: str,
    output_path: Path,
    file_name: str,
) -> Path:
    """
    Plot noise amplitude sigma_11 as a function of polar radius r.

    Plots values of sigma_11 at a single (center) slice over rho, for multiple
    slices of theta. Also plots the mean sigma_11 over all theta slices.

    Parameters
    ----------
    tensor_field
        The noise amplitude tensor field, evaluated on a mesh grid.
    meshgrid
        The meshgrid corresponding to the tensor field.
    fig_title
        The title for the figure.
    output_path
        The directory to save the plot.
    file_name
        The name of the output file.

    Returns
    -------
    :
        The path to the saved plot.
    """
    fig, ax = plt.subplots()
    ax.set_title(fig_title)
    ax.set_xlabel("r")
    ax.set_ylabel("sigma_11")
    center_slice_index = meshgrid[2].shape[2] // 2
    num_theta_slices = meshgrid[0].shape[1] // 10
    r_values = meshgrid[1][0, :, 0]
    mean_sigma_11 = np.mean(tensor_field[:, :, center_slice_index, 0, 0], axis=0)

    for theta_index in range(0, meshgrid[0].shape[1], num_theta_slices):
        theta_value = meshgrid[0][theta_index, 0, 0]
        sigma_11_value = tensor_field[theta_index, :, center_slice_index, 0, 0]
        ax.plot(
            r_values,
            sigma_11_value,
            label=f"theta={theta_value:.3f}",
            linewidth=1.0,
            alpha=0.5,
        )
    ax.plot(
        r_values,
        mean_sigma_11,
        label="mean",
        linestyle="--",
        color="black",
    )
    ax.legend()
    return save_plot_to_path(fig, output_path, file_name)
