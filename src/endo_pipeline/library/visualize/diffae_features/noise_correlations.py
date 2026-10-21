import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.visualize.diffae_features.flow_field_3d import get_slice_indexes
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONT_FAMILY, FONTSIZE_LARGE, FONTSIZE_MEDIUM
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)

DIFFUSION_TENSOR_CONTOUR_COLORMAP: str = "viridis"
"""Colormap for diffusion tensor entry contour plots in 3D flow field visualization."""

DIFFUSION_TENSOR_CONTOUR_VMIN: float = 0.0
"""Fixed minimum colorbar value for diffusion tensor entry contour plots."""

DIFFUSION_TENSOR_CONTOUR_VMAX: float = 0.75
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
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> None:
    """
    Plot the noise correlation as a function of lag directly from the tidy
    noise statistics dataframe (see `compute_noise_statistics_dataframe`),
    rather than from the full two-timepoint noise correlation function.

    Parameters
    ----------
    noise_statistics_dataframe
        Tidy dataframe of noise correlation statistics, as produced by
        `compute_noise_statistics_dataframe`.
    column_names
        List of feature column names to plot.
    plot_title
        Title for the plot.
    output_path
        Directory where the plot will be saved.
    file_name
        Optional, specific name of the file to save the plot as.
    """
    n_dim = len(column_names)
    for i in range(n_dim):
        for j in range(i, n_dim):
            pair_dataframe = noise_statistics_dataframe[
                (noise_statistics_dataframe[Column.NoiseStatistics.FEATURE_1] == column_names[i])
                & (noise_statistics_dataframe[Column.NoiseStatistics.FEATURE_2] == column_names[j])
            ].sort_values(Column.NoiseStatistics.LAG)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                pair_dataframe[Column.NoiseStatistics.LAG],
                pair_dataframe[Column.NoiseStatistics.CORRELATION],
                "o-",
            )
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Lag")
            ax.set_ylabel("$R_{{ij}}$")
            # add measurement error variance as text if it is not NaN
            measurement_error_variance = pair_dataframe[
                Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE
            ].iloc[0]
            if not pd.isna(measurement_error_variance):
                ax.text(
                    0.95,
                    0.95,
                    f"Measurement error variance: {measurement_error_variance:.5f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
            ax.set_ylim(-1.05, 1.05)
            column_label_i = COLUMN_METADATA[column_names[i]].label or column_names[i]
            column_label_j = COLUMN_METADATA[column_names[j]].label or column_names[j]
            ax.set_title(
                f"{plot_title}\nR_{{ij}} vs lag for (i,j) = ({column_label_i}, {column_label_j})"
            )
            figure_name = file_name or "noise_correlation_vs_lag"
            figure_name = f"{figure_name}_{column_names[i]}_{column_names[j]}"
            save_plot_to_path(fig, output_path, figure_name)


def plot_noise_tensor_contours(
    meshgrid: tuple[np.ndarray, np.ndarray],
    tensor_values: np.ndarray,
    entry_labels: list[str],
    axes_limits: tuple[tuple[float, float], tuple[float, float]],
    x_label: str,
    y_label: str,
    n_rows: int = NROWS_DIFFUSION_TENSOR_CONTOURS,
    n_cols: int = NCOLS_DIFFUSION_TENSOR_CONTOURS,
    figsize: tuple[float, float] = FIGSIZE_DIFFUSION_TENSOR_CONTOURS,
    colormap: str = DIFFUSION_TENSOR_CONTOUR_COLORMAP,
    num_levels: int = DIFFUSION_TENSOR_CONTOUR_LEVELS,
    cbar_num_ticks: int = DIFFUSION_TENSOR_CONTOUR_CBAR_NUM_TICKS,
    vmin: float = DIFFUSION_TENSOR_CONTOUR_VMIN,
    vmax: float = DIFFUSION_TENSOR_CONTOUR_VMAX,
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

    Returns
    -------
    :
        Matplotlib Figure and 2D array of Axes with the contour plots of each
        tensor entry.

    """
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="constrained")
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
        colorbar_ticks = np.linspace(vmin, vmax, cbar_num_ticks)
        fig.colorbar(contour, ax=ax_flat[entry_index], ticks=colorbar_ticks, format="%.2f")
        ax_flat[entry_index].set_title(entry_label, fontsize=FONTSIZE_MEDIUM)

    # hide any unused panels (e.g. fewer entries than n_rows * n_cols)
    for ax_ in ax_flat[len(entry_labels) :]:
        ax_.axis("off")

    ax_grid = ax.reshape(n_rows, n_cols)
    for ax_ in ax_grid[-1, :]:
        ax_.set_xlabel(x_label, fontsize=FONTSIZE_MEDIUM)
    for ax_ in ax_grid[:, 0]:
        ax_.set_ylabel(y_label, fontsize=FONTSIZE_MEDIUM)
    for ax_ in ax_flat[: len(entry_labels)]:
        ax_.set_xlim(*axes_limits[0])
        ax_.set_ylim(*axes_limits[1])
        ax_.set_aspect("auto", adjustable="box")

    return fig, ax


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
