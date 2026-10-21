import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


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
