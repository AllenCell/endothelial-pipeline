import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)

SUMMARY_LABEL_COLUMN = "dataset_and_shear_stress"
"""Temporary column used to place each dataset and flow condition on the summary axis."""


def plot_noise_amplitude(
    sigma_t: np.ndarray,
    timepoints_array: np.ndarray,
    column_names: list[str],
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> Path:
    """
    Plot the noise amplitude for each feature across timepoints.

    ## Input array shapes

    The input array of computed noise amplitudes should be of shape (T, D),
    where T is the number of timepoints and D is the number of features. The
    corresponding input array of timepoints should be of shape (T,), and the
    list of column names should have length D.

    Parameters
    ----------
    sigma_t
        Array of noise amplitudes for each feature across timepoints.
    timepoints_array
        Array of timepoints corresponding to the noise amplitudes.
    column_names
        List of feature names corresponding to the dimensions of the
        noise amplitude array.
    plot_title
        Title for the plot.
    output_path
        Directory where the plot will be saved.
    file_name
        Optional, specific name of the file to save the plot as.
    """
    # sigma_i(t) = sqrt(R_ii(t, t)); stationary noise has a flat sigma_i(t)
    n_dim = len(column_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_dim):
        ax.plot(timepoints_array, sigma_t[:, i], label=column_names[i])
    ax.set_xlabel("Frame number $t$")
    ax.set_ylabel(f"{Unicode.SIGMA}$_i(t) = \\sqrt{{R_{{ii}}(t, t)}}$")
    ax.set_title(plot_title)
    ax.legend()
    figure_name = file_name or "noise_amplitude"
    return save_plot_to_path(fig, output_path, figure_name)


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


def plot_residual_autocorrelation(
    lags: np.ndarray,
    correlations: np.ndarray,
    correlation_bounds: np.ndarray,
    column_names: list[str],
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> Path:
    """
    Plot the residual autocorrelation function against lag, with a surrogate band.

    The shaded band is the percentile interval of the same statistic computed on
    time-permuted surrogates, so an autocorrelation that stays inside the band is
    indistinguishable from white noise at this sample size.

    Parameters
    ----------
    lags
        Array of lags, in frames, of shape (L,).
    correlations
        Normalized residual correlations of shape (L, D, D).
    correlation_bounds
        Surrogate percentile bounds of shape (2, L, D).
    column_names
        List of feature names, of length D.
    plot_title
        Title for the plot.
    output_path
        Directory where the plot will be saved.
    file_name
        Optional, specific name of the file to save the plot as.
    """
    n_dim = len(column_names)
    fig, axes = plt.subplots(1, n_dim, figsize=(5 * n_dim, 4), squeeze=False)

    for i in range(n_dim):
        ax = axes[0, i]
        ax.fill_between(
            lags,
            correlation_bounds[0, :, i],
            correlation_bounds[1, :, i],
            alpha=0.3,
            color="grey",
            label="Surrogate interval",
            zorder=1,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
        ax.plot(lags, correlations[:, i, i], marker="o", zorder=3)
        ax.set_xlabel(f"Lag {Unicode.TAU} (frames)")
        ax.set_ylabel(f"{Unicode.RHO}$_{{ii}}(${Unicode.TAU}$)$")
        column_label = COLUMN_METADATA[column_names[i]].label or column_names[i]
        ax.set_title(column_label)
        if i == 0:
            ax.legend()

    fig.suptitle(plot_title)
    fig.tight_layout()
    figure_name = file_name or "residual_autocorrelation"
    return save_plot_to_path(fig, output_path, figure_name)


def plot_residual_power_spectrum(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    spectral_exponents: np.ndarray,
    column_names: list[str],
    plot_title: str,
    output_path: Path,
    file_name: str | None = None,
) -> Path | None:
    """
    Plot the residual power spectrum with the fitted power law.

    A flat spectrum means white noise. A spectrum that falls with frequency
    indicates persistent noise, while one that rises indicates differenced
    observation error.

    Parameters
    ----------
    frequencies
        Array of frequencies, in inverse frames, of shape (F,).
    power_spectrum
        Mean residual power spectrum of shape (F, D).
    spectral_exponents
        Fitted exponent beta for each feature, of shape (D,).
    column_names
        List of feature names, of length D.
    plot_title
        Title for the plot.
    output_path
        Directory where the plot will be saved.
    file_name
        Optional, specific name of the file to save the plot as.
    """
    if frequencies.size == 0:
        logger.warning("No power spectrum available to plot. Skipping.")
        return None

    n_dim = len(column_names)
    fig, axes = plt.subplots(1, n_dim, figsize=(5 * n_dim, 4), squeeze=False)
    for i in range(n_dim):
        ax = axes[0, i]
        power = power_spectrum[:, i]
        ax.loglog(frequencies, power, marker=".", linestyle="none", zorder=2)

        if np.isfinite(spectral_exponents[i]):
            # anchor the fitted power law to the mean power so it overlays the data
            reference = np.exp(np.nanmean(np.log(power[power > 0]))) if (power > 0).any() else 1.0
            midpoint = np.exp(np.mean(np.log(frequencies)))
            fit = reference * (frequencies / midpoint) ** (-spectral_exponents[i])
            ax.loglog(
                frequencies,
                fit,
                linestyle="--",
                color="black",
                label=f"{Unicode.BETA} = {spectral_exponents[i]:.2f}",
                zorder=3,
            )
            ax.legend()

        ax.set_xlabel("Frequency $f$ (1/frames)")
        ax.set_ylabel("$S(f)$")
        column_label = COLUMN_METADATA[column_names[i]].label or column_names[i]
        ax.set_title(column_label)

    fig.suptitle(plot_title)
    fig.tight_layout()
    figure_name = file_name or "residual_power_spectrum"
    return save_plot_to_path(fig, output_path, figure_name)


def plot_noise_signature_summary(
    statistics: pd.DataFrame,
    statistic_names: list[str],
    column_names: list[str],
    plot_title: str,
    output_path: Path,
    reference_values: dict[str, float] | None = None,
    file_name: str | None = None,
) -> Path | None:
    """
    Compare noise signature statistics across every dataset and flow condition.

    One panel is drawn per statistic, with the datasets along the horizontal
    axis and one series per feature. Where a statistic has a value that
    corresponds to white noise observed without error, that value is drawn as a
    dashed reference line so departures are visible at a glance.

    Parameters
    ----------
    statistics
        Concatenated noise signature dataframe for all datasets.
    statistic_names
        Names of the statistics to draw, one panel each.
    column_names
        List of feature names to draw as separate series.
    plot_title
        Title for the plot.
    output_path
        Directory where the plot will be saved.
    reference_values
        Optional mapping from statistic name to its white noise reference value.
    file_name
        Optional, specific name of the file to save the plot as.
    """
    if statistics.empty:
        logger.warning("No noise signature statistics available to summarize. Skipping.")
        return None

    # one tick per dataset and flow condition, since each is analyzed separately
    labels = (
        statistics[Column.DATASET].astype(str)
        + "\n"
        + statistics[Column.SHEAR_STRESS].map(lambda shear: f"{shear:g}")
    )
    ordered_labels = sorted(labels.unique())
    positions = {label: index for index, label in enumerate(ordered_labels)}

    frame = statistics.assign(**{SUMMARY_LABEL_COLUMN: labels})
    n_statistics = len(statistic_names)
    fig, axes = plt.subplots(
        n_statistics,
        1,
        figsize=(max(6.0, 1.6 * len(ordered_labels)), 3.0 * n_statistics),
        sharex=True,
        squeeze=False,
    )

    for panel_index, statistic_name in enumerate(statistic_names):
        ax = axes[panel_index, 0]
        for column_name in column_names:
            selected = frame[
                (frame[Column.NoiseSignature.STATISTIC] == statistic_name)
                & (frame[Column.NoiseSignature.FEATURE] == column_name)
            ]
            if selected.empty:
                continue
            column_label = COLUMN_METADATA[column_name].label or column_name
            ax.plot(
                [positions[label] for label in selected[SUMMARY_LABEL_COLUMN]],
                selected[Column.NoiseSignature.VALUE].to_numpy(),
                marker="o",
                linestyle="-",
                label=column_label,
                zorder=3,
            )

        if reference_values is not None and statistic_name in reference_values:
            ax.axhline(
                reference_values[statistic_name],
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=2,
            )

        ax.set_ylabel(statistic_name.replace("_", " "))
        if panel_index == 0:
            ax.legend(loc="best", fontsize="small")

    axes[-1, 0].set_xticks(range(len(ordered_labels)), labels=ordered_labels, fontsize="small")
    axes[-1, 0].set_xlabel("Dataset and shear stress")

    fig.suptitle(plot_title)
    fig.tight_layout()
    figure_name = file_name or "noise_signature_summary"
    return save_plot_to_path(fig, output_path, figure_name)
