from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.method_constants import OUTLIER_THRESHOLD


def plot_gfp_outliers_rolling(
    timepoint_means: np.ndarray,
    rolling_median: np.ndarray,
    lower_threshold: np.ndarray,
    upper_threshold: np.ndarray,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    save_dir: Path,
    percent: float = OUTLIER_THRESHOLD,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> None:
    """
    Plot timepoint-level mean intensities with rolling mean ± percentage
    thresholds and outliers.

    Parameters
    ----------
    timepoint_means
        Array of mean intensities for each timepoint.
    rolling_median
        Array of rolling median values for the timepoints.
    lower_threshold
        Array of lower threshold values for the timepoints.
    upper_threshold
        Array of upper threshold values for the timepoints.
    dark_outliers
        Indices of timepoints identified as dark outliers.
    bright_outliers
        Indices of timepoints identified as bright outliers.
    dataset_name
        Name of the dataset being analyzed.
    position
        Position index within the dataset.
    save_dir
        The directory to save the plot to.
    percent
        Threshold percentage for identifying outliers.
    figure_size
        The size of the figure to generate.
    """

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(timepoint_means, label="Intensity", color="black", alpha=0.7)
    ax.plot(rolling_median, label="Rolling median", color="blue", alpha=0.9)
    ax.plot(lower_threshold, color="red", linestyle="--", label=f"Lower {int(percent*100)}%")
    ax.plot(upper_threshold, color="orange", linestyle="--", label=f"Upper {int(percent*100)}%")

    if dark_outliers:
        ax.scatter(
            dark_outliers,
            timepoint_means[dark_outliers],
            color="red",
            label="Dark outliers",
            zorder=5,
        )

    if bright_outliers:
        ax.scatter(
            bright_outliers,
            timepoint_means[bright_outliers],
            color="orange",
            label="Bright outliers",
            zorder=5,
        )

    info_lines = []
    if dark_outliers:
        info_lines.append(f"Dark outliers: {dark_outliers}")
    if bright_outliers:
        info_lines.append(f"Bright outliers: {bright_outliers}")

    if info_lines:
        print("\n".join(info_lines))

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mean VE-cadherin\nintensity in Z-stack (a.u.)")
    ax.tick_params(axis="both", which="major")

    ncols = 4 if not dark_outliers and not bright_outliers else 3
    (lines, labels) = plt.gca().get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        loc="upper left",
        ncol=ncols,
        fontsize=FONTSIZE_XSMALL,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.25,
        borderaxespad=0.25,
    )

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3

    save_plot_to_path(fig, save_dir, f"gfp_outliers_{dataset_name}_P{position}", file_format=".svg")
