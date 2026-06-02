import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import NUM_ZSLICES
from endo_pipeline.settings.method_constants import OUTLIER_THRESHOLD, PARTIAL_DARK_THRESHOLD

logger = logging.getLogger(__name__)


def plot_bf_outliers(
    mean_intensity: np.ndarray,
    rolling_median: np.ndarray,
    dark_threshold: np.ndarray,
    partial_dark_threshold: np.ndarray,
    bright_threshold: np.ndarray,
    dark_outliers: list[int],
    partial_dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    save_dir: Path,
    num_zslices: int = NUM_ZSLICES,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 3),
) -> None:
    """
    Plot intensity data with thresholds and outliers, embedding outlier info on
    the plot.

    Parameters
    ----------
    mean_intensity
        The intensity data flattened indices of timepoints and z-slices.
    rolling_median
        The rolling median of the intensity data.
    dark_threshold
        The lower threshold for detecting dark outliers.
    partial_dark_threshold
        The partial dark threshold for detecting less severe dark outliers.
    bright_threshold
        The upper threshold for detecting bright outliers.
    dark_outliers
        Indices of dark outliers in the data.
    partial_dark_outliers
        Indices of partial dark outliers in the data.
    bright_outliers
        Indices of bright outliers in the data.
    dataset_name
        The name of the dataset being analyzed, used for labeling the plot.
    position
        The position identifier within the dataset, used for labeling the plot.
    save_dir
        The directory to save the plot to.
    num_zslices
        The number of z-slices per timepoint.
    figure_size
        The size of the figure to generate.
    """

    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")

    ax.plot(mean_intensity, label="Intensity", color="black", alpha=0.5)
    ax.plot(
        rolling_median,
        label="Rolling median",
        color="black",
        alpha=1,
        zorder=4,
    )
    ax.plot(
        dark_threshold,
        label=f"Lower {OUTLIER_THRESHOLD*100}%",
        color="red",
        linestyle="--",
    )
    ax.plot(
        partial_dark_threshold,
        label=f"Partial {PARTIAL_DARK_THRESHOLD*100}%",
        color="purple",
        linestyle="--",
    )
    ax.plot(
        bright_threshold,
        label=f"Upper {OUTLIER_THRESHOLD*100}%",
        color="orange",
        linestyle="--",
    )

    ax.scatter(
        dark_outliers, mean_intensity[dark_outliers], color="red", label="Dark outliers", zorder=5
    )
    ax.scatter(
        partial_dark_outliers,
        mean_intensity[partial_dark_outliers],
        color="purple",
        label="Partial dark outliers",
        zorder=5,
    )
    ax.scatter(
        bright_outliers,
        mean_intensity[bright_outliers],
        color="orange",
        label="Bright outliers",
        zorder=5,
    )

    outlier_groups = [
        ("Dark", dark_outliers),
        ("Partial dark", partial_dark_outliers),
        ("Bright", bright_outliers),
    ]

    info_lines = ["timepoint: [z-slices]\n"]
    for title, indices in outlier_groups:
        d = {
            t: [i % num_zslices for i in indices if i // num_zslices == t]
            for t in sorted({i // num_zslices for i in indices})
        }
        if d:
            info_lines.append(f"{title}:\n" + "\n".join(f"{t}: {z}" for t, z in d.items()))

    if len(info_lines) > 1:
        print("\n\n".join(info_lines))

    mean_for_lim = np.mean(mean_intensity)
    ax.set_xlabel("Index (flattened Z-slices)")
    ax.set_ylabel("Mean BF intensity in Z-slice (a.u.)")

    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="both", which="minor")

    # Secondary X-axis for timepoints
    def index_to_tp(x):
        return x // num_zslices

    def tp_to_index(t):
        return t * num_zslices

    secax = ax.secondary_xaxis("top", functions=(index_to_tp, tp_to_index))
    secax.set_xlabel("Time (frames)")
    max_tp = mean_intensity.shape[0] // num_zslices
    secax.set_xticks(np.arange(0, max_tp + 1, 50))
    secax.tick_params(axis="x")

    # Insert a "fake" third entry to get the legend to divide nicely into
    # three columns.
    (lines, labels) = plt.gca().get_legend_handles_labels()
    lines.insert(2, plt.Line2D([0], [0], linestyle="none", marker="none"))
    labels.insert(2, "")
    ax.legend(
        lines,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=FONTSIZE_XSMALL,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    ax.set_ylim(mean_for_lim - mean_for_lim * 0.04, mean_for_lim + mean_for_lim * 0.04)

    # reduce label padding
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    secax.xaxis.labelpad = 3

    save_plot_to_path(
        fig,
        save_dir,
        f"bf_outliers_{dataset_name}_P{position}",
        file_format=".svg",
        tight_layout=False,
    )
