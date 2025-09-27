import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig, get_available_zarr_files
from endo_pipeline.io.input import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.workflows.production.image_data import NUM_ZSLICES

THRESHOLD = 1
"""Percentage to use for thresholding dark and bright outliers."""

ROLLING_WINDOW = 12
"""Number of timepoints to use for rolling window calculation (1 hour)."""


def plot_gfp_outliers_rolling(
    tp_means: np.ndarray,
    rolling_mean: np.ndarray,
    lower_threshold: np.ndarray,
    upper_threshold: np.ndarray,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    window: int = ROLLING_WINDOW,
    percent: float = THRESHOLD,
) -> None:
    """
    Plot TP-level mean intensities with rolling mean ± percentage thresholds and outliers.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tp_means, label="TP Mean Intensity", color="black", alpha=0.7)
    ax.plot(rolling_mean, label=f"Rolling Mean (window={window})", color="blue", alpha=0.9)

    # Threshold curves
    ax.plot(lower_threshold, color="red", linestyle="--", label=f"Lower ({int(percent*100)}%)")
    ax.plot(upper_threshold, color="orange", linestyle="--", label=f"Upper ({int(percent*100)}%)")

    # Outliers
    if dark_outliers:
        ax.scatter(
            dark_outliers, tp_means[dark_outliers], color="red", label="Dark Outliers", zorder=5
        )
    if bright_outliers:
        ax.scatter(
            bright_outliers,
            tp_means[bright_outliers],
            color="orange",
            label="Bright Outliers",
            zorder=5,
        )

    # Annotate text
    info_lines = []
    if dark_outliers:
        info_lines.append(f"Dark: {dark_outliers}")
    if bright_outliers:
        info_lines.append(f"Bright: {bright_outliers}")

    if info_lines:
        fig.text(
            1.02,
            0.5,
            "\n".join(info_lines),
            fontsize=10,
            va="center",
            ha="left",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Mean Intensity (across Z)")
    ax.set_title(f"{dataset_name} - Position {position}")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

    save_dir = get_output_path(f"gfp_outliers_{int(percent*100)}pct", dataset_name)
    save_plot_to_path(fig, save_dir, f"gfp_outliers_P{position}")
    plt.close(fig)


def detect_egfp_scope_errors(
    dataset_config: DatasetConfig,
    position: int,
    visualize: bool = False,
    window: int = ROLLING_WINDOW,
    percent: float = THRESHOLD,
) -> list[int]:
    """
    Detect EGFP scope errors based on per-timepoint mean with rolling mean ± percentage thresholds.
    Uses all z-slices in a timepoint to compute the per-tp mean.
    Returns list of outlier timepoints.
    """
    zarr_files = get_available_zarr_files(dataset_config)
    gfp_zarr = load_zarr_as_dask_array(
        zarr_files[position], channels=["EGFP"], level=1, squeeze=True
    )

    # Compute mean intensity across spatial dimensions (Y, X)
    intensity_array = gfp_zarr.mean(axis=(-2, -1))  # now (T, Z)

    # Compute per-timepoint mean (across Z)
    tp_means = intensity_array.mean(axis=1).compute().astype(float)  # shape (T,)

    # Rolling median
    series = pd.Series(tp_means)
    rolling_median = series.rolling(window, center=True).median()

    # Pad edges
    start_val = np.nanmedian(tp_means[:window])
    end_val = np.nanmedian(tp_means[-window:])
    rolling_median.iloc[: window // 2] = start_val
    rolling_median.iloc[-window // 2 :] = end_val
    rolling_median = rolling_median.to_numpy()

    # Thresholds
    lower_threshold = rolling_median * (1 - percent)
    upper_threshold = rolling_median * (1 + percent)

    # Outlier timepoints
    dark_outliers = np.where(tp_means < lower_threshold)[0].tolist()
    bright_outliers = np.where(tp_means > upper_threshold)[0].tolist()

    egfp_scope_error = sorted(set(dark_outliers + bright_outliers))

    if visualize:
        plot_gfp_outliers_rolling(
            tp_means,
            rolling_median,
            lower_threshold,
            upper_threshold,
            dark_outliers,
            bright_outliers,
            dataset_config.name,
            position,
            window=window,
            percent=percent,
        )

    return egfp_scope_error
