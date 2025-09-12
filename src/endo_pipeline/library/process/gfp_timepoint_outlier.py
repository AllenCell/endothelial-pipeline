import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import DatasetConfig, get_available_zarr_files
from endo_pipeline.io.input import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.settings.image_data import NUM_ZSLICES


def plot_gfp_outliers_std(
    data_np: np.ndarray,
    mean_val: float,
    lower_threshold: float,
    upper_threshold: float,
    dark_outliers: list[int],
    bright_outliers: list[int],
    dataset_name: str,
    position: int,
    num_zslices: int = NUM_ZSLICES,
    n_std: int = 3,
) -> None:
    """
    Plot intensity data with global mean ± N*std thresholds and outliers.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(data_np, label="Intensity", color="black", alpha=0.6)

    # Plot mean + thresholds
    ax.axhline(mean_val, color="blue", linestyle="--", label="Mean")
    ax.axhline(
        lower_threshold, color="red", linestyle="--", label=f"Lower Threshold ({lower_threshold})"
    )
    ax.axhline(upper_threshold, color="orange", linestyle="--", label=f"Upper Threshold ({n_std}σ)")

    # Scatter outliers
    ax.scatter(dark_outliers, data_np[dark_outliers], color="red", label="Dark Outliers", zorder=5)
    ax.scatter(
        bright_outliers, data_np[bright_outliers], color="orange", label="Bright Outliers", zorder=5
    )

    # Annotate grouped outliers by timepoint
    outlier_groups = [("Dark", dark_outliers), ("Bright", bright_outliers)]
    info_lines = ["timepoint: [z-slices]\n"]
    for title, indices in outlier_groups:
        d = {
            t: [i % num_zslices for i in indices if i // num_zslices == t]
            for t in sorted(set(i // num_zslices for i in indices))
        }
        if d:
            info_lines.append(f"{title}:\n" + "\n".join(f"{t}: {z}" for t, z in d.items()))

    if len(info_lines) > 1:
        fig.text(
            1.02,
            0.5,
            "\n\n".join(info_lines),
            fontsize=10,
            va="center",
            ha="left",
            transform=ax.transAxes,
        )

    # Secondary x-axis for timepoints
    def index_to_tp(x):
        return x // num_zslices

    def tp_to_index(t):
        return t * num_zslices

    secax = ax.secondary_xaxis("top", functions=(index_to_tp, tp_to_index))
    secax.set_xlabel("Timepoint (every 25 Z-slices)")
    max_tp = data_np.shape[0] // num_zslices
    secax.set_xticks(np.arange(0, max_tp + 1, 25))

    ax.set_xlabel("Flattened Index (T, Z-slices)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{dataset_name} - Position {position}\n")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

    save_dir = get_output_path(f"gfp_outliers_{n_std}std", dataset_name)
    save_plot_to_path(fig, save_dir, f"gfp_outliers_P{position}")
    plt.close(fig)


def detect_egfp_scope_errors(
    dataset_config: DatasetConfig,
    position: int,
    visualize: bool = False,
    n_std: int = 3,
) -> tuple[list[int], list[int]]:
    """
    Detect EGFP scope errors based on global mean ± N*std thresholds.
    Returns lists of dark and bright outlier indices.
    """
    zarr_files = get_available_zarr_files(dataset_config)
    gfp_zarr = load_zarr_as_dask_array(
        zarr_files[position], channels=["EGFP"], level=1, squeeze=True
    )

    # 1 Compute mean intensity over x/y axes
    intensity_array = gfp_zarr.mean(axis=(-2, -1))
    flattened_img_data = intensity_array.flatten()

    # 2 Convert to pandas Series for rolling median
    data_np = flattened_img_data.compute()

    # Compute global mean/std
    mean_val = np.mean(data_np)
    std_val = np.std(data_np)

    lower_threshold = 100
    upper_threshold = mean_val + n_std * std_val

    # Outlier indices
    dark_outliers = np.where(data_np < lower_threshold)[0].tolist()
    bright_outliers = np.where(data_np > upper_threshold)[0].tolist()

    egfp_scope_error = sorted(
        {int(idx // NUM_ZSLICES) for idx in set(dark_outliers + bright_outliers)}
    )

    if visualize:
        plot_gfp_outliers_std(
            data_np,
            mean_val,
            lower_threshold,
            upper_threshold,
            dark_outliers,
            bright_outliers,
            dataset_config.name,
            position,
            n_std=n_std,
        )

    return egfp_scope_error
