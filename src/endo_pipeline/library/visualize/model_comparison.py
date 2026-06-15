"""Plotting methods for model comparison workflows."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.settings.colors import MODEL_COMPARISON_EXAMPLE_GROUP_COLORS
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_QC_LABEL_MAP

MODEL_COMPARISON_DENOISE_COLUMNS = {
    "corr": Column.MODEL_COMPARISON_CORRELATION,
    "ssim": Column.MODEL_COMPARISON_SSIM,
    "lpips": Column.MODEL_COMPARISON_LPIPS,
}
"""Mapping of metrics key to specific denoised metric column names."""

MODEL_COMPARISON_BASELINE_METRICS = {
    "corr": Column.MODEL_COMPARISON_BASELINE_CORRELATION,
    "ssim": Column.MODEL_COMPARISON_BASELINE_SSIM,
    "lpips": Column.MODEL_COMPARISON_BASELINE_LPIPS,
}
"""Mapping of metrics key to specific baseline metric column names."""


def plot_model_comparison_bars(
    output_path: Path,
    model_runs: list[tuple[str, str]],
    aggregate_by_group: dict,
    metrics: list[Literal["corr", "ssim", "lpips"]],
    include_baseline: bool = True,
    figure_size: tuple[float, float] = (6.5, 3),
) -> None:
    """
    Plot model comparison bar plots for select metrics.

    This method generate one bar plot per metric comparing aggregated metrics
    across all models, grouped by example groups.

    Parameters
    ----------
    output_path
        Path to save output plots.
    model_runs
        List of model runs as (model_manifest_name, run_name).
    aggregate_df
        Dataframe containing metric data aggregated by example group and seed.
    metrics
        List of metrics to plot.
    include_baseline
        True to include reference baseline values in plot, False otherwise.
    figure_size
        Size of output plots.
    """

    # Set general bar plot layout
    n_groups = len(aggregate_by_group.keys())
    bar_width = 0.9 / n_groups
    x_pos = np.arange(len(model_runs))
    model_run_labels = [DEFAULT_MODEL_QC_LABEL_MAP[mr] for mr in model_runs]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")

        denoise_metric = MODEL_COMPARISON_DENOISE_COLUMNS[metric]
        baseline_metric = MODEL_COMPARISON_BASELINE_METRICS[metric]

        for index, (group_name, group_metrics) in enumerate(aggregate_by_group.items()):
            color = MODEL_COMPARISON_EXAMPLE_GROUP_COLORS[group_name]

            # Get total number of examples aggregated to calculate mean/stdev
            n = " ".join({str(group_metrics[mr]["N"]) for mr in model_runs})

            # Extract mean and stdev for given metric for each model run
            means = [group_metrics[mr][f"{denoise_metric}_mean"] for mr in model_runs]
            stdevs = [group_metrics[mr][f"{denoise_metric}_stdev"] for mr in model_runs]

            # Plot mean and stdev as bar
            offset = bar_width * index
            ax.bar(
                x_pos + offset,
                means,
                bar_width,
                color=color,
                yerr=stdevs,
                capsize=4,
                label=f"{group_name} (n = {n})",
            )

            # Plot baseline values, if requested
            if include_baseline:
                ax.axhline(
                    y=group_metrics[model_runs[0]][f"{baseline_metric}_mean"],
                    color=color,
                    linestyle="--",
                    label=f"{group_name} baseline",
                )

        # Label x axsis
        ax.set_xlabel("Latent Size / Conditioning", fontsize=FONTSIZE_MEDIUM)
        ax.set_xticks(x_pos + ((n_groups - 1) / 2) * bar_width, model_run_labels)
        ax.tick_params(axis="x", labelsize=FONTSIZE_SMALL)

        # Label y axis
        ax.set_ylabel(get_label_for_column(denoise_metric), fontsize=FONTSIZE_MEDIUM)
        ax.tick_params(axis="y", labelsize=FONTSIZE_SMALL)
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend
        ax.legend(
            fontsize=FONTSIZE_XSMALL, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9
        )

        file_name = f"model_comparison_{metric}.png"
        save_plot_to_path(fig, output_path, file_name, tight_layout=False)


def save_model_comparison_summary(
    output_path: Path,
    model_runs: list[tuple[str, str]],
    aggregate_by_group: dict,
    metrics: list[Literal["corr", "ssim", "lpips"]],
    include_baseline: bool = True,
) -> None:
    """
    Plot model comparison bar plots for select metrics.

    This method generate one bar plot per metric comparing all models, grouped
    by the example groups

    Parameters
    ----------
    output_path
        Path to save output plots.
    model_runs
        List of model runs as (model_manifest_name, run_name).
    aggregate_df
        Dataframe containing metric data aggregated by example group and seed.
    metrics
        List of metrics to plot.
    include_baseline
        True to include reference baseline values in summary, False otherwise.
    """

    summary_lines = ["=" * 80, "SUMMARY: Model Comparison", "=" * 80]

    example_groups = sorted(aggregate_by_group.keys())

    if include_baseline:
        summary_lines.append("\nBaseline (Timepoint - Next Timepoint Comparison):")

        for group_name in example_groups:
            metrics_lines = []
            aggregate_metrics = aggregate_by_group[group_name][model_runs[0]]
            n = aggregate_metrics["N"]

            for metric in metrics:
                mean = aggregate_metrics[f"{MODEL_COMPARISON_DENOISE_COLUMNS[metric]}_mean"]
                std = aggregate_metrics[f"{MODEL_COMPARISON_DENOISE_COLUMNS[metric]}_stdev"]
                metrics_lines.append(f"{metric.upper()}: {mean:.3f} ± {std:.3f}")

            summary_lines.append(f"  {group_name} (n = {n}) - {', '.join(metrics_lines)}")

        summary_lines.append("\n" + "-" * 80)

    for model_run in model_runs:
        summary_lines.append(f"\n{' / '.join(model_run)}")

        for group_name in example_groups:
            metrics_lines = []
            aggregate_metrics = aggregate_by_group[group_name][model_run]
            n = aggregate_metrics["N"]

            for metric in metrics:
                mean = aggregate_metrics[f"{MODEL_COMPARISON_DENOISE_COLUMNS[metric]}_mean"]
                std = aggregate_metrics[f"{MODEL_COMPARISON_DENOISE_COLUMNS[metric]}_stdev"]
                metrics_lines.append(f"{metric.upper()}: {mean:.3f} ± {std:.3f}")

            summary_lines.append(f"  {group_name} (n = {n}) - {', '.join(metrics_lines)}")

    summary_lines.append("\n" + "=" * 80)
    summary_text = "\n".join(summary_lines)

    file_path = output_path / "model_performance_summary.txt"
    file_path.write_text(summary_text + "\n")

    print(summary_text)
