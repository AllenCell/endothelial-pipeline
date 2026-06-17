"""Plotting methods for model comparison workflows."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings.colors import MODEL_COMPARISON_EXAMPLE_GROUP_COLORS
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, FONTSIZE_XSMALL
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_COMPARISON_RUNS,
    DEFAULT_MODEL_QC_LABEL_MAP,
)

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


@figure_panel("Contact sheet comparing DiffAE model predictions")
def make_cross_model_comparison_panel(
    output_path: Path,
    num_gpus: int | None = None,
    figure_size: tuple[float, float] = (6.5, 2),
) -> Path:
    """
    Create contact sheet comparing denoising results across model runs.

    Parameters
    ----------
    output_path
        Output path to save contact sheet.
    num_gpus
        Number of GPUs to use. If None, run on CPU.
    figure_size
        Size of contact sheet.

    Returns
    -------
    :
        Path to output contact sheet.
    """

    from typing import cast

    from matplotlib.layout_engine import LayoutEngine
    from numpy.random import default_rng
    from omegaconf import DictConfig

    from endo_pipeline.io import load_model
    from endo_pipeline.io.load_models import instantiate_model_target_class
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.model.model_comparison import (
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
    )
    from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings.examples import DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
    from endo_pipeline.settings.workflow_defaults import RANDOM_SEED

    # Collect examples for panel
    all_diffusion_examples = []  # same three diffusion examples for every model
    all_noise_images = []  # same three noise images for every model
    all_denoised_examples = []

    for model_index, (model_manifest_name, run_name) in enumerate(DEFAULT_MODEL_COMPARISON_RUNS):
        # Load model for run and get model config. First load the model without
        # instantiation to grab the model config, then instantiate for use later.
        model_manifest = load_model_manifest(model_manifest_name)
        model_location = model_manifest.locations[run_name]
        model_ = load_model(model_location, instantiate=False)
        model_config: DictConfig = cast(DictConfig, model_.cfg)
        model = instantiate_model_target_class(model_)

        for example_index, example in enumerate(DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES):
            # On first model pass, also extract target VE-cadherin crops
            # and pre-generate noise images (shared across all models)
            # On the first model, generate the transformed diffusion example
            # image (target VE-cadherin) and the noise image (which will be
            # shared across all models) with a deterministic seed
            if model_index == 0:
                rng = default_rng(seed=RANDOM_SEED + example_index)
                diffusion_ex = load_transformed_diffusion_example_image(example, model_config)
                all_diffusion_examples.append(diffusion_ex)
                all_noise_images.append(rng.standard_normal(size=diffusion_ex.shape))

            # Load transformed conditioning example
            conditioning_ex = load_transformed_conditioning_example_image(example, model_config)

            # Apply noise to conditioning image and then denoise
            noise = all_noise_images[example_index]
            latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=num_gpus)
            denoised_ex = generate_from_coords_and_noised_image(model, latent, noise, num_gpus)

            # Add denoised examples to list for use in contact sheet
            all_denoised_examples.append(denoised_ex)

    # Build panels and set column titles
    panels = [
        *[img.squeeze() for img in all_diffusion_examples],
        *[img.squeeze() for img in all_denoised_examples],
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=len(DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES),
        max_cols=len(DEFAULT_MODEL_COMPARISON_RUNS) + 1,
        direction="top-down first",
        subplot_kwargs={"frame_on": False},
        fig_kwargs={"figsize": figure_size},
        use_constrained_layout=True,
    )

    # Add column titles with adjusted sizing
    titles = ["Target\nVE-cadherin"] + list(DEFAULT_MODEL_QC_LABEL_MAP.values())
    for index, title in enumerate(titles):
        ax = fig.get_axes()[index]
        fontsize = FONTSIZE_SMALL * 0.9 if index == 0 else FONTSIZE_SMALL
        ax.set_title(title, fontsize=fontsize, pad=3)

    # Adjust the layout padding
    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**{"h_pad": 0.02, "w_pad": 0.02})
    fig.canvas.draw()

    # Add scale bar on every image with text label only on the top-left panel
    for i, ax in enumerate(fig.get_axes()):
        add_scalebar(
            ax,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            scale_bar_um=20,
            bar_thickness=3,
            padding=5,
            location="lower right",
            include_label=(i == 0),
        )

    return save_plot_to_path(
        fig,
        output_path,
        "cross_model_comparison",
        file_format=".svg",
        tight_layout=False,
        show_and_close=True,
    )


@figure_panel("Bar plot of Pearson correlation coefficient for model predictions")
def make_model_prediction_correlation_panel(
    output_path: Path,
    figure_size: tuple[float, float] = (6, 2.5),
) -> Path:
    """
    Plot Pearson correlation across seeds and examples for replicate examples.

    Parameters
    ----------
    output_path
        Output path to save bar plot.
    figure_size
        Size of bar plot.

    Returns
    -------
    :
        Path to output bar plot.
    """

    from endo_pipeline.library.model.model_comparison import (
        aggregate_model_comparison_metrics,
        group_aggregate_model_comparison_metrics,
        load_model_comparison_metrics,
    )

    # Set defaults for model runs, example group, and metric
    model_runs = DEFAULT_MODEL_COMPARISON_RUNS
    metric = "corr"
    example_group: Literal["training", "validation", "replicate"] = "replicate"

    # Load all metrics and then aggregate by example group and random seed
    all_metrics_df = load_model_comparison_metrics(model_runs, [example_group])
    aggregate_df = aggregate_model_comparison_metrics(all_metrics_df)
    aggregate_dict = group_aggregate_model_comparison_metrics(aggregate_df)
    metrics_for_group = aggregate_dict[example_group]

    # Set plot positioning and colors
    x_pos = np.arange(len(model_runs))
    model_run_labels = [DEFAULT_MODEL_QC_LABEL_MAP[mr] for mr in model_runs]
    color = MODEL_COMPARISON_EXAMPLE_GROUP_COLORS[example_group]

    # Extract mean and stdev for given metric for each model run
    column_name = MODEL_COMPARISON_DENOISE_COLUMNS[metric]
    means = [metrics_for_group[mr][f"{column_name}_mean"] for mr in model_runs]
    stdevs = [metrics_for_group[mr][f"{column_name}_stdev"] for mr in model_runs]

    # Plot mean and stdev as bar
    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.bar(
        x_pos,
        means,
        0.6,
        yerr=stdevs,
        capsize=5,
        color=color,
    )

    # Label x axis
    ax.set_xlabel("Conditioning channel-Latent dimensions", fontsize=FONTSIZE_MEDIUM)
    ax.set_xticks(x_pos, model_run_labels)
    ax.tick_params(axis="x", labelsize=FONTSIZE_SMALL)

    # Label y axis
    ax.set_ylabel(get_label_for_column(column_name), fontsize=FONTSIZE_MEDIUM)
    ax.tick_params(axis="y", labelsize=FONTSIZE_SMALL)
    ax.grid(True, alpha=0.3, axis="y")

    return save_plot_to_path(
        fig,
        output_path,
        "replicate_correlation_bar_plot",
        file_format=".svg",
        tight_layout=False,
        show_and_close=True,
    )


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

    file_path = output_path / "model_comparison_summary.txt"
    file_path.write_text(summary_text + "\n")

    print(summary_text)
