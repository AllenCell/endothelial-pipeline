"""Visualization functions for Model QC workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM
from endo_pipeline.settings.plot_defaults import (
    MODEL_QC_FIG_KWARGS,
    MODEL_QC_GRIDSPEC_KWARGS,
    MODEL_QC_PLOT_DIRECTION,
    MODEL_QC_SUBPLOT_KWARGS,
)
from endo_pipeline.settings.workflow_defaults import (
    IMAGE_METRIC_DATASET_COLORS,
    METRIC_TEXT_BOX_PROPS,
)

if TYPE_CHECKING:
    import matplotlib.figure


# ========================
# Contact Sheet Functions
# ========================


def create_denoising_contact_sheet(
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    denoising_results: dict[str, list[np.ndarray]],
    label_for_conditioning: str,
    noise_levels: list[float],
    font_size_medium: int,
    font_size_large: int,
) -> matplotlib.figure.Figure:
    """
    Create a contact sheet figure showing denoising results at various noise levels.

    Parameters
    ----------
    conditioning_input_crop
        The conditioning input image crop.
    diffusion_input_crop
        The diffusion input image crop (ground truth).
    denoising_results
        Dictionary from run_denoising_experiments containing denoised images.
    label_for_conditioning
        Label for the conditioning input (e.g., "Brightfield" or "CDH5").
    noise_levels
        List of noise levels used.
    font_size_medium
        Medium font size for labels.
    font_size_large
        Large font size for supertitle.

    Returns
    -------
        The contact sheet figure.
    """
    cdh5_labels = [
        "Original CDH5",
        "Noised CDH5",
        f"{label_for_conditioning}\nembedding",
        "Scrambled\nembedding",
        "Scrambled\ninput image",
    ]
    noise_labels = [f"{level * 100:.0f}% Noise" for level in [*noise_levels, 1]]
    num_images_denoised = len(noise_labels)

    images_to_denoise = denoising_results["images_to_denoise"]
    denoised_normal = denoising_results["denoised_normal"]
    denoised_scrambled_embedding = denoising_results["denoised_scrambled_embedding"]
    denoised_scrambled_input = denoising_results["denoised_scrambled_input"]

    panels = [
        *[conditioning_input_crop.squeeze()] * num_images_denoised,
        *[diffusion_input_crop.squeeze()] * num_images_denoised,
        *[img.squeeze() for img in images_to_denoise],
        *[img.squeeze() for img in denoised_normal],
        *[img.squeeze() for img in denoised_scrambled_embedding],
        *[img.squeeze() for img in denoised_scrambled_input],
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=num_images_denoised,
        max_cols=6,
        col_titles=[f"{label_for_conditioning} input", *cdh5_labels],
        row_titles=noise_labels,
        direction=cast(Literal["left-right first", "top-down first"], MODEL_QC_PLOT_DIRECTION),
        font_size=font_size_medium,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs=MODEL_QC_FIG_KWARGS,
    )

    # Adjust the layout to make space for supertitles
    fig.subplots_adjust(top=0.9)
    all_axes = fig.get_axes()
    col_4_pos = all_axes[4].get_position()
    center_x = col_4_pos.x0 + (col_4_pos.width / 2)
    fig.text(
        x=center_x,
        y=0.97,
        s="Predicted CDH5 Images",
        ha="center",
        fontsize=font_size_large,
    )

    return fig


def create_summary_contact_sheet(
    example_results: list[np.ndarray],
    num_examples: int,
    label_for_conditioning: str,
    font_size_medium: int,
) -> matplotlib.figure.Figure:
    """
    Create a summary contact sheet showing 100% noise denoising results.

    Parameters
    ----------
    example_results
        List of images: [cond1, gt1, pred1, cond2, gt2, pred2, ...].
    num_examples
        Number of examples in the set.
    label_for_conditioning
        Label for the conditioning input (e.g., "Brightfield" or "CDH5").
    font_size_medium
        Medium font size for labels.

    Returns
    -------
        The summary contact sheet figure.
    """
    num_cols = 3
    fig = make_contact_sheet(
        panels=example_results,
        max_rows=num_examples,
        max_cols=num_cols,
        col_titles=[
            f"{label_for_conditioning} input",
            "Original CDH5",
            "Predicted CDH5",
        ],
        row_titles=[f"Example {i + 1}" for i in range(num_examples)],
        direction="left-right first",
        font_size=font_size_medium,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs={"figsize": (num_cols * 1.5, num_examples * 1.5)},
        use_constrained_layout=True,
    )

    return fig


def create_comparison_bar_plot(
    models_data: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    filename: str,
    model_labels: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    show_baseline: bool = False,
) -> None:
    """
    Create a comparison bar plot for a specific metric across models.

    Parameters
    ----------
    models_data
        List of model data dictionaries with validation and rep2 metrics.
    metric_key
        Key for the metric (e.g., 'corr', 'ssim', 'lpips').
    ylabel
        Label for the y-axis.
    title
        Title of the plot.
    output_path
        Path to save the plot.
    filename
        Filename for the saved plot.
    model_labels
        List of labels for each model on the x-axis. If None, uses generic
        "Model 1", "Model 2", etc. labels.
    ylim
        Optional y-axis limits as (min, max).
    show_baseline
        Whether to show baseline metrics as horizontal dashed lines.
    """

    # Use provided labels or generate generic ones
    if model_labels is not None:
        model_labels_short = model_labels[: len(models_data)]
    else:
        model_labels_short = [f"Model {i+1}" for i in range(len(models_data))]
    num_models = len(model_labels_short)
    x_pos = np.arange(num_models)
    bar_width = 0.35
    with plt.style.context("endo_pipeline.figure"):
        fig, ax = plt.subplots(figsize=(max(12, num_models * 1.5 + 3), 7))
    validation_means = [m["validation"][f"{metric_key}_mean"] for m in models_data]
    validation_stds = [m["validation"][f"{metric_key}_std"] for m in models_data]
    rep2_means = [m["rep2"][f"{metric_key}_mean"] for m in models_data]
    rep2_stds = [m["rep2"][f"{metric_key}_std"] for m in models_data]
    ax.bar(
        x_pos - bar_width / 2,
        validation_means,
        bar_width,
        yerr=validation_stds,
        capsize=5,
        label="Validation",
        color=IMAGE_METRIC_DATASET_COLORS["validation_positions"],
        alpha=0.8,
    )
    ax.bar(
        x_pos + bar_width / 2,
        rep2_means,
        bar_width,
        yerr=rep2_stds,
        capsize=5,
        label="Rep2",
        color=IMAGE_METRIC_DATASET_COLORS["rep_2_positions"],
        alpha=0.8,
    )

    # Add baseline horizontal lines if available
    if show_baseline and models_data and models_data[0].get("baseline_validation") is not None:
        baseline_val_mean = models_data[0]["baseline_validation"][f"{metric_key}_mean"]
        baseline_rep2_mean = models_data[0]["baseline_rep2"][f"{metric_key}_mean"]

        ax.axhline(
            y=baseline_val_mean,
            color=IMAGE_METRIC_DATASET_COLORS["validation_positions"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Consecutive crops (Val)",
        )
        ax.axhline(
            y=baseline_rep2_mean,
            color=IMAGE_METRIC_DATASET_COLORS["rep_2_positions"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Consecutive crops (Rep2)",
        )

    ax.set_xlabel("Latent Size / Conditioning", fontsize=FONTSIZE_MEDIUM)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)
    ax.set_title(title, fontsize=FONTSIZE_LARGE)
    ax.set_xticks(x_pos)
    # Only rotate labels when there are many models
    rotation = 45 if num_models > 4 else 0
    ha_labels = "right" if num_models > 4 else "center"
    ax.set_xticklabels(
        model_labels_short, fontsize=FONTSIZE_MEDIUM, rotation=rotation, ha=ha_labels
    )
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.subplots_adjust(right=0.72)

    # Color legend: top-right, outside the axes but inside the figure
    ax.legend(
        fontsize=FONTSIZE_MEDIUM,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.9,
    )

    save_plot_to_path(fig, output_path, filename)
    plt.close(fig)


def create_rep2_correlation_bar_plot(
    models_data: list[dict],
    model_labels: list[str],
    output_path: Path,
    filename: str,
    title: str = "Pearson Correlation (Rep 2)",
    ylabel: str = "Pearson Correlation (100% Noise)",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Single-series Rep-2 Pearson-correlation bar chart for the supp. figure.

    Parameters
    ----------
    models_data
        Per-model summary dicts as produced by
        :func:`endo_pipeline.library.model.model_qc.metrics.build_models_data`.
    model_labels
        Short x-axis labels (one per model), e.g. ``["8 BF", "16 BF", ...,
        "1024 CDH5"]``.
    output_path
        Directory to save the figure into.
    filename
        Filename (without extension) for the saved figure.
    title
        Plot title.
    ylabel
        Y-axis label.
    ylim
        Optional explicit y-axis limits.
    """
    if len(model_labels) != len(models_data):
        raise ValueError(
            f"model_labels length ({len(model_labels)}) must match models_data "
            f"length ({len(models_data)})"
        )

    num_models = len(models_data)
    x_pos = np.arange(num_models)
    rep2_means = [m["rep2"]["corr_mean"] for m in models_data]
    rep2_stds = [m["rep2"]["corr_std"] for m in models_data]

    with plt.style.context("endo_pipeline.figure"):
        fig, ax = plt.subplots(figsize=(max(12, num_models * 1.5 + 1), 7))

    ax.bar(
        x_pos,
        rep2_means,
        0.6,
        yerr=rep2_stds,
        capsize=5,
        color=IMAGE_METRIC_DATASET_COLORS["rep_2_positions"],
        alpha=0.85,
    )

    ax.set_xlabel("Latent Size / Conditioning", fontsize=FONTSIZE_MEDIUM)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)
    ax.set_title(title, fontsize=FONTSIZE_LARGE)
    ax.set_xticks(x_pos)
    rotation = 45 if num_models > 4 else 0
    ha_labels = "right" if num_models > 4 else "center"
    ax.set_xticklabels(model_labels, fontsize=FONTSIZE_MEDIUM, rotation=rotation, ha=ha_labels)
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)

    save_plot_to_path(fig, output_path, filename, file_format=".svg")
    plt.close(fig)


def create_contact_sheet_with_metrics_column(
    panels: list,
    metrics: list[dict],
    num_rows: int,
    num_img_cols: int,
    col_titles: list[str],
    row_titles: list[str],
    fontsize_medium: int,
    fontsize_small: int,
    subplot_kwargs: dict,
    gridspec_kwargs: dict,
    fig_kwargs: dict,
    direction: str = "top-down first",
    show_row_header_column: bool = False,
) -> matplotlib.figure.Figure:
    """
    Create a contact sheet with an additional metrics column.

    Parameters
    ----------
    panels
        List of image panels to display
    metrics
        List of metric dictionaries, one per row
    num_rows
        Number of rows
    num_img_cols
        Number of image columns (metrics will be in an additional column)
    col_titles
        Titles for image columns
    row_titles
        Titles for rows
    fontsize_medium
        Medium font size
    fontsize_small
        Small font size
    subplot_kwargs
        Kwargs for subplots
    gridspec_kwargs
        Kwargs for gridspec
    fig_kwargs
        Kwargs for figure
    direction
        Panel organization: "top-down first" (column-major) or "left-right first" (row-major)
    show_row_header_column
        If True, adds a dedicated column for row headers (e.g., noise levels)

    Returns
    -------
        The created figure
    """
    row_header_offset = 1 if show_row_header_column else 0
    total_cols = row_header_offset + num_img_cols + 1

    # Create figure with GridSpec
    if show_row_header_column:
        adjusted_gridspec_kwargs = {
            "hspace": 0.005,
            "wspace": 0.005,
            "left": 0.04,
            "right": 0.99,
            "top": 0.94,
            "bottom": 0.01,
        }
    else:
        adjusted_gridspec_kwargs = {
            "hspace": 0.005,
            "wspace": 0.005,
            "left": 0.04,
            "right": 0.99,
            "top": 0.96,
            "bottom": 0.01,
        }

    adjusted_fig_kwargs = fig_kwargs.copy()
    if show_row_header_column and "figsize" in adjusted_fig_kwargs:
        w, h = adjusted_fig_kwargs["figsize"]
        adjusted_fig_kwargs["figsize"] = (w + 0.5, h)
    fig = plt.figure(**adjusted_fig_kwargs)

    width_ratios = []
    if show_row_header_column:
        width_ratios.append(0.15)
    width_ratios.extend([1] * num_img_cols)
    width_ratios.append(0.4)

    if show_row_header_column:
        height_ratios = [0.08] + [1] * num_rows
    else:
        height_ratios = [0.05] + [1] * num_rows

    gs = GridSpec(
        num_rows + 1,
        total_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        **adjusted_gridspec_kwargs,
    )

    if show_row_header_column:
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")

    for col_idx in range(num_img_cols):
        ax = fig.add_subplot(gs[0, col_idx + row_header_offset])
        ax.text(
            0.5,
            0.5,
            col_titles[col_idx],
            ha="center",
            va="center",
            fontsize=fontsize_medium,
        )
        ax.axis("off")

    ax = fig.add_subplot(gs[0, num_img_cols + row_header_offset])
    ax.text(0.5, 0.5, "Metrics", ha="center", va="center", fontsize=fontsize_medium)
    ax.axis("off")

    for row_idx in range(num_rows):
        if show_row_header_column:
            ax = fig.add_subplot(gs[row_idx + 1, 0])
            ax.text(
                0.5,
                0.5,
                row_titles[row_idx],
                ha="center",
                va="center",
                fontsize=fontsize_small,
                rotation=90,
            )
            ax.axis("off")

        for col_idx in range(num_img_cols):
            if direction == "top-down first":
                panel_idx = col_idx * num_rows + row_idx
            else:
                panel_idx = row_idx * num_img_cols + col_idx

            ax = fig.add_subplot(gs[row_idx + 1, col_idx + row_header_offset])

            if panel_idx < len(panels):
                ax.imshow(panels[panel_idx], cmap="gray", aspect="equal")

            if col_idx == 0 and not show_row_header_column:
                ax.set_ylabel(
                    row_titles[row_idx],
                    fontsize=fontsize_small,
                    rotation=90,
                    ha="right",
                    va="center",
                    labelpad=5,
                )

            ax.axis("off")

        ax = fig.add_subplot(gs[row_idx + 1, num_img_cols + row_header_offset])
        ax.axis("off")

        if row_idx < len(metrics):
            metric_text = (
                f"Corr: {metrics[row_idx]['correlation']:.3f}\n"
                f"LPIPS: {metrics[row_idx]['lpips']:.3f}"
            )
            ax.text(
                0.5,
                0.5,
                metric_text,
                ha="center",
                va="center",
                fontsize=fontsize_small,
                bbox=METRIC_TEXT_BOX_PROPS,
            )

    return fig
