"""Visualization functions for Model QC workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from endo_pipeline.io import get_output_path
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.plot_defaults import (
    MODEL_QC_FIG_KWARGS,
    MODEL_QC_GRIDSPEC_KWARGS,
    MODEL_QC_PLOT_DIRECTION,
    MODEL_QC_SUBPLOT_KWARGS,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_QC_LABELS,
    DEFAULT_MODEL_QC_MANIFEST_NAMES,
    DEFAULT_MODEL_QC_RUN_NAMES,
    IMAGE_METRIC_DATASET_COLORS,
    METRIC_TEXT_BOX_PROPS,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from endo_pipeline.library.model.model_qc.evaluation import ModelKey


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


def create_validation_examples_contact_sheet(
    conditioning_crops: list[np.ndarray],
    diffusion_crops: list[np.ndarray],
    denoised_by_conditioning: list[np.ndarray],
    denoised_scrambled_latent: list[np.ndarray],
    denoised_scrambled_input: list[np.ndarray],
    label_for_conditioning: str,
    *,
    pixel_size: float,
    figure_width: float,
    figure_height: float = 3.4,
    scalebar_um: int = 10,
    scalebar_location: str = "lower right",
    max_rows: int = 3,
    font_size: int = 10,
) -> matplotlib.figure.Figure:
    """Build the publication-styled DiffAE validation contact sheet.

    One row per validation example, with five columns: the conditioning
    encoder input, the target VE-cadherin, and three denoised predictions
    (true conditioning latent + two negative controls). Used as the panel-A
    contact sheet for both the VE-cadherin-conditioned model figure
    (``supp-fig-diffae-model``) and the brightfield-conditioned schematic
    figure (``supp-fig-diffae-schematic``).

    The conditioning channel differs by model (brightfield vs. VE-cadherin),
    so the column labels are parametrized via ``label_for_conditioning``; the
    diffusion target is always VE-cadherin.

    Parameters
    ----------
    conditioning_crops
        Per-example conditioning input crops (column 1).
    diffusion_crops
        Per-example ground-truth VE-cadherin crops (column 2).
    denoised_by_conditioning
        Per-example denoised outputs from the true conditioning latent (column 3).
    denoised_scrambled_latent
        Per-example denoised outputs from a scrambled latent vector (column 4).
    denoised_scrambled_input
        Per-example denoised outputs from a latent extracted from a
        pixel-scrambled conditioning image (column 5).
    label_for_conditioning
        Label for the conditioning channel (e.g. ``"Brightfield"`` or
        ``"VE-cadherin"``).
    pixel_size
        Physical pixel size for the scale bar.
    figure_width
        Figure width in inches.
    figure_height
        Figure height in inches.
    scalebar_um
        Scale-bar length in microns.
    scalebar_location
        Corner of each panel to place the scale bar in (every image gets a
        bar; only the top-left panel is annotated with the length label).
    max_rows
        Maximum number of example rows.
    font_size
        Font size for column titles and the prediction supertitle.

    Returns
    -------
        The contact sheet figure (caller is responsible for saving).
    """
    titles = [
        f"{label_for_conditioning}\nencoder input",
        "Target\nVE-cadherin",
        f"{label_for_conditioning}\nlatent vector",
        "Scrambled\nlatent vector",
        "Scrambled\ninput image",
    ]

    panels = [
        img
        for img_list in [
            conditioning_crops,
            diffusion_crops,
            denoised_by_conditioning,
            denoised_scrambled_latent,
            denoised_scrambled_input,
        ]
        for img in img_list
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=max_rows,
        max_cols=5,
        col_titles=titles,
        row_titles=None,
        direction=cast(Literal["left-right first", "top-down first"], MODEL_QC_PLOT_DIRECTION),
        font_size=font_size,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs={"figsize": (figure_width, figure_height)},
    )

    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0)
    all_axes = fig.get_axes()
    col_3_pos = all_axes[3].get_position()
    center_x = col_3_pos.x0 + (col_3_pos.width / 2)
    fig.text(
        x=center_x,
        y=0.97,
        s="Predicted VE-cadherin",
        ha="center",
        fontsize=font_size,
    )
    # Scale bar on every image; only the top-left panel carries the text label
    # so the sheet isn't cluttered with the length repeated on every tile.
    for i, ax in enumerate(all_axes):
        if not ax.images:
            continue
        add_scalebar(
            ax,
            pixel_size=pixel_size,
            scale_bar_um=scalebar_um,
            bar_thickness=3,
            padding=5,
            location=scalebar_location,
            include_label=(i == 0),
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
    ylabel: str = "Pearson correlation r value",
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    file_format: Literal[".png", ".svg", ".pdf"] = ".svg",
    label_fontsize: float = FONTSIZE_MEDIUM,
    tick_fontsize: float = FONTSIZE_SMALL,
    save_kwargs: dict | None = None,
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
    output_path, filename, file_format
        Where / how to write the figure.  ``filename`` should not include
        the extension; ``file_format`` provides it.
    ylabel, ylim
        Plot annotations.
    figsize
        Figure size in inches.  Defaults to a width that scales with the
        number of bars when omitted.
    label_fontsize, tick_fontsize
        Override font sizes when this helper is used as a multi-panel
        figure component (so type sizes match the sibling panels).
    save_kwargs
        Extra keyword arguments forwarded to
        :func:`endo_pipeline.io.output.save_plot_to_path` (e.g.
        ``{"pad_inches": 0, "transparent": True}`` to match the
        ``supp_fig_diffae_model`` panel-A export).
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

    if figsize is None:
        figsize = (max(12.0, num_models * 1.5 + 1.0), 7.0)
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x_pos,
        rep2_means,
        0.6,
        yerr=rep2_stds,
        capsize=5,
        color=IMAGE_METRIC_DATASET_COLORS["rep_2_positions"],
        alpha=0.85,
    )

    ax.set_xlabel("Number of latent dimensions, conditioning channel", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xticks(x_pos)
    rotation = 45 if num_models > 4 else 0
    ha_labels = "right" if num_models > 4 else "center"
    ax.set_xticklabels(model_labels, fontsize=tick_fontsize, rotation=rotation, ha=ha_labels)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)

    save_plot_to_path(fig, output_path, filename, file_format=file_format, **(save_kwargs or {}))
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


def create_comparison_plots_and_summary(
    models_data: list[dict[str, Any]],
    model_keys: list[ModelKey],
    seeds_to_evaluate: list[int],
    baseline_data: dict[str, dict[str, float]],
    compute_baseline: bool,
) -> None:
    """Create comparison bar plots and log the summary table.

    Generates one bar plot per metric (correlation, SSIM, LPIPS) comparing
    all models on validation and rep-2 splits, and prints a formatted
    summary table.

    Parameters
    ----------
    models_data
        Per-model summary dicts each containing ``"validation"`` and
        ``"rep2"`` sub-dicts with ``*_mean`` / ``*_std`` floats.
    model_keys
        Ordered list of ``ModelKey``, one per model.  Used for axis labels
        and the legend text in each bar plot.  Must align positionally with
        ``models_data``.
    seeds_to_evaluate
        Seeds used during evaluation; displayed in titles when >1.
    baseline_data
        Baseline mean/std statistics for ``"validation"`` and ``"rep2"``.
        Shown as horizontal dashed lines when ``compute_baseline`` is True.
    compute_baseline
        Whether to overlay baseline reference lines on the bar plots.
    """
    seed_suffix = f"_seeds_{len(seeds_to_evaluate)}" if len(seeds_to_evaluate) > 1 else ""
    comparison_output_path = get_output_path(
        "model_qc",
        "comparison",
        f"models_{len(model_keys)}{seed_suffix}",
    )

    # Determine model labels for bar-plot x-axis ticks.
    # The curated DEFAULT_MODEL_QC_LABELS list is a display override for the
    # specific 10-model latent-dimension sweep (pairs of manifest/run in
    # DEFAULT_MODEL_QC_MANIFEST_NAMES / DEFAULT_MODEL_QC_RUN_NAMES).  Only
    # use curated labels when *all* model keys are members of that sweep;
    # otherwise fall back to each model's own ``ModelKey.label`` (a two-line
    # ``manifest\nrun`` string) so ticks carry meaningful identifiers instead
    # of generic ``"Model N"``.
    #
    # Membership is checked with plain ``(manifest_name, run_name)`` tuples
    # rather than constructing ``ModelKey`` instances: ``ModelKey`` is a
    # ``NamedTuple``, so it compares and hashes equal to a plain tuple of the
    # same field values.  Avoiding the constructor lets ``ModelKey`` be
    # imported only under ``TYPE_CHECKING`` at module top.
    _default_sweep_pairs: set[tuple[str, str]] = set(
        zip(DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True)
    )
    if all((k.manifest_name, k.run_name) in _default_sweep_pairs for k in model_keys):
        # All models belong to the curated sweep; map each key to its
        # corresponding short label by positional order within the sweep.
        sweep_label_map: dict[tuple[str, str], str] = {
            (m, r): lbl
            for m, r, lbl in zip(
                DEFAULT_MODEL_QC_MANIFEST_NAMES,
                DEFAULT_MODEL_QC_RUN_NAMES,
                DEFAULT_MODEL_QC_LABELS,
                strict=True,
            )
        }
        model_labels = [sweep_label_map[(k.manifest_name, k.run_name)] for k in model_keys]
    else:
        model_labels = [k.label for k in model_keys]

    seeds_info = (
        f" (averaged over {len(seeds_to_evaluate)} seeds)" if len(seeds_to_evaluate) > 1 else ""
    )

    metric_configs: list[tuple[str, str, str, dict[str, Any]]] = [
        ("corr", "Pearson Correlation (100% Noise)", "Correlation", {}),
        ("ssim", "SSIM Score (100% Noise)", "SSIM", {}),
        ("lpips", "LPIPS Score (100% Noise)", "LPIPS", {}),
    ]

    # Create comparison plots for each metric
    for metric_key, ylabel, title_base, extra_kw in metric_configs:
        create_comparison_bar_plot(
            models_data=models_data,
            metric_key=metric_key,
            ylabel=ylabel,
            title=f"{title_base}{seeds_info}",
            output_path=comparison_output_path,
            filename=f"{metric_key}_comparison_100_noise",
            model_labels=model_labels,
            show_baseline=compute_baseline,
            **extra_kw,
        )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: Model Performance{seeds_info}")
    print("=" * 80)

    if compute_baseline and baseline_data["validation"]["corr_mean"] > 0:
        print("\nBASELINE (Temporal - Next Timepoint Comparison):")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            b = baseline_data[split_key]
            print(
                f"  {split_label} - Corr: {b['corr_mean']:.3f} ± {b['corr_std']:.3f}, "
                f"SSIM: {b['ssim_mean']:.3f} ± {b['ssim_std']:.3f}, "
                f"LPIPS: {b['lpips_mean']:.3f} ± {b['lpips_std']:.3f}"
            )
        print("-" * 80)

    for model_data in models_data:
        print(f"\n{model_data['model_label']}:")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            d = model_data[split_key]
            print(
                f"  {split_label} - Corr: {d['corr_mean']:.3f} ± {d['corr_std']:.3f}, "
                f"SSIM: {d['ssim_mean']:.3f} ± {d['ssim_std']:.3f}, "
                f"LPIPS: {d['lpips_mean']:.3f} ± {d['lpips_std']:.3f}"
            )
    print("=" * 80)
