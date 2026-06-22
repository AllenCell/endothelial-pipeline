"""Visualization functions for Model QC workflow."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.plot_defaults import (
    MODEL_QC_GRIDSPEC_KWARGS,
    MODEL_QC_PLOT_DIRECTION,
    MODEL_QC_SUBPLOT_KWARGS,
)
from endo_pipeline.settings.workflow_defaults import IMAGE_METRIC_DATASET_COLORS

if TYPE_CHECKING:
    import matplotlib.figure


# ========================
# Contact Sheet Functions
# ========================


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
) -> "matplotlib.figure.Figure":
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
