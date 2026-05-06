"""Thin save-to-disk wrappers around :mod:`endo_pipeline.library.visualize.model_qc_plots`.

All figure-creation logic lives in ``model_qc_plots``; this module only
adds the example-ID filename generation and the save / close step.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.model_qc_plots import (
    create_contact_sheet_with_metrics_column,
    create_denoising_contact_sheet,
    create_summary_contact_sheet,
)
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.plot_defaults import (
    MODEL_QC_FIG_KWARGS,
    MODEL_QC_GRIDSPEC_KWARGS,
    MODEL_QC_SUBPLOT_KWARGS,
)

if TYPE_CHECKING:
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey
    from endo_pipeline.settings.examples import ExampleImage

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist yet.

    Parameters
    ----------
        Directory path to create.

    Returns
    -------
        The same path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _example_id(example: "ExampleImage") -> str:
    """Build a compact string identifier for an example image.

    Parameters
    ----------
    example
        The example image metadata.

    Returns
    -------
        Identifier like ``"20250818_20XP0T5X100Y200"``.
    """
    return (
        f"{example.dataset_name}P{example.position}T{example.timepoint}"
        f"X{example.crop_x_start}Y{example.crop_y_start}"
    )


def save_negative_control_sheet(
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    images_to_denoise: list[np.ndarray],
    denoised_images: list[np.ndarray],
    denoised_scrambled_emb: list[np.ndarray],
    denoised_scrambled_input: list[np.ndarray],
    label_for_conditioning: str,
    noise_levels: list[float],
    output_path: Path,
    example: "ExampleImage",
) -> None:
    """Create and save the negative-control contact sheet for one example.

    Delegates figure creation to
    :func:`~endo_pipeline.library.visualize.model_qc_plots.create_denoising_contact_sheet`.

    Parameters
    ----------
    conditioning_input_crop : np.ndarray
        Cropped conditioning input image.
    diffusion_input_crop : np.ndarray
        Cropped ground-truth diffusion input image.
    images_to_denoise : list of np.ndarray
        Noised images at each noise level.
    denoised_images : list of np.ndarray
        Denoised outputs from normal conditioning.
    denoised_scrambled_emb : list of np.ndarray
        Denoised outputs from scrambled embedding.
    denoised_scrambled_input : list of np.ndarray
        Denoised outputs from scrambled input image.
    label_for_conditioning : str
        Label for the conditioning channel (e.g. ``"Brightfield"``).
    noise_levels : list of float
        Fractional noise levels (e.g. ``[0.25, 0.5, 0.75]``).
    output_path : Path
        Directory where the figure will be saved.
    example : ExampleImage
        The example metadata used in the filename.
    """
    _ensure_dir(output_path)

    denoising_results: dict[str, list[np.ndarray]] = {
        "images_to_denoise": images_to_denoise,
        "denoised_normal": denoised_images,
        "denoised_scrambled_embedding": denoised_scrambled_emb,
        "denoised_scrambled_input": denoised_scrambled_input,
    }
    fig = create_denoising_contact_sheet(
        conditioning_input_crop=conditioning_input_crop,
        diffusion_input_crop=diffusion_input_crop,
        denoising_results=denoising_results,
        label_for_conditioning=label_for_conditioning,
        noise_levels=noise_levels,
        font_size_medium=FONTSIZE_MEDIUM,
        font_size_large=FONTSIZE_LARGE,
    )
    save_plot_to_path(fig, output_path, f"denoising_contact_sheet_{_example_id(example)}")
    plt.close(fig)


def save_intermediate_contact_sheet(
    conditioning_input_crop: np.ndarray,
    ground_truth: np.ndarray,
    images_to_denoise: list[np.ndarray],
    denoised_images: list[np.ndarray],
    metrics: list[dict],
    label_for_conditioning: str,
    noise_labels: list[str],
    output_path: Path,
    example: "ExampleImage",
) -> None:
    """Create and save per-example intermediate contact sheet with metrics.

    Delegates figure creation to
    :func:`~endo_pipeline.library.visualize.model_qc_plots.create_contact_sheet_with_metrics_column`.

    Parameters
    ----------
    conditioning_input_crop
        Cropped conditioning input image.
    ground_truth
        Ground-truth diffusion image (squeezed).
    images_to_denoise
        Noised images at each noise level.
    denoised_images
        Denoised outputs at each noise level.
    metrics
        Per-noise-level metric dictionaries.
    label_for_conditioning
        Label for the conditioning channel.
    noise_labels
        Row labels for each noise level.
    output_path
        Directory where the figure will be saved.
    example
        The example metadata used in the filename.
    """
    _ensure_dir(output_path)

    n = len(noise_labels)
    panels = [
        *[conditioning_input_crop.squeeze()] * n,
        *[ground_truth] * n,
        *[img.squeeze() for img in images_to_denoise],
        *[img.squeeze() for img in denoised_images],
    ]
    fig = create_contact_sheet_with_metrics_column(
        panels=panels,
        metrics=metrics,
        num_rows=n,
        num_img_cols=4,
        col_titles=[
            f"{label_for_conditioning} input",
            "Original CDH5",
            "Noised CDH5",
            "Predicted CDH5",
        ],
        row_titles=noise_labels,
        fontsize_medium=FONTSIZE_MEDIUM,
        fontsize_small=FONTSIZE_SMALL,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs=MODEL_QC_FIG_KWARGS,
        direction="top-down first",
        show_row_header_column=True,
    )
    save_plot_to_path(fig, output_path, f"denoising_contact_sheet_{_example_id(example)}")
    plt.close(fig)


def save_summary_figure(
    example_results_100: list[np.ndarray],
    example_metrics_100: list[dict],
    num_examples: int,
    label_for_conditioning: str,
    example_set_label: str,
    model_key: "ModelKey",
    compute_metrics: bool,
    output_path: Path,
) -> None:
    """Create and save the per-example-set summary figure.

    Delegates figure creation to
    :func:`~endo_pipeline.library.visualize.model_qc_plots.create_contact_sheet_with_metrics_column`
    (when metrics are available) or
    :func:`~endo_pipeline.library.visualize.model_qc_plots.create_summary_contact_sheet`.

    The output filename is the same regardless of whether metrics are
    computed.  Per-model disambiguation is handled by ``output_path``,
    which already includes the manifest and resolved run name.

    Parameters
    ----------
    example_results_100
        Flat list of panel images for the summary grid.
    example_metrics_100
        Metrics for each example (used when ``compute_metrics`` is True).
    num_examples
        Number of QC examples in this set.
    label_for_conditioning
        Label for the conditioning channel.
    example_set_label
        Human-readable name for the example set (e.g. ``"validation_positions"``).
    model_key
        ``ModelKey`` providing both identity and a human-readable ``.label``.
    compute_metrics
        Whether to include a metrics column in the figure.
    output_path
        Directory where the figure will be saved.
    """
    _ensure_dir(output_path)

    num_img_cols = 3
    col_titles = [
        f"{label_for_conditioning} input",
        "Original CDH5",
        "Predicted CDH5",
    ]
    row_titles = [f"Example {i + 1}" for i in range(num_examples)]

    if compute_metrics:
        fig = create_contact_sheet_with_metrics_column(
            panels=example_results_100,
            metrics=example_metrics_100,
            num_rows=num_examples,
            num_img_cols=num_img_cols,
            col_titles=col_titles,
            row_titles=row_titles,
            fontsize_medium=FONTSIZE_MEDIUM,
            fontsize_small=FONTSIZE_SMALL,
            subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
            gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
            fig_kwargs={"figsize": ((num_img_cols + 1) * 2, num_examples * 1.8)},
            direction="left-right first",
        )
        fig.suptitle(
            f"100% Noise Denoising - {example_set_label} - {model_key.label}",
            fontsize=FONTSIZE_LARGE,
            y=0.995,
        )
        # legacy layout path: keep plt.tight_layout() applied by save_plot_to_path
        tight_layout = True
    else:
        fig = create_summary_contact_sheet(
            example_results=example_results_100,
            num_examples=num_examples,
            label_for_conditioning=label_for_conditioning,
            font_size_medium=FONTSIZE_MEDIUM,
        )
        # create_summary_contact_sheet builds the figure with constrained layout;
        # mixing plt.tight_layout() emits a UserWarning and disables constrained
        # layout, so opt out here.
        tight_layout = False

    save_plot_to_path(
        fig, output_path, "contact_sheet_predict_all_examples", tight_layout=tight_layout
    )
    plt.close(fig)
