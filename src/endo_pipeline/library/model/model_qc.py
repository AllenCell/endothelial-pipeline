import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import tifffile
from numpy.random import Generator

logger = logging.getLogger(__name__)


# =====================
# Shared lazy imports
# =====================


def _lazy_imports():
    """Return commonly needed modules (avoids repeating imports everywhere)."""
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL

    return SimpleNamespace(
        load_dataset_config=load_dataset_config,
        load_image=load_image,
        crop_image=crop_image,
        apply_img_transforms=apply_img_transforms,
        create_data_dict_loaded_image=create_data_dict_loaded_image,
        get_image_transforms=get_image_transforms,
        get_target_image_from_sample=get_target_image_from_sample,
        get_zarr_location_for_position=get_zarr_location_for_position,
        DIFFAE_ZARR_RESOLUTION_LEVEL=DIFFAE_ZARR_RESOLUTION_LEVEL,
    )


class SimpleNamespace:
    """Minimal attribute-access wrapper around keyword arguments."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist yet. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================
# Image loading helpers
# =========================


def _load_transformed_image(example, model_config, diffusion_input_key, timepoint=None):
    """Load a zarr image, apply transforms, and return the transform dict."""
    deps = _lazy_imports()

    dataset_config = deps.load_dataset_config(example.dataset_name)
    zarr_loc = deps.get_zarr_location_for_position(dataset_config, example.position)
    tp = timepoint if timepoint is not None else example.timepoint
    img = deps.load_image(
        zarr_loc,
        level=deps.DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=tp,
        squeeze=True,
        compute=True,
    )
    data = deps.create_data_dict_loaded_image(img)
    transforms = deps.get_image_transforms(model_config)
    sample = deps.apply_img_transforms(transforms, data)
    return sample, zarr_loc, transforms


def load_and_preprocess_example_crop(
    example: Any,
    model_config: Any,
    crop_size: int,
    channel_key_for_conditioning_input: str,
    diffusion_input_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image for a given example and preprocess it for model QC.

    Returns (conditioning_input_crop, diffusion_input_crop).
    """
    deps = _lazy_imports()
    sample, _, _ = _load_transformed_image(example, model_config, diffusion_input_key)

    conditioning_img = deps.get_target_image_from_sample(
        sample, target_key=channel_key_for_conditioning_input
    )
    diffusion_img = deps.get_target_image_from_sample(sample, target_key=diffusion_input_key)

    conditioning_crop = deps.crop_image(
        conditioning_img,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )
    diffusion_crop = deps.crop_image(
        diffusion_img,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )
    return conditioning_crop, diffusion_crop


# ==================
# Denoising helpers
# ==================


def _denoise_images(model, latent_vector, images_to_denoise, num_gpus):
    """Run denoising on a list of images with the given latent vector."""
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )

    return [
        generate_from_coords_and_noised_image(model, latent_vector, img, num_gpus=num_gpus)
        for img in images_to_denoise
    ]


def run_denoising_experiments(
    model: Any,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    rng: Generator,
    noise_levels: list[float],
    num_gpus: int,
) -> dict[str, list[np.ndarray]]:
    """Run denoising experiments with normal and negative control conditioning.

    Returns dict with keys: images_to_denoise, noise_image, denoised_normal,
    denoised_scrambled_embedding, denoised_scrambled_input, conditioning_latent.
    """
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import add_noise_to_image

    latent = get_latent_vector_from_crop(model, conditioning_input_crop, num_gpus=num_gpus)
    noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

    noisy_images = [
        add_noise_to_image(diffusion_input_crop, noise_image, level) for level in noise_levels
    ]
    images_to_denoise = [*noisy_images, noise_image]

    # Experiment 1: normal conditioning
    denoised_normal = _denoise_images(model, latent, images_to_denoise, num_gpus)

    # Experiment 2: scrambled latent vector
    latent_scrambled = rng.permuted(latent)
    denoised_scrambled_emb = _denoise_images(model, latent_scrambled, images_to_denoise, num_gpus)

    # Experiment 3: latent from scrambled input image
    img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
        conditioning_input_crop.shape
    )
    latent_from_scrambled = get_latent_vector_from_crop(model, img_scrambled, num_gpus=num_gpus)
    denoised_scrambled_input = _denoise_images(
        model, latent_from_scrambled, images_to_denoise, num_gpus
    )

    return {
        "images_to_denoise": images_to_denoise,
        "noise_image": [noise_image],
        "denoised_normal": denoised_normal,
        "denoised_scrambled_embedding": denoised_scrambled_emb,
        "denoised_scrambled_input": denoised_scrambled_input,
        "conditioning_latent": [latent],
    }


# =================
# TIFF I/O helpers
# =================


def save_image_as_tiff(image, output_path: Path, filename: str) -> None:
    """Save an image array as a TIFF file."""
    output_file = output_path / f"{filename}.tiff"
    image_to_save = image.squeeze()
    if not np.issubdtype(image_to_save.dtype, np.floating):
        image_to_save = image_to_save.astype(np.float32)
    tifffile.imwrite(output_file, image_to_save)


def save_denoising_crops(
    output_path: Path,
    dataset_name: str,
    position: int,
    timepoint: int,
    start_x: int,
    start_y: int,
    conditioning_input_crop,
    diffusion_input_crop,
    noisy_diffusion_input_images: list,
    noise_image,
    denoised_images: list,
    noise_levels: list[float],
) -> Path:
    """Save all denoising crops as TIFF files to a structured directory."""
    crops_output_path = (
        output_path / "crops" / f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}"
    )
    crops_output_path.mkdir(parents=True, exist_ok=True)

    save_image_as_tiff(conditioning_input_crop, crops_output_path, "conditioning_input")
    save_image_as_tiff(diffusion_input_crop, crops_output_path, "ground_truth")

    for noised_img, noise_level in zip(noisy_diffusion_input_images, noise_levels, strict=False):
        save_image_as_tiff(noised_img, crops_output_path, f"noised_{int(noise_level * 100):03d}pct")

    save_image_as_tiff(noise_image, crops_output_path, "noised_100pct")

    for idx, denoised_img in enumerate(denoised_images):
        pct = int(noise_levels[idx] * 100) if idx < len(noise_levels) else 100
        save_image_as_tiff(denoised_img, crops_output_path, f"denoised_from_{pct:03d}pct_noise")

    logger.debug("Saved crops to %s", crops_output_path)
    return crops_output_path


# =====================
# Result dict helpers
# =====================


def _empty_metric_bucket() -> dict:
    """Return an empty metrics bucket for one example set."""
    return {
        "correlations_100": [],
        "ssims_100": [],
        "lpips_100": [],
        "baseline_correlations": [],
        "baseline_ssims": [],
        "baseline_lpips": [],
    }


def _make_result_skeleton(model_idx, model_label, run_name, random_seed) -> dict:
    """Build the top-level result dict with empty metric buckets."""
    return {
        "model_idx": model_idx,
        "model_label": model_label,
        "run_name": run_name,
        "random_seed": random_seed,
        "validation_positions": _empty_metric_bucket(),
        "rep_2_positions": _empty_metric_bucket(),
        "training_positions": _empty_metric_bucket(),
    }


def _make_model_label(manifest_name: str, run_name: str) -> str:
    if len(manifest_name) > 30:
        return f"{manifest_name[:30]}..._{run_name[:10]}"
    return f"{manifest_name}_{run_name}"


# =====================================
# Baseline & per-example metric helpers
# =====================================


def _compute_baseline_for_example(
    example,
    model_config,
    diffusion_input_key,
    crop_size,
    ground_truth,
    lpips_calculator,
):
    """Compare ground truth to next-timepoint CDH5. Returns (corr, ssim, lpips) or None."""
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity as ssim

    deps = _lazy_imports()
    try:
        sample_next, _, _ = _load_transformed_image(
            example,
            model_config,
            diffusion_input_key,
            timepoint=example.timepoint + 1,
        )
        diffusion_next = deps.get_target_image_from_sample(
            sample_next,
            target_key=diffusion_input_key,
        )
        crop_next = deps.crop_image(
            diffusion_next,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        ).squeeze()

        corr = pearsonr(ground_truth.ravel(), crop_next.ravel())[0]
        ssim_val = ssim(
            ground_truth,
            crop_next,
            data_range=ground_truth.max() - ground_truth.min(),
        )
        lpips_val = lpips_calculator.compute(ground_truth, crop_next)
        return corr, ssim_val, lpips_val
    except Exception as e:
        logger.warning(
            "Could not compute baseline for %s P%s T%s -> T%s: %s",
            example.dataset_name,
            example.position,
            example.timepoint,
            example.timepoint + 1,
            e,
        )
        return None


# =================
# Plotting helpers
# =================


def _save_negative_control_sheet(
    conditioning_input_crop,
    diffusion_input_crop,
    images_to_denoise,
    denoised_images,
    denoised_scrambled_emb,
    denoised_scrambled_input,
    label_for_conditioning,
    cdh5_labels,
    noise_labels,
    output_path,
    example,
):
    """Create and save the negative-control contact sheet for one example."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
    from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_FIG_KWARGS,
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_PLOT_DIRECTION,
        MODEL_QC_SUBPLOT_KWARGS,
    )

    n = len(noise_labels)
    panels = [
        *[conditioning_input_crop.squeeze()] * n,
        *[diffusion_input_crop.squeeze()] * n,
        *[img.squeeze() for img in images_to_denoise],
        *[img.squeeze() for img in denoised_images],
        *[img.squeeze() for img in denoised_scrambled_emb],
        *[img.squeeze() for img in denoised_scrambled_input],
    ]
    fig = make_contact_sheet(
        panels=panels,
        max_rows=n,
        max_cols=6,
        col_titles=[f"{label_for_conditioning} input", *cdh5_labels],
        row_titles=noise_labels,
        direction=cast(Literal["left-right first", "top-down first"], MODEL_QC_PLOT_DIRECTION),
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs=MODEL_QC_FIG_KWARGS,
    )
    fig.subplots_adjust(top=0.9)
    col_4_pos = fig.get_axes()[4].get_position()
    fig.text(
        x=col_4_pos.x0 + col_4_pos.width / 2,
        y=0.97,
        s="Predicted CDH5 Images",
        ha="center",
        fontsize=FONTSIZE_LARGE,
    )
    _id = (
        f"{example.dataset_name}P{example.position}T{example.timepoint}"
        f"X{example.crop_x_start}Y{example.crop_y_start}"
    )
    save_plot_to_path(fig, output_path, f"denoising_contact_sheet_{_id}")
    plt.close(fig)


def _save_intermediate_contact_sheet(
    conditioning_input_crop,
    ground_truth,
    images_to_denoise,
    denoised_images,
    metrics,
    label_for_conditioning,
    noise_labels,
    output_path,
    example,
):
    """Create and save per-example intermediate contact sheet with metrics."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_contact_sheet_with_metrics_column,
    )
    from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_FIG_KWARGS,
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_SUBPLOT_KWARGS,
    )

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
    _id = (
        f"{example.dataset_name}P{example.position}T{example.timepoint}"
        f"X{example.crop_x_start}Y{example.crop_y_start}"
    )
    save_plot_to_path(fig, output_path, f"denoising_contact_sheet_{_id}")
    plt.close(fig)


def _save_summary_figure(
    example_results_100,
    example_metrics_100,
    num_examples,
    label_for_conditioning,
    example_set_label,
    model_idx,
    compute_metrics,
    output_path,
    model_label: str | None = None,
):
    """Create and save the per-example-set summary figure."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_contact_sheet_with_metrics_column,
    )
    from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_SUBPLOT_KWARGS,
    )

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
        title_model = model_label or f"Model {model_idx + 1}"
        fig.suptitle(
            f"100% Noise Denoising - {example_set_label} - {title_model}",
            fontsize=FONTSIZE_LARGE,
            y=0.995,
        )
        filename = f"contact_sheet_predict_all_examples_model{model_idx + 1}"
    else:
        fig = make_contact_sheet(
            panels=example_results_100,
            max_rows=num_examples,
            max_cols=num_img_cols,
            col_titles=col_titles,
            row_titles=row_titles,
            direction="left-right first",
            font_size=FONTSIZE_MEDIUM,
            subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
            gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
            fig_kwargs={"figsize": (num_img_cols * 1.5, num_examples * 1.5)},
        )
        filename = "contact_sheet_predict_all_examples"

    save_plot_to_path(fig, output_path, filename)
    plt.close(fig)


# ================================
# Core Model Evaluation Function
# ================================


def evaluate_single_model(
    model_idx: int,
    manifest_name: str,
    run_name_input: str | None,
    random_seed: int,
    example_sets_all: list,
    example_sets_for_metrics: set,
    save_intermediate_plots: bool,
    save_crops_as_tiff: bool,
    include_negative_controls: bool,
    compute_metrics: bool,
    compute_baseline: bool = True,
    is_default_seed: bool = True,
) -> dict:
    r"""Evaluate a single model and return its metrics.

    Supports both basic QC mode (negative controls) and comparison mode
    (quantitative metrics).
    """
    from numpy.random import default_rng

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.io import get_config_dict_from_mlflow, get_output_path, load_model
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import add_noise_to_image
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        MODEL_QC_NOISE_LEVELS,
    )

    # Conditional import for metrics
    lpips_calculator = None
    _compute_denoising_metrics = None
    if compute_metrics:
        from endo_pipeline.library.analyze.image_metrics import (
            LPIPSCalculator,
            compute_denoising_metrics,
        )

        _compute_denoising_metrics = compute_denoising_metrics
        lpips_calculator = LPIPSCalculator()

    # --- Setup ---
    NOISE_LABELS = [f"{level * 100:.0f}% Noise" for level in [*MODEL_QC_NOISE_LEVELS, 1]]
    rng = default_rng(seed=random_seed)

    model_manifest = load_model_manifest(manifest_name)
    run_name_ = (
        get_most_recent_run_name(model_manifest) if run_name_input is None else run_name_input
    )
    model_location = model_manifest.locations[run_name_]
    model_label = _make_model_label(manifest_name, run_name_)

    logger.info("Processing model %d: %s (seed=%d)", model_idx + 1, manifest_name, random_seed)

    if model_location.mlflowid is not None:
        model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    else:
        raise ValueError("mlflowid is None")
    crop_size = model_config.model.image_shape[-1]
    cond_key = model_config.model.condition_key
    diffusion_key = DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
    label_for_conditioning = "Brightfield" if cond_key == "raw_bf" else "CDH5"
    model = load_model(model_location, instantiate=True)

    cdh5_labels = [
        "Original CDH5",
        "Noised CDH5",
        f"{label_for_conditioning}\nembedding",
        "Scrambled\nembedding",
        "Scrambled\ninput image",
    ]

    result = _make_result_skeleton(model_idx, model_label, run_name_, random_seed)

    # --- Process each example set ---
    for example_set, example_set_label in example_sets_all:
        include_in_metrics = example_set_label in example_sets_for_metrics

        if DEMO_MODE:
            logger.info("DEMO MODE: Limiting example set to first example only")
            example_set = example_set[:1]

        example_results_100: list = []
        example_metrics_100: list = []

        output_path = get_output_path(
            "model_qc",
            manifest_name,
            run_name_,
            example_set_label,
            create_directories=False,
        )

        for example in example_set:

            # Reuse the existing helper for image loading & cropping
            conditioning_input_crop, diffusion_input_crop = load_and_preprocess_example_crop(
                example,
                model_config,
                crop_size,
                cond_key,
                diffusion_key,
            )

            latent = get_latent_vector_from_crop(
                model,
                conditioning_input_crop,
                num_gpus=NUM_GPUS,
            )
            noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

            # Full noise levels when we need plots; 100%-only otherwise
            run_all = save_intermediate_plots or include_negative_controls
            if run_all:
                noisy_images = [
                    add_noise_to_image(diffusion_input_crop, noise_image, lvl)
                    for lvl in MODEL_QC_NOISE_LEVELS
                ]
                images_to_denoise = [*noisy_images, noise_image]
            else:
                noisy_images = []
                images_to_denoise = [noise_image]

            denoised_images = _denoise_images(
                model,
                latent,
                images_to_denoise,
                NUM_GPUS,
            )

            # Negative controls
            if include_negative_controls and is_default_seed:
                scrambled_emb = rng.permuted(latent)
                denoised_scrambled_emb = _denoise_images(
                    model,
                    scrambled_emb,
                    images_to_denoise,
                    NUM_GPUS,
                )
                img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
                    conditioning_input_crop.shape
                )
                latent_scrambled_input = get_latent_vector_from_crop(
                    model,
                    img_scrambled,
                    num_gpus=NUM_GPUS,
                )
                denoised_scrambled_input = _denoise_images(
                    model,
                    latent_scrambled_input,
                    images_to_denoise,
                    NUM_GPUS,
                )
                _save_negative_control_sheet(
                    conditioning_input_crop,
                    diffusion_input_crop,
                    images_to_denoise,
                    denoised_images,
                    denoised_scrambled_emb,
                    denoised_scrambled_input,
                    label_for_conditioning,
                    cdh5_labels,
                    NOISE_LABELS,
                    output_path,
                    example,
                )
            else:
                # Consume RNG state for reproducibility
                _ = rng.permuted(latent)
                _ = rng.permuted(conditioning_input_crop.ravel())

            # Save crops
            if save_crops_as_tiff and is_default_seed and compute_metrics:
                save_denoising_crops(
                    output_path=output_path,
                    dataset_name=example.dataset_name,
                    position=example.position,
                    timepoint=example.timepoint,
                    start_x=example.crop_x_start,
                    start_y=example.crop_y_start,
                    conditioning_input_crop=conditioning_input_crop,
                    diffusion_input_crop=diffusion_input_crop,
                    noisy_diffusion_input_images=noisy_images,
                    noise_image=noise_image,
                    denoised_images=denoised_images,
                    noise_levels=list(MODEL_QC_NOISE_LEVELS),
                )

            ground_truth = diffusion_input_crop.squeeze()

            # Baseline metrics (next-timepoint comparison)
            if compute_baseline and compute_metrics and include_in_metrics:
                baseline = _compute_baseline_for_example(
                    example,
                    model_config,
                    diffusion_key,
                    crop_size,
                    ground_truth,
                    lpips_calculator,
                )
                if baseline is not None:
                    corr, ssim_val, lpips_val = baseline
                    result[example_set_label]["baseline_correlations"].append(corr)
                    result[example_set_label]["baseline_ssims"].append(ssim_val)
                    result[example_set_label]["baseline_lpips"].append(lpips_val)

            # Model metrics
            metrics: list[dict] | None = None
            if compute_metrics and include_in_metrics:
                if _compute_denoising_metrics is not None:
                    metrics, metrics_100 = _compute_denoising_metrics(
                        ground_truth=ground_truth,
                        denoised_images=denoised_images,
                        lpips_calculator=lpips_calculator,
                        compute_all_noise_levels=save_intermediate_plots and is_default_seed,
                    )
                    result[example_set_label]["correlations_100"].append(metrics_100["correlation"])
                    result[example_set_label]["ssims_100"].append(metrics_100["ssim"])
                    result[example_set_label]["lpips_100"].append(metrics_100["lpips"])
                    example_metrics_100.append(metrics_100)

                if save_intermediate_plots and is_default_seed:
                    _save_intermediate_contact_sheet(
                        conditioning_input_crop,
                        ground_truth,
                        images_to_denoise,
                        denoised_images,
                        metrics,
                        label_for_conditioning,
                        NOISE_LABELS,
                        output_path,
                        example,
                    )

            example_results_100.extend(
                [
                    conditioning_input_crop.squeeze(),
                    ground_truth,
                    denoised_images[-1].squeeze(),
                ]
            )

        # Summary figure (default seed only)
        if (include_negative_controls or save_intermediate_plots) and is_default_seed:
            # Only use metrics layout if we actually computed metrics for this set
            has_metrics = compute_metrics and include_in_metrics and len(example_metrics_100) > 0
            _save_summary_figure(
                example_results_100,
                example_metrics_100,
                len(example_set),
                label_for_conditioning,
                example_set_label,
                model_idx,
                has_metrics,
                output_path,
                model_label=model_label,
            )

    return result


# ============================
# Metrics Aggregation Helpers
# ============================


def aggregate_seed_metrics(
    all_seed_results: dict[int, dict[int, dict]],
    model_manifest_names: list[str],
    example_sets_for_metrics: set[str],
    seeds_to_evaluate: list[int],
) -> tuple[dict[str, list[dict]], list[str]]:
    r"""Aggregate per-seed metrics into combined metric dictionaries per model.

    Parameters
    ----------
    all_seed_results
        Nested dict: ``{model_idx: {seed: result_dict}}``.
    model_manifest_names
        List of manifest names (one per model).
    example_sets_for_metrics
        Set of example-set labels to aggregate (e.g. ``{"validation_positions", "rep_2_positions"}``).
    seeds_to_evaluate
        The list of seeds that were evaluated.

    Returns
    -------
    all_metrics
        ``{example_set_label: [aggregated_dict_per_model, ...]}``.
    model_labels
        Ordered list of human-readable model labels extracted from results.
    """
    all_metrics: dict[str, list[dict]] = {label: [] for label in example_sets_for_metrics}
    model_labels: list[str] = []

    for model_idx in range(len(model_manifest_names)):
        seed_results = all_seed_results[model_idx]
        first_result = next(iter(seed_results.values()))
        model_labels.append(first_result["model_label"])

        for example_set_label in example_sets_for_metrics:
            all_corrs: list[float] = []
            all_ssims: list[float] = []
            all_lpips: list[float] = []
            all_baseline_corrs: list[float] = []
            all_baseline_ssims: list[float] = []
            all_baseline_lpips: list[float] = []

            for _seed, result in seed_results.items():
                all_corrs.extend(result[example_set_label]["correlations_100"])
                all_ssims.extend(result[example_set_label]["ssims_100"])
                all_lpips.extend(result[example_set_label]["lpips_100"])
                if "baseline_correlations" in result[example_set_label]:
                    all_baseline_corrs.extend(result[example_set_label]["baseline_correlations"])
                    all_baseline_ssims.extend(result[example_set_label]["baseline_ssims"])
                    all_baseline_lpips.extend(result[example_set_label]["baseline_lpips"])

            aggregated_metrics = {
                "model_idx": model_idx,
                "model_label": first_result["model_label"],
                "example_set": example_set_label,
                "correlations_100": all_corrs,
                "ssims_100": all_ssims,
                "lpips_100": all_lpips,
                "baseline_correlations": all_baseline_corrs,
                "baseline_ssims": all_baseline_ssims,
                "baseline_lpips": all_baseline_lpips,
                "num_seeds": len(seeds_to_evaluate),
            }
            all_metrics[example_set_label].append(aggregated_metrics)

    return all_metrics, model_labels


def compute_baseline_data(
    all_metrics: dict[str, list[dict]],
    compute_baseline: bool,
) -> dict[str, dict[str, float]]:
    r"""Extract baseline (temporal next-timepoint) statistics from aggregated metrics.

    Parameters
    ----------
    all_metrics
        Output of :func:`aggregate_seed_metrics`.
    compute_baseline
        Whether baseline metrics were computed during evaluation.

    Returns
    -------
    baseline_data
        ``{"validation": {metric_stat: value, ...}, "rep2": {...}}``.
    """
    _zero: dict[str, float] = {
        "corr_mean": 0.0,
        "corr_std": 0.0,
        "ssim_mean": 0.0,
        "ssim_std": 0.0,
        "lpips_mean": 0.0,
        "lpips_std": 0.0,
    }
    baseline_data: dict[str, dict[str, float]] = {
        "validation": dict(_zero),
        "rep2": dict(_zero),
    }
    if not compute_baseline:
        return baseline_data

    for split_key, metric_key in [
        ("validation", "validation_positions"),
        ("rep2", "rep_2_positions"),
    ]:
        for data in all_metrics[metric_key]:
            if data["baseline_correlations"]:
                baseline_data[split_key] = {
                    "corr_mean": float(np.mean(data["baseline_correlations"])),
                    "corr_std": float(np.std(data["baseline_correlations"])),
                    "ssim_mean": float(np.mean(data["baseline_ssims"])),
                    "ssim_std": float(np.std(data["baseline_ssims"])),
                    "lpips_mean": float(np.mean(data["baseline_lpips"])),
                    "lpips_std": float(np.std(data["baseline_lpips"])),
                }
                break  # Baseline is same for all models

    return baseline_data


def build_models_data(
    all_metrics: dict[str, list[dict]],
    model_manifest_names: list[str],
    baseline_data: dict[str, dict[str, float]],
    compute_baseline: bool,
) -> list[dict[str, Any]]:
    r"""Build the per-model data dicts consumed by comparison bar-plot helpers.

    Parameters
    ----------
    all_metrics
        Output of :func:`aggregate_seed_metrics`.
    model_manifest_names
        List of manifest names (one per model).
    baseline_data
        Output of :func:`compute_baseline_data`.
    compute_baseline
        Whether baseline metrics were computed during evaluation.

    Returns
    -------
    models_data
        List of dicts, one per model, with validation/rep2 summary statistics.
    """
    _zero: dict[str, float] = {
        "corr_mean": 0.0,
        "corr_std": 0.0,
        "ssim_mean": 0.0,
        "ssim_std": 0.0,
        "lpips_mean": 0.0,
        "lpips_std": 0.0,
    }
    models_data: list[dict[str, Any]] = []

    for model_idx in range(len(model_manifest_names)):
        model_entry: dict[str, Any] = {
            "model_idx": model_idx,
            "model_label": None,
            "validation": dict(_zero),
            "rep2": dict(_zero),
            "baseline_validation": baseline_data["validation"] if compute_baseline else None,
            "baseline_rep2": baseline_data["rep2"] if compute_baseline else None,
        }

        for split_key, metric_key in [
            ("validation", "validation_positions"),
            ("rep2", "rep_2_positions"),
        ]:
            for data in all_metrics[metric_key]:
                if data["model_idx"] == model_idx:
                    if model_entry["model_label"] is None:
                        model_entry["model_label"] = data["model_label"]
                    model_entry[split_key] = {
                        "corr_mean": float(np.mean(data["correlations_100"])),
                        "corr_std": float(np.std(data["correlations_100"])),
                        "ssim_mean": float(np.mean(data["ssims_100"])),
                        "ssim_std": float(np.std(data["ssims_100"])),
                        "lpips_mean": float(np.mean(data["lpips_100"])),
                        "lpips_std": float(np.std(data["lpips_100"])),
                    }
                    break

        models_data.append(model_entry)

    return models_data


def create_comparison_plots_and_summary(
    models_data: list[dict[str, Any]],
    model_manifest_names: list[str],
    run_names: list[str | None],
    seeds_to_evaluate: list[int],
    baseline_data: dict[str, dict[str, float]],
    compute_baseline: bool,
) -> None:
    r"""Create comparison bar plots and log the summary table.

    Parameters
    ----------
    models_data
        Output of :func:`build_models_data`.
    model_manifest_names
        List of manifest names (one per model).
    run_names
        List of run names (one per model; ``None`` entries become ``"latest"``).
    seeds_to_evaluate
        The list of seeds that were evaluated.
    baseline_data
        Output of :func:`compute_baseline_data`.
    compute_baseline
        Whether baseline metrics were computed during evaluation.
    """
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.model_qc_plots import create_comparison_bar_plot
    from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_QC_LABELS

    seed_suffix = f"_seeds_{len(seeds_to_evaluate)}" if len(seeds_to_evaluate) > 1 else ""
    comparison_output_path = get_output_path(
        "model_qc",
        "comparison",
        f"models_{len(model_manifest_names)}{seed_suffix}",
    )

    # Determine model labels
    if len(models_data) == len(DEFAULT_MODEL_QC_LABELS):
        model_labels = DEFAULT_MODEL_QC_LABELS
    else:
        model_labels = [f"Model {i + 1}" for i in range(len(models_data))]

    seeds_info = (
        f" (averaged over {len(seeds_to_evaluate)} seeds)" if len(seeds_to_evaluate) > 1 else ""
    )
    legend_text = f"Model Details{seeds_info}:\n" + "\n".join(
        f"{model_labels[i]}: "
        f"{model_manifest_names[i]}\n         Run: {run_names[i] if run_names[i] else 'latest'}"
        for i in range(len(models_data))
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
            legend_text=legend_text,
            model_labels=model_labels,
            show_baseline=compute_baseline,
            **extra_kw,
        )

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY: Model Performance{seeds_info}")
    logger.info("=" * 80)

    if compute_baseline and baseline_data["validation"]["corr_mean"] > 0:
        logger.info("\nBASELINE (Temporal - Next Timepoint Comparison):")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            b = baseline_data[split_key]
            logger.info(
                f"  {split_label} - Corr: {b['corr_mean']:.3f} ± {b['corr_std']:.3f}, "
                f"SSIM: {b['ssim_mean']:.3f} ± {b['ssim_std']:.3f}, "
                f"LPIPS: {b['lpips_mean']:.3f} ± {b['lpips_std']:.3f}"
            )
        logger.info("-" * 80)

    for model_data in models_data:
        logger.info(f"\n{model_data['model_label']}:")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            d = model_data[split_key]
            logger.info(
                f"  {split_label} - Corr: {d['corr_mean']:.3f} ± {d['corr_std']:.3f}, "
                f"SSIM: {d['ssim_mean']:.3f} ± {d['ssim_std']:.3f}, "
                f"LPIPS: {d['lpips_mean']:.3f} ± {d['lpips_std']:.3f}"
            )
    logger.info("=" * 80)
