from pathlib import Path
import logging
from typing import Any

import numpy as np
from numpy.random import Generator
import tifffile

logger = logging.getLogger(__name__)


def load_and_preprocess_example_crop(
    example: Any,
    model_config: Any,
    crop_size: int,
    channel_key_for_conditioning_input: str,
    diffusion_input_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image for a given example and preprocess it for model QC.

    This function loads the image from the zarr location, applies the
    transforms from the model config, and crops both the conditioning
    and diffusion input images.

    Parameters
    ----------
    example
        Example object containing dataset_name, position, timepoint, crop coords.
    model_config
        Model configuration containing image processing transforms.
    crop_size
        Size of the square crop to extract.
    channel_key_for_conditioning_input
        Key for the conditioning input channel (e.g., "raw_bf" or "raw_cdh5").
    diffusion_input_key
        Key for the diffusion input channel (e.g., "raw_cdh5").

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (conditioning_input_crop, diffusion_input_crop).
    """
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

    dataset_config = load_dataset_config(example.dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
    img = load_image(
        zarr_loc,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=example.timepoint,
        squeeze=True,
        compute=True,
    )

    # Apply transforms from model config
    data = create_data_dict_loaded_image(img)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    # Extract the processed conditioning and diffusion images
    transformed_conditioning_input_image = get_target_image_from_sample(
        sample, target_key=channel_key_for_conditioning_input
    )
    transformed_diffusion_input_image = get_target_image_from_sample(
        sample, target_key=diffusion_input_key
    )

    # Crop both images to the same region
    conditioning_input_crop = crop_image(
        transformed_conditioning_input_image,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )
    diffusion_input_crop = crop_image(
        transformed_diffusion_input_image,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    )

    return conditioning_input_crop, diffusion_input_crop


def run_denoising_experiments(
    model: Any,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    rng: Generator,
    noise_levels: list[float],
    num_gpus: int,
) -> dict[str, list[np.ndarray]]:
    """
    Run denoising experiments with normal and negative control conditioning.

    Performs three denoising experiments:
    1. Normal: Using the latent vector from the conditioning input crop.
    2. Scrambled embedding: Using a randomly shuffled latent vector.
    3. Scrambled input: Using latent vector from a scrambled input image.

    Parameters
    ----------
    model
        The loaded DiffAE model.
    conditioning_input_crop
        The conditioning input image crop.
    diffusion_input_crop
        The diffusion input image crop (ground truth CDH5).
    rng
        NumPy random number generator.
    noise_levels
        List of noise levels to test (e.g., [0.25, 0.5, 0.75]).
    num_gpus
        Number of GPUs to use for inference.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Dictionary containing:
        - "images_to_denoise": List of noised images (including pure noise).
        - "noise_image": The pure noise image.
        - "denoised_normal": Denoised images using normal conditioning.
        - "denoised_scrambled_embedding": Denoised with scrambled latent vector.
        - "denoised_scrambled_input": Denoised with latent from scrambled image.
        - "conditioning_latent": The conditioning latent vector.
    """
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        add_noise_to_image,
        generate_from_coords_and_noised_image,
    )

    # Get latent vector embedding of the conditioning crop
    conditioning_crop_latent_vector = get_latent_vector_from_crop(
        model, conditioning_input_crop, num_gpus=num_gpus
    )

    # Sample random noise image
    noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

    # Add noise to diffusion input with increasing weight
    noisy_diffusion_input_images = [
        add_noise_to_image(diffusion_input_crop, noise_image, noise_level)
        for noise_level in noise_levels
    ]

    # Images to denoise: noised images + pure noise
    images_to_denoise = [*noisy_diffusion_input_images, noise_image]

    # Experiment 1: Normal denoising with proper conditioning
    denoised_images_normal = [
        generate_from_coords_and_noised_image(
            model, conditioning_crop_latent_vector, noised_image, num_gpus=num_gpus
        )
        for noised_image in images_to_denoise
    ]

    # Experiment 2: Negative control - scrambled latent vector
    latent_vector_scrambled = rng.permuted(conditioning_crop_latent_vector)
    denoised_images_scrambled_embedding = [
        generate_from_coords_and_noised_image(
            model, latent_vector_scrambled, noised_image, num_gpus=num_gpus
        )
        for noised_image in images_to_denoise
    ]

    # Experiment 3: Negative control - latent from scrambled input image
    img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
        conditioning_input_crop.shape
    )
    latent_vector_from_img_scrambled = get_latent_vector_from_crop(
        model, img_scrambled, num_gpus=num_gpus
    )
    denoised_images_scrambled_input = [
        generate_from_coords_and_noised_image(
            model, latent_vector_from_img_scrambled, noised_image, num_gpus=num_gpus
        )
        for noised_image in images_to_denoise
    ]

    return {
        "images_to_denoise": images_to_denoise,
        "noise_image": noise_image,
        "denoised_normal": denoised_images_normal,
        "denoised_scrambled_embedding": denoised_images_scrambled_embedding,
        "denoised_scrambled_input": denoised_images_scrambled_input,
        "conditioning_latent": conditioning_crop_latent_vector,
    }


def save_image_as_tiff(
    image,
    output_path: Path,
    filename: str,
) -> None:
    """
    Save an image array as a TIFF file.

    Parameters
    ----------
    image
        The image array to save. Can be 2D or 3D (with channel dimension).
    output_path
        Path to directory where TIFF should be saved.
    filename
        Filename for the saved TIFF (without extension).
    """
    import numpy as np
    import tifffile

    output_file = output_path / f"{filename}.tiff"
    # Ensure image is in the right format for tifffile
    image_to_save = image.squeeze()
    # Convert to float32 if not already a float type to preserve precision
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
    """
    Save all denoising crops as TIFF files to a structured directory.

    Parameters
    ----------
    output_path
        Base output path for saving files.
    dataset_name
        Name of the dataset.
    position
        Position index.
    timepoint
        Timepoint index.
    start_x
        X coordinate of crop start.
    start_y
        Y coordinate of crop start.
    conditioning_input_crop
        The conditioning input image crop.
    diffusion_input_crop
        The ground truth diffusion input crop.
    noisy_diffusion_input_images
        List of noised images at various noise levels.
    noise_image
        Pure noise image.
    denoised_images
        List of denoised output images.
    noise_levels
        List of noise level values (e.g., [0.25, 0.5, 0.75]).

    Returns
    -------
    Path
        Path to the directory where crops were saved.
    """
    crops_output_path = (
        output_path / "crops" / f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}"
    )
    crops_output_path.mkdir(parents=True, exist_ok=True)

    # Save conditioning input crop
    save_image_as_tiff(conditioning_input_crop, crops_output_path, "conditioning_input")

    # Save ground truth (diffusion input crop)
    save_image_as_tiff(diffusion_input_crop, crops_output_path, "ground_truth")

    # Save noised images at each noise level
    for noised_img, noise_level in zip(noisy_diffusion_input_images, noise_levels, strict=False):
        noise_pct = int(noise_level * 100)
        save_image_as_tiff(noised_img, crops_output_path, f"noised_{noise_pct:03d}pct")

    # Save pure noise image
    save_image_as_tiff(noise_image, crops_output_path, "noised_100pct")

    # Save denoised images at each noise level
    for denoise_idx, denoised_img in enumerate(denoised_images):
        if denoise_idx < len(noise_levels):
            noise_pct = int(noise_levels[denoise_idx] * 100)
        else:
            noise_pct = 100
        save_image_as_tiff(
            denoised_img, crops_output_path, f"denoised_from_{noise_pct:03d}pct_noise"
        )

    logger.debug("Saved crops to %s", crops_output_path)
    return crops_output_path


# =============================================================================
# Core Model Evaluation Function
# =============================================================================

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
    """
    Evaluate a single model and return its metrics.

    This function is designed to be called in parallel for different models.
    It supports both basic QC mode (with negative controls) and comparison mode
    (with quantitative metrics).

    Parameters
    ----------
    model_idx
        Index of the model being processed.
    manifest_name
        Name of the model manifest.
    run_name_input
        Run name within the manifest, or None for most recent.
    random_seed
        Random seed for noise generation.
    example_sets_all
        List of (example_set, label) tuples to process.
    example_sets_for_metrics
        Set of example set labels to include in metrics.
    save_intermediate_plots
        Whether to save intermediate contact sheets.
    save_crops_as_tiff
        Whether to save crops as TIFF files.
    include_negative_controls
        Whether to include negative control experiments (scrambled embedding,
        scrambled input image).
    compute_metrics
        Whether to compute quantitative metrics (correlation, SSIM, LPIPS).
    compute_baseline
        Whether to compute baseline metrics by comparing ground truth CDH5
        to CDH5 from the next timepoint (timepoint + 1). This provides a
        temporal baseline for evaluating model performance.
    is_default_seed
        If True, save crops and intermediate plots. If False, only compute metrics.

    Returns
    -------
    dict
        Dictionary containing model metrics and metadata.
    """
    import logging

    import matplotlib.pyplot as plt
    from numpy.random import default_rng

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        get_config_dict_from_mlflow,
        get_output_path,
        load_image,
        load_model,
    )
    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        add_noise_to_image,
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_contact_sheet_with_metrics_column,
    )
    from endo_pipeline.manifests import (
        get_most_recent_run_name,
        get_zarr_location_for_position,
        load_model_manifest,
    )
    from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_FIG_KWARGS,
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_PLOT_DIRECTION,
        MODEL_QC_SUBPLOT_KWARGS,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        MODEL_QC_NOISE_LEVELS,
    )

    logger = logging.getLogger(__name__)

    # Conditional imports for metrics
    if compute_metrics:
        from endo_pipeline.library.analyze.image_metrics import (
            LPIPSCalculator,
            compute_denoising_metrics,
        )

        lpips_calculator = LPIPSCalculator()

    # Set up noise labels
    NOISE_LABELS = [f"{level * 100:.0f}% Noise" for level in [*MODEL_QC_NOISE_LEVELS, 1]]
    NUM_IMAGES_DENOISED = len(NOISE_LABELS)

    # Initialize RNG
    rng = default_rng(seed=random_seed)

    # Load model manifest and get location
    model_manifest = load_model_manifest(manifest_name)
    run_name_ = (
        get_most_recent_run_name(model_manifest) if run_name_input is None else run_name_input
    )
    model_location = model_manifest.locations[run_name_]

    # Create model label
    model_label = (
        f"{manifest_name[:30]}..._{run_name_[:10]}"
        if len(manifest_name) > 30
        else f"{manifest_name}_{run_name_}"
    )

    logger.info(f"Processing model {model_idx + 1}: {manifest_name} (seed={random_seed})")

    # Get model config and load model
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    crop_size = model_config.model.image_shape[-1]
    channel_key_for_conditioning_input = model_config.model.condition_key
    label_for_conditioning = (
        "Brightfield" if channel_key_for_conditioning_input == "raw_bf" else "CDH5"
    )
    model = load_model(model_location, instantiate=True)

    # Set up CDH5 labels for negative control plots
    CDH5_LABELS = [
        "Original CDH5",
        "Noised CDH5",
        f"{label_for_conditioning}\nembedding",
        "Scrambled\nembedding",
        "Scrambled\ninput image",
    ]

    # Storage for metrics
    result = {
        "model_idx": model_idx,
        "model_label": model_label,
        "run_name": run_name_,
        "random_seed": random_seed,
        "validation_positions": {
            "correlations_100": [],
            "ssims_100": [],
            "lpips_100": [],
            "baseline_correlations": [],
            "baseline_ssims": [],
            "baseline_lpips": [],
        },
        "rep_2_positions": {
            "correlations_100": [],
            "ssims_100": [],
            "lpips_100": [],
            "baseline_correlations": [],
            "baseline_ssims": [],
            "baseline_lpips": [],
        },
        "training_positions": {
            "correlations_100": [],
            "ssims_100": [],
            "lpips_100": [],
            "baseline_correlations": [],
            "baseline_ssims": [],
            "baseline_lpips": [],
        },
    }

    # Process each example set
    for example_set, example_set_label in example_sets_all:
        include_in_metrics = example_set_label in example_sets_for_metrics

        if DEMO_MODE:
            logger.info("DEMO MODE: Limiting example set to first example only")
            example_set = example_set[:1]

        example_results_100 = []
        example_metrics_100 = []

        for example in example_set:
            dataset_name = example.dataset_name
            position = example.position
            timepoint = example.timepoint
            start_x = example.crop_x_start
            start_y = example.crop_y_start

            # Get output path
            output_path = get_output_path(
                "model_qc",
                manifest_name,
                run_name_,
                example_set_label,
            )

            dataset_config = load_dataset_config(dataset_name)
            zarr_loc = get_zarr_location_for_position(dataset_config, position)
            img = load_image(
                zarr_loc,
                level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                timepoints=timepoint,
                squeeze=True,
                compute=True,
            )

            # Get image processing steps and apply transforms
            data = create_data_dict_loaded_image(img)
            transforms = get_image_transforms(model_config)
            sample = apply_img_transforms(transforms, data)

            transformed_conditioning_input_image = get_target_image_from_sample(
                sample, target_key=channel_key_for_conditioning_input
            )
            transformed_diffusion_input_image = get_target_image_from_sample(
                sample, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
            )

            conditioning_input_crop = crop_image(
                transformed_conditioning_input_image, start_x, start_y, crop_size
            )
            diffusion_input_crop = crop_image(
                transformed_diffusion_input_image, start_x, start_y, crop_size
            )

            conditioning_crop_latent_vector = get_latent_vector_from_crop(
                model, conditioning_input_crop, num_gpus=NUM_GPUS
            )

            noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

            noisy_diffusion_input_images = [
                add_noise_to_image(diffusion_input_crop, noise_image, noise_level)
                for noise_level in MODEL_QC_NOISE_LEVELS
            ]

            images_to_denoise = [*noisy_diffusion_input_images, noise_image]
            denoised_images = [
                generate_from_coords_and_noised_image(
                    model, conditioning_crop_latent_vector, noised_image, num_gpus=NUM_GPUS
                )
                for noised_image in images_to_denoise
            ]

            # Negative control experiments (only when requested and default seed)
            if include_negative_controls and is_default_seed:
                # Scrambled latent vector
                latent_vector_scrambled = rng.permuted(conditioning_crop_latent_vector)
                denoised_images_by_random_cond = [
                    generate_from_coords_and_noised_image(
                        model, latent_vector_scrambled, noised_image, num_gpus=NUM_GPUS
                    )
                    for noised_image in images_to_denoise
                ]

                # Scrambled input image -> latent vector
                img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
                    conditioning_input_crop.shape
                )
                latent_vector_from_img_scrambled = get_latent_vector_from_crop(
                    model, img_scrambled, num_gpus=NUM_GPUS
                )
                denoised_images_by_random_cond_latent_scramble = [
                    generate_from_coords_and_noised_image(
                        model, latent_vector_from_img_scrambled, noised_image, num_gpus=NUM_GPUS
                    )
                    for noised_image in images_to_denoise
                ]

                # Create negative control contact sheet
                panels = [
                    *[conditioning_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
                    *[diffusion_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
                    *[img.squeeze() for img in images_to_denoise],
                    *[img.squeeze() for img in denoised_images],
                    *[img.squeeze() for img in denoised_images_by_random_cond],
                    *[img.squeeze() for img in denoised_images_by_random_cond_latent_scramble],
                ]
                fig = make_contact_sheet(
                    panels=panels,
                    max_rows=NUM_IMAGES_DENOISED,
                    max_cols=6,
                    col_titles=[f"{label_for_conditioning} input", *CDH5_LABELS],
                    row_titles=NOISE_LABELS,
                    direction=MODEL_QC_PLOT_DIRECTION,
                    font_size=FONTSIZE_MEDIUM,
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
                    fontsize=FONTSIZE_LARGE,
                )
                save_plot_to_path(
                    fig,
                    output_path,
                    f"denoising_negative_controls_{dataset_name}P{position}T{timepoint}X{start_x}Y{start_y}",
                )
                plt.close(fig)
            else:
                # Consume RNG state to keep it consistent across modes
                _ = rng.permuted(conditioning_crop_latent_vector)
                _ = rng.permuted(conditioning_input_crop.ravel())

            # Save crops only for default seed
            if save_crops_as_tiff and is_default_seed and compute_metrics:
                save_denoising_crops(
                    output_path=output_path,
                    dataset_name=dataset_name,
                    position=position,
                    timepoint=timepoint,
                    start_x=start_x,
                    start_y=start_y,
                    conditioning_input_crop=conditioning_input_crop,
                    diffusion_input_crop=diffusion_input_crop,
                    noisy_diffusion_input_images=noisy_diffusion_input_images,
                    noise_image=noise_image,
                    denoised_images=denoised_images,
                    noise_levels=MODEL_QC_NOISE_LEVELS,
                )

            ground_truth = diffusion_input_crop.squeeze()

            # Compute baseline metrics (comparing to next timepoint)
            if compute_baseline and compute_metrics and include_in_metrics:
                try:
                    # Load image at timepoint + 1
                    img_next_timepoint = load_image(
                        zarr_loc,
                        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                        timepoints=timepoint + 1,
                        squeeze=True,
                        compute=True,
                    )
                    
                    # Apply same transforms
                    data_next = create_data_dict_loaded_image(img_next_timepoint)
                    sample_next = apply_img_transforms(transforms, data_next)
                    
                    # Get CDH5 from next timepoint
                    transformed_diffusion_next = get_target_image_from_sample(
                        sample_next, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
                    )
                    
                    # Crop at same location
                    diffusion_crop_next = crop_image(
                        transformed_diffusion_next, start_x, start_y, crop_size
                    )
                    next_timepoint_cdh5 = diffusion_crop_next.squeeze()
                    
                    # Compute baseline metrics: ground_truth vs next_timepoint
                    from scipy.stats import pearsonr
                    from skimage.metrics import structural_similarity as ssim
                    
                    baseline_corr = pearsonr(
                        ground_truth.ravel(), next_timepoint_cdh5.ravel()
                    )[0]
                    baseline_ssim = ssim(
                        ground_truth, next_timepoint_cdh5, data_range=ground_truth.max() - ground_truth.min()
                    )
                    baseline_lpips = lpips_calculator.compute(ground_truth, next_timepoint_cdh5)
                    
                    result[example_set_label]["baseline_correlations"].append(baseline_corr)
                    result[example_set_label]["baseline_ssims"].append(baseline_ssim)
                    result[example_set_label]["baseline_lpips"].append(baseline_lpips)
                    
                except Exception as e:
                    logger.warning(
                        f"Could not compute baseline for {dataset_name} P{position} "
                        f"T{timepoint} -> T{timepoint + 1}: {e}"
                    )

            # Compute metrics if requested
            if compute_metrics and include_in_metrics:
                metrics, metrics_100 = compute_denoising_metrics(
                    ground_truth=ground_truth,
                    denoised_images=denoised_images,
                    lpips_calculator=lpips_calculator,
                    compute_all_noise_levels=save_intermediate_plots and is_default_seed,
                )

                # Store metrics
                result[example_set_label]["correlations_100"].append(metrics_100["correlation"])
                result[example_set_label]["ssims_100"].append(metrics_100["ssim"])
                result[example_set_label]["lpips_100"].append(metrics_100["lpips"])

                example_metrics_100.append(metrics_100)

                # Save intermediate plots with metrics column only for default seed
                if save_intermediate_plots and is_default_seed:
                    contact_panels = [
                        *[conditioning_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
                        *[ground_truth] * NUM_IMAGES_DENOISED,
                        *[img.squeeze() for img in images_to_denoise],
                        *[img.squeeze() for img in denoised_images],
                    ]

                    contact_col_titles = [
                        f"{label_for_conditioning} input",
                        "Original CDH5",
                        "Noised CDH5",
                        "Predicted CDH5",
                    ]

                    fig = create_contact_sheet_with_metrics_column(
                        panels=contact_panels,
                        metrics=metrics,
                        num_rows=NUM_IMAGES_DENOISED,
                        num_img_cols=4,
                        col_titles=contact_col_titles,
                        row_titles=NOISE_LABELS,
                        fontsize_medium=FONTSIZE_MEDIUM,
                        fontsize_small=FONTSIZE_SMALL,
                        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
                        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
                        fig_kwargs=MODEL_QC_FIG_KWARGS,
                        direction="top-down first",
                        show_row_header_column=True,
                    )

                    save_plot_to_path(
                        fig,
                        output_path,
                        f"denoising_contact_sheet_{dataset_name}P{position}T{timepoint}X{start_x}Y{start_y}",
                    )
                    plt.close(fig)

            example_results_100.append(conditioning_input_crop.squeeze())
            example_results_100.append(ground_truth)
            example_results_100.append(denoised_images[-1].squeeze())

        # Summary figure only for default seed
        if save_intermediate_plots and is_default_seed:
            num_img_cols = 3
            num_rows = len(example_set)

            if compute_metrics:
                fig = create_contact_sheet_with_metrics_column(
                    panels=example_results_100,
                    metrics=example_metrics_100,
                    num_rows=num_rows,
                    num_img_cols=num_img_cols,
                    col_titles=[
                        f"{label_for_conditioning} input",
                        "Original CDH5",
                        "Predicted CDH5",
                    ],
                    row_titles=[f"Example {i+1}" for i in range(num_rows)],
                    fontsize_medium=FONTSIZE_MEDIUM,
                    fontsize_small=FONTSIZE_SMALL,
                    subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
                    gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
                    fig_kwargs={"figsize": ((num_img_cols + 1) * 2, num_rows * 1.8)},
                    direction="left-right first",
                )

                fig.suptitle(
                    f"100% Noise Denoising - {example_set_label} - Model {model_idx + 1}",
                    fontsize=FONTSIZE_LARGE,
                    y=0.995,
                )
            else:
                # Simple contact sheet without metrics column
                fig = make_contact_sheet(
                    panels=example_results_100,
                    max_rows=num_rows,
                    max_cols=num_img_cols,
                    col_titles=[
                        f"{label_for_conditioning} input",
                        "Original CDH5",
                        "Predicted CDH5",
                    ],
                    row_titles=[f"Example {i+1}" for i in range(num_rows)],
                    direction="left-right first",
                    font_size=FONTSIZE_MEDIUM,
                    subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
                    gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
                    fig_kwargs={"figsize": (num_img_cols * 1.5, num_rows * 1.5)},
                )

            save_plot_to_path(
                fig,
                output_path,
                f"contact_sheet_predict_all_examples_model{model_idx + 1}",
            )
            plt.close(fig)

    return result
