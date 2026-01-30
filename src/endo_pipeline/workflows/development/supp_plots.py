from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.cli import tags
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.workflow_defaults import (
    DATASET_COLORS,
    METRIC_TEXT_BOX_PROPS,
    RANDOM_SEED,
)

TAGS = ["diffae", tags.TEST_READY, tags.GPU]


def _save_crop_as_tiff(
    image: np.ndarray,
    output_path: Path,
    filename: str,
) -> None:
    """
    Save a crop as a TIFF file.

    Parameters
    ----------
    image
        The image array to save. Can be 2D or 3D (with channel dimension).
    output_path
        Path to directory where TIFF should be saved.
    filename
        Filename for the saved TIFF (without extension).
    """
    import tifffile

    output_file = output_path / f"{filename}.tiff"
    # Ensure image is in the right format for tifffile
    image_to_save = image.squeeze()
    # Convert to float32 if not already a float type to preserve precision
    if not np.issubdtype(image_to_save.dtype, np.floating):
        image_to_save = image_to_save.astype(np.float32)
    tifffile.imwrite(output_file, image_to_save)


def _save_all_crops_as_tiff(
    output_path: Path,
    dataset_name: str,
    position: int,
    timepoint: int,
    start_x: int,
    start_y: int,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    noisy_diffusion_input_images: list[np.ndarray],
    noise_image: np.ndarray,
    denoised_images: list[np.ndarray],
    noise_levels: list[float],
    logger,
) -> None:
    """
    Save all crops as TIFF files to a structured directory.

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
    logger
        Logger instance for debug messages.
    """
    crops_output_path = (
        output_path / "crops" / f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}"
    )
    crops_output_path.mkdir(parents=True, exist_ok=True)

    # Save conditioning input crop
    _save_crop_as_tiff(conditioning_input_crop, crops_output_path, "conditioning_input")

    # Save ground truth (diffusion input crop)
    _save_crop_as_tiff(diffusion_input_crop, crops_output_path, "ground_truth")

    # Save noised images at each noise level
    for noised_img, noise_level in zip(noisy_diffusion_input_images, noise_levels, strict=False):
        noise_pct = int(noise_level * 100)
        _save_crop_as_tiff(noised_img, crops_output_path, f"noised_{noise_pct:03d}pct")

    # Save pure noise image
    _save_crop_as_tiff(noise_image, crops_output_path, "noised_100pct")

    # Save denoised images at each noise level
    for denoise_idx, denoised_img in enumerate(denoised_images):
        if denoise_idx < len(noise_levels):
            noise_pct = int(noise_levels[denoise_idx] * 100)
        else:
            noise_pct = 100
        _save_crop_as_tiff(
            denoised_img, crops_output_path, f"denoised_from_{noise_pct:03d}pct_noise"
        )

    logger.debug(f"Saved crops to {crops_output_path}")


def _compute_denoising_metrics(
    ground_truth: np.ndarray,
    denoised_images: list[np.ndarray],
    lpips_calculator,
    compute_all_noise_levels: bool,
) -> tuple[list[dict] | None, dict]:
    """
    Compute image quality metrics for denoised images.

    Parameters
    ----------
    ground_truth
        The ground truth image (squeezed).
    denoised_images
        List of denoised output images at various noise levels.
    lpips_calculator
        LPIPS calculator instance.
    compute_all_noise_levels
        If True, compute metrics for all noise levels. If False, only compute
        metrics for the 100% noise level.

    Returns
    -------
    tuple
        A tuple of (metrics_list, metrics_100) where metrics_list is None if
        compute_all_noise_levels is False.
    """
    from endo_pipeline.library.analyze.image_metrics import compute_all_metrics

    if compute_all_noise_levels:
        metrics = []
        for denoised_img in denoised_images:
            denoised_squeezed = denoised_img.squeeze()
            metrics.append(
                compute_all_metrics(ground_truth, denoised_squeezed, lpips_calculator).to_dict()
            )
        metrics_100 = metrics[-1]
    else:
        denoised_100 = denoised_images[-1].squeeze()
        metrics_100 = compute_all_metrics(ground_truth, denoised_100, lpips_calculator).to_dict()
        metrics = None

    return metrics, metrics_100


def _create_comparison_bar_plot(
    models_data: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path,
    filename: str,
    legend_text: str,
    ylim: tuple[float, float] | None = None,
    text_box_loc: str = "upper right",
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
    legend_text
        Text for the model details legend box.
    ylim
        Optional y-axis limits as (min, max).
    text_box_loc
        Location for the model details text box (default: "upper right").
        Options: "upper right", "lower right", "upper left", "lower left".
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.io.output import save_plot_to_path

    model_labels_short = [f"Model {i+1}" for i in range(len(models_data))]
    x_pos = np.arange(len(model_labels_short))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(model_labels_short) * 1.5), 7))
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
        color=DATASET_COLORS["validation_positions"],
        alpha=0.8,
    )
    ax.bar(
        x_pos + bar_width / 2,
        rep2_means,
        bar_width,
        yerr=rep2_stds,
        capsize=5,
        label="Rep2",
        color=DATASET_COLORS["rep_2_positions"],
        alpha=0.8,
    )
    ax.set_xlabel("Model", fontsize=FONTSIZE_MEDIUM)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)
    ax.set_title(title, fontsize=FONTSIZE_LARGE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels_short, fontsize=FONTSIZE_MEDIUM)
    ax.legend(fontsize=FONTSIZE_MEDIUM, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Position text box based on text_box_loc parameter
    text_box_positions = {
        "upper right": (0.98, 0.98, "top", "right"),
        "lower right": (0.98, 0.02, "bottom", "right"),
        "upper left": (0.02, 0.98, "top", "left"),
        "lower left": (0.02, 0.02, "bottom", "left"),
    }
    x_pos_text, y_pos_text, va, ha = text_box_positions.get(
        text_box_loc, (0.98, 0.98, "top", "right")
    )
    ax.text(
        x_pos_text,
        y_pos_text,
        legend_text,
        transform=ax.transAxes,
        fontsize=FONTSIZE_SMALL - 2,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=METRIC_TEXT_BOX_PROPS,
        family="monospace",
    )
    plt.tight_layout()
    save_plot_to_path(fig, output_path, filename)
    plt.close(fig)


def _create_contact_sheet_with_metrics_column(
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
) -> plt.Figure:
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
    matplotlib.figure.Figure
        The created figure
    """
    from matplotlib.gridspec import GridSpec

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
        # All examples plot
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

    # Set up width ratios: narrow for row header, equal for images, narrow for metrics
    width_ratios = []
    if show_row_header_column:
        width_ratios.append(0.15)  # Row header column (much narrower for rotated text)
    width_ratios.extend([1] * num_img_cols)  # Image columns
    width_ratios.append(0.4)  # Metrics column (narrower)

    # Height ratios: smaller title row for all_examples, equal image rows
    if show_row_header_column:
        height_ratios = [0.08] + [1] * num_rows
    else:
        # All examples: smaller title row to reduce gap
        height_ratios = [0.05] + [1] * num_rows

    gs = GridSpec(
        num_rows + 1,  # +1 for column titles
        total_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        **adjusted_gridspec_kwargs,
    )

    # Add row header column title (empty) if showing row headers
    if show_row_header_column:
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")

    # Add column title row
    for col_idx in range(num_img_cols):
        ax = fig.add_subplot(gs[0, col_idx + row_header_offset])
        ax.text(
            0.5,
            0.5,
            col_titles[col_idx],
            ha="center",
            va="center",
            fontsize=fontsize_medium,
            weight="bold",
        )
        ax.axis("off")

    # Add metrics column title
    ax = fig.add_subplot(gs[0, num_img_cols + row_header_offset])
    ax.text(0.5, 0.5, "Metrics", ha="center", va="center", fontsize=fontsize_medium, weight="bold")
    ax.axis("off")

    # Add image panels with correct indexing based on direction
    for row_idx in range(num_rows):
        # Add row header if showing row header column
        if show_row_header_column:
            ax = fig.add_subplot(gs[row_idx + 1, 0])
            ax.text(
                0.5,
                0.5,
                row_titles[row_idx],
                ha="center",
                va="center",
                fontsize=fontsize_small,
                weight="bold",
                rotation=90,  # Rotate label 90 degrees
            )
            ax.axis("off")

        for col_idx in range(num_img_cols):
            if direction == "top-down first":
                panel_idx = col_idx * num_rows + row_idx
            else:
                panel_idx = row_idx * num_img_cols + col_idx

            ax = fig.add_subplot(gs[row_idx + 1, col_idx + row_header_offset])

            # Display image with grayscale colormap
            if panel_idx < len(panels):
                ax.imshow(panels[panel_idx], cmap="gray", aspect="equal")

            # Add row title on first column (only if not using dedicated row header column)
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

        # Add metrics panel for this row
        ax = fig.add_subplot(gs[row_idx + 1, num_img_cols + row_header_offset])
        ax.axis("off")

        # Display metrics as text
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
                family="monospace",
            )

    return fig


def main(
    model_manifest_name: list[str],
    run_name: list[str] | None = None,
    random_seed: int = RANDOM_SEED,
    save_intermediate_plots: bool = False,
    save_crops_as_tiff: bool = False,
) -> None:
    """
    Run quality check assessment for trained models.
    This workflow loads one or more trained DiffAE models and evaluates them on
    validation and rep2 example sets. It generates contact sheets showing denoising
    performance at various noise levels, and creates comparison plots of various
    metrics across models and datasets.

    Parameters
    ----------
    model_manifest_name
        Name(s) of the model manifest to load the model from. Provide once per model.
        For single model: --model_manifest_name model1
        For multiple models: --model_manifest_name model1 --model_manifest_name model2
    run_name
        Run name(s) within the model manifest to load. Provide once per model.
        If not provided, the most recent run is used for each model.
        For single model: --run_name run1
        For multiple models: --run_name run1 --run_name run2
    random_seed
        Random seed for reproducibility of noise generation.
    save_intermediate_plots
        If True, saves individual contact sheets for each example and summary figures
        for each model. If False (default), only generates final comparison plots
        across all models.
    save_crops_as_tiff
        If True, saves each crop (conditioning input, ground truth, noised images,
        and denoised images) as individual TIFF files. Files are saved to a 'crops'
        subdirectory within the output path.
    """
    import logging
    from typing import Any

    import matplotlib.pyplot as plt
    import numpy as np
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
    from endo_pipeline.library.analyze.image_metrics import LPIPSCalculator
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        add_noise_to_image,
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.manifests import (
        get_most_recent_run_name,
        get_zarr_location_for_position,
        load_model_manifest,
    )
    from endo_pipeline.settings.examples import (
        MODEL_QC_EXAMPLES_REP_2_POSITIONS,
        MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
        MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
    )
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_FIG_KWARGS,
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_SUBPLOT_KWARGS,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        MODEL_QC_NOISE_LEVELS,
    )

    logger = logging.getLogger(__name__)
    # Initialize LPIPS calculator (lazy loading - only initialized on first use)
    lpips_calculator = LPIPSCalculator()
    # Normalize inputs to lists
    model_manifest_names = model_manifest_name

    if run_name is None:
        run_names = [None] * len(model_manifest_names)
    else:
        run_names = run_name

    # Ensure we have matching numbers of manifests and run names
    if len(run_names) != len(model_manifest_names):
        raise ValueError(
            f"Number of run_names ({len(run_names)}) must match "
            f"number of model_manifest_names ({len(model_manifest_names)})"
        )
    example_sets_all = [
        (MODEL_QC_EXAMPLES_TRAINING_POSITIONS, "training_positions"),
        (MODEL_QC_EXAMPLES_VALIDATION_POSITIONS, "validation_positions"),
        (MODEL_QC_EXAMPLES_REP_2_POSITIONS, "rep_2_positions"),
    ]
    # Sets to include in metrics comparison
    example_sets_for_metrics = {"validation_positions", "rep_2_positions"}
    if DEMO_MODE:
        logger.info("DEMO MODE: Limiting MODEL_QC_EXAMPLES to training set")
        example_sets_all = [example_sets_all[0]]
        example_sets_for_metrics = {"training_positions"}
    # Set defaults for plot titles
    NOISE_LABELS = [f"{level * 100:.0f}% Noise" for level in [*MODEL_QC_NOISE_LEVELS, 1]]
    NUM_IMAGES_DENOISED = len(NOISE_LABELS)
    # Storage for all metrics across models and datasets
    all_metrics: dict[str, list[dict]] = {"validation_positions": [], "rep_2_positions": []}

    # Model labels for plotting
    model_labels = []
    # Process each model
    for model_idx, (manifest_name, run_name_input) in enumerate(
        zip(model_manifest_names, run_names, strict=False)
    ):
        logger.info(
            f"Processing model {model_idx + 1}/{len(model_manifest_names)}: {manifest_name}"
        )

        # Re-seed RNG for each model - this ensures each model evaluation starts with the same RNG state
        rng = default_rng(seed=random_seed)

        # Load model manifest and get location for run_name
        model_manifest = load_model_manifest(manifest_name)
        run_name_ = (
            get_most_recent_run_name(model_manifest) if run_name_input is None else run_name_input
        )
        model_location = model_manifest.locations[run_name_]

        # Create model label for plots
        model_label = (
            f"{manifest_name[:30]}..._{run_name_[:10]}"
            if len(manifest_name) > 30
            else f"{manifest_name}_{run_name_}"
        )
        model_labels.append(model_label)
        # Model config has info about image processing steps
        model_config = get_config_dict_from_mlflow(model_location.mlflowid)
        crop_size = model_config.model.image_shape[-1]  # assumes square crops
        # Get the condition and diffusion image keys from model config
        channel_key_for_conditioning_input = model_config.model.condition_key
        label_for_conditioning = (
            "Brightfield" if channel_key_for_conditioning_input == "raw_bf" else "CDH5"
        )
        # Load model as instantiated DiffAE object
        model = load_model(model_location, instantiate=True)
        # Process each example set
        for example_set, example_set_label in example_sets_all:
            # Check if this set should be included in metrics comparison
            include_in_metrics = example_set_label in example_sets_for_metrics
            if DEMO_MODE:
                logger.info("DEMO MODE: Limiting example set to first example only")
                example_set = example_set[:1]
            # Store 100% denoised example results for this model and dataset
            example_results_100 = []
            example_metrics_100 = []

            # Store metrics only for 100% noise level
            dataset_metrics = {
                "model_idx": model_idx,
                "model_label": model_label,
                "example_set": example_set_label,
                "correlations_100": [],
                "ssims_100": [],
                "lpips_100": [],
            }
            # Process each dataset
            for example in example_set:
                dataset_name = example.dataset_name
                logger.info(f"  Processing dataset: {dataset_name}")
                # Extract position, timepoint, and crop position
                position = example.position
                timepoint = example.timepoint
                start_x = example.crop_x_start
                start_y = example.crop_y_start
                # Only get output path for sets we'll save plots for (avoids empty folders)
                if include_in_metrics:
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
                # Extract the processed conditioning and diffusion images
                transformed_conditioning_input_image = get_target_image_from_sample(
                    sample, target_key=channel_key_for_conditioning_input
                )
                transformed_diffusion_input_image = get_target_image_from_sample(
                    sample, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
                )
                # Crop both images to the same region
                conditioning_input_crop = crop_image(
                    transformed_conditioning_input_image, start_x, start_y, crop_size
                )
                diffusion_input_crop = crop_image(
                    transformed_diffusion_input_image, start_x, start_y, crop_size
                )
                # Get latent vector embedding of the crop used for conditioning
                conditioning_crop_latent_vector = get_latent_vector_from_crop(
                    model, conditioning_input_crop, num_gpus=NUM_GPUS
                )
                # Sample random noise image with fixed seed
                noise_image = rng.standard_normal(size=diffusion_input_crop.shape)
                # Advance RNG state to match the outputs from model_qc.py which does scrambling experiments
                # These calls are needed to keep RNG in sync even though we don't use the results
                _ = rng.permuted(conditioning_crop_latent_vector)  # scrambled latent vector
                _ = rng.permuted(conditioning_input_crop.ravel())  # scrambled input image
                # Add noise_image to diffusion_input_crop with increasing weight
                noisy_diffusion_input_images = [
                    add_noise_to_image(diffusion_input_crop, noise_image, noise_level)
                    for noise_level in MODEL_QC_NOISE_LEVELS
                ]
                # Reconstruct starting with each noised ground truth image
                images_to_denoise = [*noisy_diffusion_input_images, noise_image]
                denoised_images = [
                    generate_from_coords_and_noised_image(
                        model, conditioning_crop_latent_vector, noised_image, num_gpus=NUM_GPUS
                    )
                    for noised_image in images_to_denoise
                ]

                # Save crops as TIFF files if requested
                if save_crops_as_tiff and include_in_metrics:
                    _save_all_crops_as_tiff(
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
                        logger=logger,
                    )

                # Compute metrics
                ground_truth = diffusion_input_crop.squeeze()

                metrics, metrics_100 = _compute_denoising_metrics(
                    ground_truth=ground_truth,
                    denoised_images=denoised_images,
                    lpips_calculator=lpips_calculator,
                    compute_all_noise_levels=save_intermediate_plots,
                )

                # Store the 100% noise metrics
                dataset_metrics["correlations_100"].append(metrics_100["correlation"])
                dataset_metrics["ssims_100"].append(metrics_100["ssim"])
                dataset_metrics["lpips_100"].append(metrics_100["lpips"])

                # Only create intermediate contact sheets if flag is set and this set is included
                if save_intermediate_plots and include_in_metrics:
                    # Prepare panels for contact sheet in COLUMN-MAJOR order
                    # (all images in column 0, then all in column 1, etc.)
                    contact_panels = [
                        *[conditioning_input_crop.squeeze()] * NUM_IMAGES_DENOISED,  # Column 0
                        *[ground_truth] * NUM_IMAGES_DENOISED,  # Column 1
                        *[img.squeeze() for img in images_to_denoise],  # Column 2
                        *[img.squeeze() for img in denoised_images],  # Column 3
                    ]

                    # Column titles for the 4 image columns
                    contact_col_titles = [
                        f"{label_for_conditioning} input",
                        "Original CDH5",
                        "Noised CDH5",
                        "Predicted CDH5",
                    ]

                    # Create figure with metrics column
                    fig = _create_contact_sheet_with_metrics_column(
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
                        direction="top-down first",  # Column-major for noise levels
                        show_row_header_column=True,  # Show noise level headers
                    )

                    save_plot_to_path(
                        fig,
                        output_path,
                        f"denoising_contact_sheet_{dataset_name}P{position}T{timepoint}X{start_x}Y{start_y}",
                    )
                    plt.close(fig)
                # Store results for summary figure (100% noise case)
                example_results_100.append(conditioning_input_crop.squeeze())
                example_results_100.append(ground_truth)
                example_results_100.append(denoised_images[-1].squeeze())
                example_metrics_100.append(metrics_100)
            # Store metrics for this dataset and model (only for validation and rep2)
            if include_in_metrics:
                all_metrics[example_set_label].append(dataset_metrics)
            # Plot summary figure with only the 100% noise denoising results (if flag is set)
            if save_intermediate_plots and include_in_metrics:
                num_img_cols = 3
                num_rows = len(example_set)

                # Create summary figure with metrics column
                fig = _create_contact_sheet_with_metrics_column(
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
                    direction="left-right first",  # Row-major for examples
                )

                # Add title very close to the top
                fig.suptitle(
                    f"100% Noise Denoising - {example_set_label} - Model {model_idx + 1}",
                    fontsize=FONTSIZE_LARGE,
                    y=0.995,
                )

                save_plot_to_path(
                    fig,
                    output_path,
                    f"contact_sheet_predict_all_examples_model{model_idx + 1}",
                )
                plt.close(fig)
    # Create comparison plots across all models
    logger.info("Creating comparison plots across models...")
    # Get output path for comparison plots
    comparison_output_path = get_output_path(
        "model_qc",
        "comparison",
        f"models_{len(model_manifest_names)}",
    )
    # Prepare data for plotting
    models_data = []
    for model_idx in range(len(model_manifest_names)):
        model_entry: dict[str, Any] = {
            "model_idx": model_idx,
            "model_label": None,
            "validation": {
                "corr_mean": 0.0,
                "corr_std": 0.0,
                "ssim_mean": 0.0,
                "ssim_std": 0.0,
                "lpips_mean": 0.0,
                "lpips_std": 0.0,
            },
            "rep2": {
                "corr_mean": 0.0,
                "corr_std": 0.0,
                "ssim_mean": 0.0,
                "ssim_std": 0.0,
                "lpips_mean": 0.0,
                "lpips_std": 0.0,
            },
        }
        # Find validation data for this model
        for data in all_metrics["validation_positions"]:
            if data["model_idx"] == model_idx:
                model_entry["model_label"] = data["model_label"]
                model_entry["validation"]["corr_mean"] = np.mean(data["correlations_100"])
                model_entry["validation"]["corr_std"] = np.std(data["correlations_100"])
                model_entry["validation"]["ssim_mean"] = np.mean(data["ssims_100"])
                model_entry["validation"]["ssim_std"] = np.std(data["ssims_100"])
                model_entry["validation"]["lpips_mean"] = np.mean(data["lpips_100"])
                model_entry["validation"]["lpips_std"] = np.std(data["lpips_100"])
                break
        # Find rep2 data for this model
        for data in all_metrics["rep_2_positions"]:
            if data["model_idx"] == model_idx:
                model_entry["rep2"]["corr_mean"] = np.mean(data["correlations_100"])
                model_entry["rep2"]["corr_std"] = np.std(data["correlations_100"])
                model_entry["rep2"]["ssim_mean"] = np.mean(data["ssims_100"])
                model_entry["rep2"]["ssim_std"] = np.std(data["ssims_100"])
                model_entry["rep2"]["lpips_mean"] = np.mean(data["lpips_100"])
                model_entry["rep2"]["lpips_std"] = np.std(data["lpips_100"])
                break
        models_data.append(model_entry)
    # Create legend text with full model names
    legend_text = "Model Details:\n" + "\n".join(
        [
            f"Model {i+1}: {model_manifest_names[i]}\n           Run: {run_names[i] if run_names[i] else 'latest'}"
            for i in range(len(models_data))
        ]
    )
    # Create comparison plots using helper function
    _create_comparison_bar_plot(
        models_data=models_data,
        metric_key="corr",
        ylabel="Pearson Correlation (100% Noise)",
        title="Correlation",
        output_path=comparison_output_path,
        filename="correlation_comparison_100_noise",
        legend_text=legend_text,
        ylim=(0, 1.0),
    )
    _create_comparison_bar_plot(
        models_data=models_data,
        metric_key="ssim",
        ylabel="SSIM Score (100% Noise)",
        title="SSIM",
        output_path=comparison_output_path,
        filename="ssim_comparison_100_noise",
        legend_text=legend_text,
        ylim=(0, 1.0),
    )
    _create_comparison_bar_plot(
        models_data=models_data,
        metric_key="lpips",
        ylabel="LPIPS Score (100% Noise)",
        title="LPIPS",
        output_path=comparison_output_path,
        filename="lpips_comparison_100_noise",
        legend_text=legend_text,
        text_box_loc="lower right",
    )
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Model Performance")
    logger.info("=" * 80)
    for model_data in models_data:
        logger.info(f"\n{model_data['model_label']}:")
        logger.info(
            f"  Validation - Corr: {model_data['validation']['corr_mean']:.3f} ± {model_data['validation']['corr_std']:.3f}, "
            f"SSIM: {model_data['validation']['ssim_mean']:.3f} ± {model_data['validation']['ssim_std']:.3f}, "
            f"LPIPS: {model_data['validation']['lpips_mean']:.3f} ± {model_data['validation']['lpips_std']:.3f}"
        )
        logger.info(
            f"  Rep2       - Corr: {model_data['rep2']['corr_mean']:.3f} ± {model_data['rep2']['corr_std']:.3f}, "
            f"SSIM: {model_data['rep2']['ssim_mean']:.3f} ± {model_data['rep2']['ssim_std']:.3f}, "
            f"LPIPS: {model_data['rep2']['lpips_mean']:.3f} ± {model_data['rep2']['lpips_std']:.3f}"
        )
    logger.info("=" * 80)

    logger.info("Model QC workflow completed successfully!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
