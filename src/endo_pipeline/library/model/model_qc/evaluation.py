"""Core model evaluation logic for model QC."""

import logging
from typing import TYPE_CHECKING, NamedTuple, cast

from numpy.random import default_rng
from omegaconf import DictConfig, OmegaConf

from endo_pipeline.io import get_output_path, load_model
from endo_pipeline.io.mlflow import get_config_path_from_mlflow
from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
from endo_pipeline.library.model.diffae.generate_image import (
    add_noise_to_image,
    generate_from_coords_and_noised_image,
)
from endo_pipeline.manifests import load_model_manifest
from endo_pipeline.settings.workflow_defaults import DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT

from .image_loading import load_and_preprocess_example_crop
from .metrics import compute_baseline_for_example
from .plotting import (
    save_intermediate_contact_sheet,
    save_negative_control_sheet,
    save_summary_figure,
)
from .tiff_io import save_denoising_crops

if TYPE_CHECKING:
    from endo_pipeline.settings.examples import ExampleImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelKey - stable identity used for dict keys, output paths, and plot labels
# ---------------------------------------------------------------------------


class ModelKey(NamedTuple):
    """Manifest + run name pair uniquely identifying a model in an evaluation run.

    Hashable so it can serve as a dict key, and provides a ``.label`` property
    for display on figures (two-line ``manifest`` / ``run`` format used in
    suptitles and tick labels).
    """

    manifest_name: str
    run_name: str

    @property
    def label(self) -> str:
        """Human-readable label for display on figures."""
        return f"{self.manifest_name}\n{self.run_name}"


# ---------------------------------------------------------------------------
# Result dict helpers
# ---------------------------------------------------------------------------


def _empty_metric_bucket() -> dict[str, list]:
    """Return an empty metrics bucket for one example set.

    Returns
    -------
    bucket
        Dictionary with empty lists for each metric category.
    """
    return {
        "correlations_100": [],
        "ssims_100": [],
        "lpips_100": [],
        "baseline_correlations": [],
        "baseline_ssims": [],
        "baseline_lpips": [],
    }


def _make_result_skeleton(
    model_key: ModelKey,
    random_seed: int,
    example_set_labels: list[str],
) -> dict:
    """Build the top-level result dict with empty metric buckets.

    Creates a result dictionary pre-populated with metadata and one empty
    metric bucket per example set (e.g. ``"validation_positions"``,
    ``"rep_2_positions"``).

    Parameters
    ----------
    model_key
        ``ModelKey`` uniquely identifying this model within the evaluation
        batch.
    random_seed
        Random seed used for noise generation in this evaluation.
    example_set_labels
        Labels for each example set being evaluated.  Each label becomes a
        key in the returned dict, mapped to an empty metric bucket.

    Returns
    -------
    result
        Skeleton with keys ``"model_key"``, ``"random_seed"``,
        ``"example_set_labels"``, and one key per example set label
        containing empty metric lists.  Downstream consumers should
        derive ``model_label`` and ``run_name`` from ``model_key`` via
        ``model_key.label`` and ``model_key.run_name``.
    """
    result = {
        "model_key": model_key,
        "random_seed": random_seed,
        "example_set_labels": example_set_labels,
    }
    for label in example_set_labels:
        result[label] = _empty_metric_bucket()
    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_single_model(
    model_key: ModelKey,
    random_seed: int,
    example_sets_all: list[tuple[list["ExampleImage"], str]],
    example_sets_for_metrics: set[str],
    save_intermediate_plots: bool,
    save_crops_as_tiff: bool,
    include_negative_controls: bool,
    compute_metrics: bool,
    noise_levels: tuple[float, ...],
    compute_baseline: bool = True,
    is_default_seed: bool = True,
    num_gpus: int | None = None,
) -> dict:
    """Evaluate a single model and return its metrics.

    Supports both basic QC mode (negative controls) and comparison mode
    (quantitative metrics).

    Parameters
    ----------
    model_key
        ``ModelKey`` (manifest + resolved run name) identifying the model to
        evaluate.  The run name should already be resolved by the caller
        (e.g. ``None`` → most-recent run) so that the pair is a stable key.
    random_seed
        Random seed for noise generation reproducibility.
    example_sets_all
        Groups of example images to evaluate, each as a
        ``(examples, set_label)`` tuple.  ``examples`` is a list of
        :class:`ExampleImage` objects to run through the model, and
        ``set_label`` is a string identifier for the data split
        (e.g. ``"validation_positions"``, ``"rep_2_positions"``).
        The label is used as a key in the returned result dict.
    example_sets_for_metrics
        Example-set labels for which metrics should be computed.
    save_intermediate_plots
        Whether to save per-example contact sheets.
    save_crops_as_tiff
        Whether to save individual crops as TIFF files.
    include_negative_controls
        Whether to run negative-control experiments.
    compute_metrics
        Whether to compute quantitative metrics.
    noise_levels
        Tuple of noise fractions (e.g. ``(0.25, 0.5, 0.75)``) used for
        the denoising sweep.  A 100 % noise level is always appended
        automatically.
    compute_baseline
        Whether to compute next-timepoint baseline metrics.
    is_default_seed
        Whether this is the default seed (controls plot/crop saving).
    num_gpus
        Number of GPUs to use for model inference.  ``None`` uses CPU.

    Returns
    -------
    result
        Result dictionary containing per-example-set metrics.
    """
    manifest_name = model_key.manifest_name
    run_name = model_key.run_name

    # Conditional import for metrics
    lpips_calculator = None
    _compute_denoising_metrics = None
    if compute_metrics:
        from .image_metrics import LPIPSCalculator, compute_denoising_metrics

        _compute_denoising_metrics = compute_denoising_metrics
        lpips_calculator = LPIPSCalculator()

    # --- Setup ---
    noise_labels = [f"{level * 100:.0f}% Noise" for level in [*noise_levels, 1]]
    rng = default_rng(seed=random_seed)

    model_manifest = load_model_manifest(manifest_name)
    model_location = model_manifest.locations[run_name]

    logger.info("Processing model: %s [%s] (seed=%d)", manifest_name, run_name, random_seed)

    if model_location.mlflowid is not None:
        config_path = get_config_path_from_mlflow(model_location.mlflowid)
        model_config = cast(DictConfig, OmegaConf.create(config_path.read_text()))
    else:
        raise ValueError("mlflowid is None")
    crop_size = model_config.model.image_shape[-1]
    cond_key = model_config.model.condition_key
    diffusion_key = DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
    label_for_conditioning = "Brightfield" if cond_key == "raw_bf" else "CDH5"
    model = load_model(model_location, instantiate=True)

    result = _make_result_skeleton(
        model_key,
        random_seed,
        example_set_labels=[label for _, label in example_sets_all],
    )

    # --- Process each example set ---
    for example_set, example_set_label in example_sets_all:
        include_in_metrics = example_set_label in example_sets_for_metrics

        example_results_100: list = []
        example_metrics_100: list = []

        output_path = get_output_path(
            "model_qc",
            manifest_name,
            run_name,
            example_set_label,
            create_directories=False,
        )

        for example in example_set:

            # Load & crop
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
                num_gpus=num_gpus,
            )
            noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

            # Full noise levels when we need intermediate plots; 100%-only otherwise
            run_all = save_intermediate_plots or include_negative_controls
            if run_all:
                noisy_images = [
                    add_noise_to_image(diffusion_input_crop, noise_image, lvl)
                    for lvl in noise_levels
                ]
                images_to_denoise = [*noisy_images, noise_image]
            else:
                noisy_images = []
                images_to_denoise = [noise_image]

            denoised_images = [
                generate_from_coords_and_noised_image(model, latent, img, num_gpus=num_gpus)
                for img in images_to_denoise
            ]

            # Negative controls
            if include_negative_controls and is_default_seed:
                scrambled_emb = rng.permuted(latent)
                denoised_scrambled_emb = [
                    generate_from_coords_and_noised_image(
                        model, scrambled_emb, img, num_gpus=num_gpus
                    )
                    for img in images_to_denoise
                ]
                img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
                    conditioning_input_crop.shape
                )
                latent_scrambled_input = get_latent_vector_from_crop(
                    model,
                    img_scrambled,
                    num_gpus=num_gpus,
                )
                denoised_scrambled_input = [
                    generate_from_coords_and_noised_image(
                        model, latent_scrambled_input, img, num_gpus=num_gpus
                    )
                    for img in images_to_denoise
                ]
                save_negative_control_sheet(
                    conditioning_input_crop,
                    diffusion_input_crop,
                    images_to_denoise,
                    denoised_images,
                    denoised_scrambled_emb,
                    denoised_scrambled_input,
                    label_for_conditioning,
                    list(noise_levels),
                    output_path,
                    example,
                )
            else:
                # Consume RNG state for reproducibility
                _ = rng.permuted(latent)
                _ = rng.permuted(conditioning_input_crop.ravel())

            # Save crops - we do so only for the default random seed!
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
                    noise_levels=list(noise_levels),
                )

            ground_truth = diffusion_input_crop.squeeze()

            # Baseline metrics (next-timepoint comparison)
            if compute_baseline and compute_metrics and include_in_metrics:
                assert lpips_calculator is not None  # guaranteed by compute_metrics
                try:
                    corr, ssim_val, lpips_val = compute_baseline_for_example(
                        example,
                        model_config,
                        diffusion_key,
                        crop_size,
                        ground_truth,
                        lpips_calculator,
                    )
                    result[example_set_label]["baseline_correlations"].append(corr)
                    result[example_set_label]["baseline_ssims"].append(ssim_val)
                    result[example_set_label]["baseline_lpips"].append(lpips_val)
                except Exception:
                    logger.warning(
                        "Could not compute baseline for %s P%s T%s -> T%s",
                        example.dataset_name,
                        example.position,
                        example.timepoint,
                        example.timepoint + 1,
                        exc_info=True,
                    )

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
                    assert metrics is not None  # set when compute_all_noise_levels=True
                    save_intermediate_contact_sheet(
                        conditioning_input_crop,
                        ground_truth,
                        images_to_denoise,
                        denoised_images,
                        metrics,
                        label_for_conditioning,
                        noise_labels,
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
            save_summary_figure(
                example_results_100,
                example_metrics_100,
                len(example_set),
                label_for_conditioning,
                example_set_label,
                model_key,
                has_metrics,
                output_path,
            )

    return result
