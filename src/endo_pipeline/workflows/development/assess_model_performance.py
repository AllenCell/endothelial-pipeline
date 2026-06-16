from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    RANDOM_SEED,
)


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    example_groups: Annotated[
        list[Literal["training", "validation", "replicate"]] | None,
        Parameter(consume_multiple=True, negative_iterable=[]),
    ] = None,
    random_seed: int = RANDOM_SEED,
    save_crops_as_tiff: bool = False,
    include_negative_controls: Annotated[
        bool, Parameter(negative="--exclude-negative-controls")
    ] = True,
    include_intermediate_levels: Annotated[
        bool, Parameter(negative="--exclude-intermediate-levels")
    ] = True,
    include_metrics_in_plots: Annotated[
        bool, Parameter(negative="--exclude-metrics-in-plots")
    ] = True,
) -> None:
    """
    Assess performance for a single trained DiffAE model run.

    #diffae #model-performance #gpu

    Run denoising for a single model run to produce contact sheets for
    qualitative assessment of model performance.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe assess-model-performance -vd
    ```

    To run the workflow for select model manifest and run:

    ```bash
    uv run endopipe assess-model-performance MODEL_MANIFEST_NAME RUN_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will limit the
    examples in the contact sheets to two examples from each example group.

    Parameters
    ----------
    model_manifest_name
        Name of the DiffAE model manifest to evaluate.
    run_names
        Run name to evaluate. Defaults to most recent run.
    example_groups
        Example groups to use when assessing model performance.
    random_seed
        Seed for noise generation.
    save_crops_as_tiff
        True to save individual crops as TIFF files, False otherwise.
    include_negative_controls
        True to generate control contact sheets, False to skip.
    include_intermediate_levels
        True to generate intermediate noise level contact sheets, False to skip.
    include_metrics_in_plots
        True to include model comparison metrics on contact sheets, False to skip.
    """

    import logging
    from typing import cast

    from numpy.random import default_rng
    from omegaconf import DictConfig

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.io.load_models import instantiate_model_target_class
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.model.model_comparison import (
        ModelComparisonMetricsCalculator,
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
    )
    from endo_pipeline.library.model.model_performance import (
        add_noise_to_image,
        denoise_with_scrambled_conditioning_input,
        denoise_with_scrambled_latent_vector,
        save_model_performance_conditioning_and_diffusion_examples,
        save_model_performance_denoising_examples,
    )
    from endo_pipeline.library.visualize.model_performance import (
        plot_model_performance_intermediate_level_contact_sheet,
        plot_model_performance_negative_control_contact_sheet,
        plot_model_performance_summary_contact_sheet,
    )
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest
    from endo_pipeline.settings.examples import MODEL_COMPARISON_EXAMPLES
    from endo_pipeline.settings.workflow_defaults import MODEL_QC_NOISE_LEVELS

    logger = logging.getLogger(__name__)

    # Load model manifest and get run name (if not provided)
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = run_name or get_most_recent_run_name(model_manifest)

    output_path = get_output_path(__file__, model_manifest_name, run_name)

    # Get map of example group name to examples for selected example groups
    example_groups = example_groups or MODEL_COMPARISON_EXAMPLES.keys()
    examples = {
        group: examples
        for group, examples in MODEL_COMPARISON_EXAMPLES.items()
        if group in example_groups
    }

    # Limit to two examples from each example group if running in demo mode
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two examples from each example group")
        examples = {group: examples[:2] for group, examples in examples.items()}

    # Flatten list of examples
    all_examples = [example for examples in examples.values() for example in examples]

    # Define noise levels (appending 100% noise level)
    noise_levels = [*MODEL_QC_NOISE_LEVELS, 1.0] if include_intermediate_levels else [1.0]

    # Set RNG seed
    rng = default_rng(seed=random_seed)

    # Load model for run and get model config. First load the model without
    # instantiation to grab the model config, then instantiate for use later.
    model_location = model_manifest.locations[run_name]
    model_ = load_model(model_location, instantiate=False)
    model_config: DictConfig = cast(DictConfig, model_.cfg)
    model = instantiate_model_target_class(model_)
    conditioning_label = model_config.model.condition_key.replace("raw_", "").upper()

    # Collect 100% noise crops for summary figure
    all_conditioning_examples = []
    all_diffusion_examples = []
    all_denoised_examples = []
    all_comparison_metrics = []

    # Initialize model comparison metrics calculator (if needed)
    if include_metrics_in_plots:
        calculator = ModelComparisonMetricsCalculator()

    for example in all_examples:
        # Load transformed conditioning and diffusion examples
        conditioning_ex = load_transformed_conditioning_example_image(example, model_config)
        diffusion_ex = load_transformed_diffusion_example_image(example, model_config)

        # Apply different levels of noise to conditioning image and then denoise
        noise = rng.standard_normal(size=conditioning_ex.shape)
        latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=NUM_GPUS)
        noised_exs = [add_noise_to_image(diffusion_ex, noise, level) for level in noise_levels]
        denoised_exs = [
            generate_from_coords_and_noised_image(model, latent, noise, num_gpus=NUM_GPUS)
            for noise in noised_exs
        ]

        # Add 100% noise crops to list for use in summary figure, keyed by example
        all_conditioning_examples.append(conditioning_ex)
        all_diffusion_examples.append(diffusion_ex)
        all_denoised_examples.append(denoised_exs[-1])

        if save_crops_as_tiff:
            # Save the conditioning and diffusion crops out to tiff
            save_model_performance_conditioning_and_diffusion_examples(
                output_path=output_path,
                example=example,
                conditioning_example=conditioning_ex,
                diffusion_example=diffusion_ex,
            )

            # Save the noised and denoised crops out to tiff
            save_model_performance_denoising_examples(
                output_path=output_path,
                example=example,
                noise_examples=noised_exs,
                denoised_examples=denoised_exs,
                noise_levels=noise_levels,
            )

        if include_negative_controls:
            denoised_scrambled_latent = denoise_with_scrambled_latent_vector(
                rng, model, noised_exs, latent, NUM_GPUS
            )
            denoised_scrambled_input = denoise_with_scrambled_conditioning_input(
                rng, model, noised_exs, conditioning_ex, NUM_GPUS
            )

            plot_model_performance_negative_control_contact_sheet(
                output_path=output_path,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                example=example,
                conditioning_example=conditioning_ex,
                diffusion_example=diffusion_ex,
                noised_examples=noised_exs,
                denoised_examples=denoised_exs,
                denoised_scrambled_latent=denoised_scrambled_latent,
                denoised_scrambled_input=denoised_scrambled_input,
                noise_levels=noise_levels,
                conditioning_label=conditioning_label,
            )
        else:
            # Consume RNG state for reproducibility
            _ = rng.permuted(latent)
            _ = rng.permuted(conditioning_ex.ravel())

        if include_metrics_in_plots:
            comparison_metrics = [
                calculator.compute_all_metrics(diffusion_ex, denoised_ex)
                for denoised_ex in denoised_exs
            ]
            all_comparison_metrics.append(comparison_metrics[-1])
        else:
            comparison_metrics = None

        if include_intermediate_levels:
            plot_model_performance_intermediate_level_contact_sheet(
                output_path=output_path,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                example=example,
                conditioning_example=conditioning_ex,
                diffusion_example=diffusion_ex,
                noised_examples=noised_exs,
                denoised_examples=denoised_exs,
                noise_levels=noise_levels,
                comparison_metrics=comparison_metrics,
                conditioning_label=conditioning_label,
            )

    plot_model_performance_summary_contact_sheet(
        output_path=output_path,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        example_groups=examples,
        conditioning_examples=all_conditioning_examples,
        diffusion_examples=all_diffusion_examples,
        denoised_examples=all_denoised_examples,
        comparison_metrics=all_comparison_metrics,
        conditioning_label=conditioning_label,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
