from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_MANIFEST_NAME


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_names: list[str] | None = None,
    example_groups: Annotated[
        list[Literal["training", "validation", "replicate"]] | None,
        Parameter(consume_multiple=True, negative_iterable=[]),
    ] = None,
):
    """
    Calculate comparison metrics for model runs in given DiffAE model manifest.

    #diffae #model-comparison #test-ready #gpu

    This workflow evaluates model runs in a given DiffAE model manifest by
    calculating per-example correlation, SSIM, and LPIPS metrics on select
    examples for each selected example group.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe calculate-model-comparison-metrics MODEL_MANIFEST_NAME -d
    ```

    To run the workflow for specific model runs:

    ```bash
    uv run endopipe calculate-model-comparison-metrics MODEL_MANIFEST_NAME \
        --run-names RUN_NAME RUN_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will calculate
    metrics for two runs, for two examples and two seeds each.

    Parameters
    ----------
    model_manifest_name
        Name of the DiffAE model manifest to evaluate.
    run_names
        List of run names to evaluate. Defaults to all runs in the manifest.
    example_groups
        Example groups to use when calculating comparison metrics.
    """

    import logging
    from typing import cast

    import pandas as pd
    from numpy.random import default_rng
    from omegaconf import DictConfig

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS, UPLOAD_TO_FMS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_model,
        upload_file_to_fms,
    )
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
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import MODEL_COMPARISON_EXAMPLES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    # Filter map of all model comparison examples down to the request example
    # groups. Flatten this map into a list of (group, example, example_next)
    # where example_next is a copy of the original example at the next timepoint
    # for use with baseline model comparisons.
    all_examples = {
        group: examples
        for group, examples in MODEL_COMPARISON_EXAMPLES.items()
        if example_groups is None or group in example_groups
    }
    example_pairs = [
        (group, example, example._replace(timepoint=example.timepoint + 1))
        for group, examples in all_examples.items()
        for example in examples
    ]

    # Expand the centre seed into a symmetric range for select number of seeds.
    num_seeds = 10
    half_range = num_seeds // 2
    seeds_to_evaluate = list(range(RANDOM_SEED - half_range, RANDOM_SEED - half_range + num_seeds))

    # Get list of run names to calculate metrics for. If run names are not
    # provided, use all runs found in the manifest
    model_manifest = load_model_manifest(model_manifest_name)
    available_run_names = list(model_manifest.locations.keys())
    run_names = run_names or available_run_names

    # Check to make sure all run names are available in manifest and exit the
    # workflow early if not
    missing_runs = set(run_names) - set(available_run_names)
    if missing_runs:
        logger.error("Runs %s not found in manifest '%s'", missing_runs, model_manifest_name)
        return

    # In demo mode, limit to two model runs, two seeds, and two examples
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two runs, two seeds, and two examples")
        run_names = run_names[: min(len(run_names), 2)]
        seeds_to_evaluate = seeds_to_evaluate[:2]
        example_pairs = example_pairs[:2]

    # Build dataframe manifest for saving output dataframes
    demo_suffix = "_demo" if DEMO_MODE else ""
    name_suffix = f"_{model_manifest_name}{demo_suffix}"
    dataframe_manifest_name = f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}{name_suffix}"
    dataframe_manifest = create_dataframe_manifest(dataframe_manifest_name, workflow_name=__file__)

    # Update manifest with workflow parameters
    dataframe_manifest.parameters = {
        "model_manifest_name": model_manifest_name,
        "random_seeds": [int(s) for s in seeds_to_evaluate],
        "examples_groups": list(all_examples.keys()),
    }
    save_dataframe_manifest(dataframe_manifest)

    # Load all dataset configs used in example sets for saving to FMS
    if UPLOAD_TO_FMS:
        unique_dataset_names = sorted({example.dataset_name for _, example, _ in example_pairs})
        dataset_configs = [load_dataset_config(n) for n in unique_dataset_names]
        additional_notes = (
            f"DiffAE model comparison metrics across {num_seeds} seeds "
            "formatted with one row for each (run name, random seed, example image) combination."
        )

    # Initialize model comparison metrics calculator
    calculator = ModelComparisonMetricsCalculator()

    for run_name in run_names:
        logger.info("Evaluating model manifest '%s' run '%s'", model_manifest_name, run_name)

        results: list[dict] = []

        # Load model for run and get model config. First load uninstantiated to
        # grab the model config, then instantiate for use downstream.
        model_location = model_manifest.locations[run_name]
        model_ = load_model(model_location, instantiate=False)
        config: DictConfig = cast(DictConfig, model_.cfg)
        model = instantiate_model_target_class(model_)

        for random_seed in seeds_to_evaluate:
            # Set RNG seed
            rng = default_rng(seed=random_seed)

            for group, example, next_example in example_pairs:
                # Load transformed conditioning and diffusion examples
                conditioning_ex = load_transformed_conditioning_example_image(example, config)
                diffusion_ex = load_transformed_diffusion_example_image(example, config)
                next_diffusion_ex = load_transformed_diffusion_example_image(next_example, config)

                # Apply noise to conditioning image and then denoise
                noise = rng.standard_normal(size=conditioning_ex.shape)
                latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=NUM_GPUS)
                denoised_ex = generate_from_coords_and_noised_image(
                    model, latent, noise, num_gpus=NUM_GPUS
                )

                # Consume RNG state for reproducibility with model performance
                # assessment outputs
                _ = rng.permuted(latent)
                _ = rng.permuted(conditioning_ex.ravel())

                # Calculate comparison metrics for baseline (between input
                # diffusion image at current timepoint and the next timepoint)
                # for the denoised image (between ground truth input diffusion
                # image and result of denoising the conditioning image)
                baseline_results = calculator.compute_all_metrics(diffusion_ex, next_diffusion_ex)
                denoise_results = calculator.compute_all_metrics(diffusion_ex, denoised_ex)

                results.append(
                    {
                        Column.DiffAEData.MODEL_MANIFEST: model_manifest_name,
                        Column.DiffAEData.MODEL_RUN: run_name,
                        Column.EXAMPLE_KEY: str(example),
                        Column.RANDOM_SEED: random_seed,
                        Column.MODEL_COMPARISON_EXAMPLE_GROUP: group,
                        Column.MODEL_COMPARISON_BASELINE_CORRELATION: baseline_results.correlation,
                        Column.MODEL_COMPARISON_BASELINE_SSIM: baseline_results.ssim,
                        Column.MODEL_COMPARISON_BASELINE_LPIPS: baseline_results.lpips,
                        Column.MODEL_COMPARISON_CORRELATION: denoise_results.correlation,
                        Column.MODEL_COMPARISON_SSIM: denoise_results.ssim,
                        Column.MODEL_COMPARISON_LPIPS: denoise_results.lpips,
                    }
                )

        # Save dataframe to file
        dataframe = pd.DataFrame(results)
        file_name = f"{model_manifest_name}_{run_name}_model_comparison{demo_suffix}.parquet"
        save_path = output_path / file_name
        dataframe.to_parquet(save_path, index=False)

        # Create location object with output path
        location = dataframe_manifest.locations.get(run_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_configs,
                model_manifest=model_manifest,
                run_name=run_name,
                additional_notes=additional_notes,
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        dataframe_manifest.locations[run_name] = location
        save_dataframe_manifest(dataframe_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
