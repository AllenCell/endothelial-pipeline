from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
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
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS, UPLOAD_TO_FMS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
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
        load_nuclei_segmentation_image,
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import STRUCTURE_AWARE_COMPARISON_EXAMPLES, ExampleImage
    from endo_pipeline.settings.workflow_defaults import (
        CDH5_CONDITIONED_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_STRUCTURE_AWARE_QC_DATAFRAME_MANIFEST_PREFIX,
        GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    # Use the same datasets used in the binwidth/bandwidth sweep if no specific datasets are
    # provided
    datasets = datasets or STRUCTURE_AWARE_COMPARISON_EXAMPLES

    models_to_evaluate = [DEFAULT_MODEL_MANIFEST_NAME, CDH5_CONDITIONED_MODEL_MANIFEST_NAME]
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load the grid/patch-based feature dataframes for the selected datasets and pick a subset of
    # example crops to create structure-aware comparison metrics for
    df_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)

    # Load the dataframe and take a subset of the image crops to use as examples
    all_examples = []
    for dataset_name in datasets:
        duration = load_dataset_config(dataset_name).duration
        df = load_dataframe(df_manifest.locations[dataset_name], delay=True)
        columns_to_compute = [
            Column.DATASET,
            Column.POSITION,
            Column.TIMEPOINT,
            Column.CROP_INDEX,
            Column.DiffAEData.START_Y,
            Column.DiffAEData.START_X,
        ]
        df = df[columns_to_compute].compute().sample(n=100, replace=False, random_state=RANDOM_SEED)
        if not df[Column.TIMEPOINT].max() < duration:
            # If the timepoint of the randomly selected patch is not less that the dataset duration
            # then an error will occur when the workflow tries to load the image at timepoint + 1
            raise ValueError(
                f"Randomly selected timepoint is not less than duration for {dataset_name}"
            )
        all_examples.append(df)

    # Build list of examples paired with their next timepoint counterpart
    example_pairs = []
    for original_df in all_examples:
        copy_df = original_df.copy()
        copy_df[Column.TIMEPOINT] += 1
        for i in original_df.index:
            original_crop = original_df.loc[i].to_dict()
            next_crop = copy_df.loc[i].to_dict()
            example_original = ExampleImage(
                dataset_name=original_crop[Column.DATASET],
                position=original_crop[Column.POSITION],
                timepoint=original_crop[Column.TIMEPOINT],
                description=original_crop[Column.CROP_INDEX],
                crop_y_start=original_crop[Column.DiffAEData.START_Y],
                crop_x_start=original_crop[Column.DiffAEData.START_X],
            )
            example_next = ExampleImage(
                dataset_name=next_crop[Column.DATASET],
                position=next_crop[Column.POSITION],
                timepoint=next_crop[Column.TIMEPOINT],
                description=next_crop[Column.CROP_INDEX],
                crop_y_start=next_crop[Column.DiffAEData.START_Y],
                crop_x_start=next_crop[Column.DiffAEData.START_X],
            )
            example_pairs.append((example_original, example_next))

    # Expand the centre seed into a symmetric range for select number of seeds.
    num_seeds = 10
    half_range = num_seeds // 2
    seeds_to_evaluate = list(range(RANDOM_SEED - half_range, RANDOM_SEED - half_range + num_seeds))

    for m, model_manifest_name in enumerate(models_to_evaluate):
        # Get list of run names to calculate metrics for. If run names are not
        # provided, use all runs found in the manifest
        model_manifest = load_model_manifest(model_manifest_name)
        available_run_names = list(model_manifest.locations.keys())

        # Check to make sure all run names are available in manifest and exit the
        # workflow early if not
        if run_name not in available_run_names:
            logger.error("Run %s not found in manifest '%s'", run_name, model_manifest_name)
            return

        # In demo mode, limit to two model runs, two seeds, and two examples
        if DEMO_MODE:
            logger.warning("DEMO MODE - Limiting to two runs, two seeds, and two examples")
            seeds_to_evaluate = seeds_to_evaluate[:2]
            example_pairs = example_pairs[:2]

        # Build dataframe manifest for saving output dataframes
        demo_suffix = "_demo" if DEMO_MODE else ""
        name_suffix = f"_{model_manifest_name}{demo_suffix}"
        dataframe_manifest_name = (
            f"{DEFAULT_STRUCTURE_AWARE_QC_DATAFRAME_MANIFEST_PREFIX}{name_suffix}"
        )
        dataframe_manifest = create_dataframe_manifest(
            dataframe_manifest_name, workflow_name=__file__
        )

        # Update manifest with workflow parameters
        dataframe_manifest.parameters = {
            "model_manifest_name": model_manifest_name,
            "model_run_name": run_name,
            "random_seeds": [int(s) for s in seeds_to_evaluate],
            "datasets": datasets,
        }
        save_dataframe_manifest(dataframe_manifest)

        # Load all dataset configs used in example sets for saving to FMS
        if UPLOAD_TO_FMS:
            unique_dataset_names = sorted({example.dataset_name for example, _ in example_pairs})
            dataset_configs = [load_dataset_config(n) for n in unique_dataset_names]
            additional_notes = (
                f"DiffAE model comparison metrics across {num_seeds} seeds "
                "formatted with one row for each (run name, random seed, example image) combination."
            )

        # Initialize model comparison metrics calculator
        calculator = ModelComparisonMetricsCalculator()

        logger.info("Evaluating model manifest '%s' run '%s'", model_manifest_name, run_name)

        results: list[dict] = []

        # Load model for run and get model config. First load uninstantiated to
        # grab the model config, then instantiate for use downstream.
        model_location = model_manifest.locations[run_name]
        model_ = load_model(model_location, instantiate=False)
        config: DictConfig = cast(DictConfig, model_.cfg)
        model = instantiate_model_target_class(model_)  # type: ignore[reportArgumentType]

        for s, random_seed in enumerate(seeds_to_evaluate):
            # Set RNG seed
            rng = default_rng(seed=random_seed)

            for example, next_example in tqdm(
                example_pairs,
                desc=f"Model {m} / {len(models_to_evaluate)}: seed {s} / {len(seeds_to_evaluate)}",
            ):
                # Load transformed conditioning and diffusion examples
                conditioning_ex = load_transformed_conditioning_example_image(example, config)
                diffusion_ex = load_transformed_diffusion_example_image(example, config)
                next_diffusion_ex = load_transformed_diffusion_example_image(next_example, config)

                # Load nuclei segmentation image for the example (used to refine segmentations which
                # are used to compute segmentation-based features).
                img_nuc_seg = load_nuclei_segmentation_image(example, config)

                # Apply noise to conditioning image and then denoise
                noise = rng.standard_normal(size=conditioning_ex.shape)
                latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=NUM_GPUS)
                denoised_ex = generate_from_coords_and_noised_image(
                    model, latent, noise, num_gpus=NUM_GPUS, show_progress=False
                )

                scrambled_latent_control = generate_from_coords_and_noised_image(
                    model, rng.permuted(latent), noise, num_gpus=NUM_GPUS, show_progress=False
                )

                scrambled_image_latent = get_latent_vector_from_crop(
                    model, rng.permuted(conditioning_ex), num_gpus=NUM_GPUS
                )
                scrambled_image_control = generate_from_coords_and_noised_image(
                    model, scrambled_image_latent, noise, num_gpus=NUM_GPUS, show_progress=False
                )

                # Consume RNG state for reproducibility with model performance assessment outputs
                _ = rng.permuted(latent)
                _ = rng.permuted(conditioning_ex.ravel())

                # Calculate comparison metrics between the input image vs. itself,
                # input image at the current timepoint vs. the next timepoint, and
                # input image vs. denoised image for a given seed
                plot_name = f"{example}_{model_manifest_name}_{run_name}_seed{random_seed}"
                self_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=diffusion_ex,
                    img_nuc_seg=img_nuc_seg,
                    example_name=f"{plot_name}_self",
                )
                baseline_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=next_diffusion_ex,
                    img_nuc_seg=img_nuc_seg,
                    example_name=f"{plot_name}_baseline",
                )
                denoise_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=denoised_ex,
                    img_nuc_seg=img_nuc_seg,
                    example_name=f"{plot_name}_denoise",
                )
                scrambled_latent_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=scrambled_latent_control,
                    img_nuc_seg=img_nuc_seg,
                    example_name=f"{plot_name}_scrambled_latent",
                )
                scrambled_image_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=scrambled_image_control,
                    img_nuc_seg=img_nuc_seg,
                    example_name=f"{plot_name}_scrambled_image",
                )

                results.append(
                    {
                        # Model comparison general info
                        Column.DiffAEData.MODEL_MANIFEST: model_manifest_name,
                        Column.DiffAEData.MODEL_RUN: run_name,
                        Column.EXAMPLE_KEY: str(example),
                        Column.RANDOM_SEED: random_seed,
                        # Model comparison of input image with itself
                        Column.MODEL_COMPARISON_SELF_CORRELATION: self_results.correlation,
                        Column.MODEL_COMPARISON_SELF_SSIM: self_results.ssim,
                        Column.MODEL_COMPARISON_SELF_LPIPS: self_results.lpips,
                        Column.MODEL_COMPARISON_SELF_MEAN_ORIENTATION_DIFFERENCE: self_results.mean_orientation_difference,
                        Column.MODEL_COMPARISON_SELF_MEAN_RELATIVE_ASPECT_RATIO: self_results.mean_relative_aspect_ratio,
                        Column.MODEL_COMPARISON_SELF_MEAN_RELATIVE_AREA: self_results.mean_relative_area,
                        Column.MODEL_COMPARISON_SELF_NUMBER_OF_SEGMENTATIONS_DIFFERENCE: self_results.number_of_segmentations_difference,
                        Column.MODEL_COMPARISON_SELF_SSIM_SEG: self_results.ssim_seg,
                        Column.MODEL_COMPARISON_SELF_HAUSDORFF_SEG: self_results.hausdorff_seg,
                        # Model comparison of input image with baseline
                        Column.MODEL_COMPARISON_BASELINE_CORRELATION: baseline_results.correlation,
                        Column.MODEL_COMPARISON_BASELINE_SSIM: baseline_results.ssim,
                        Column.MODEL_COMPARISON_BASELINE_MEAN_ORIENTATION_DIFFERENCE: baseline_results.mean_orientation_difference,
                        Column.MODEL_COMPARISON_BASELINE_MEAN_RELATIVE_ASPECT_RATIO: baseline_results.mean_relative_aspect_ratio,
                        Column.MODEL_COMPARISON_BASELINE_MEAN_RELATIVE_AREA: baseline_results.mean_relative_area,
                        Column.MODEL_COMPARISON_BASELINE_NUMBER_OF_SEGMENTATIONS_DIFFERENCE: baseline_results.number_of_segmentations_difference,
                        Column.MODEL_COMPARISON_BASELINE_LPIPS: baseline_results.lpips,
                        Column.MODEL_COMPARISON_BASELINE_SSIM_SEG: baseline_results.ssim_seg,
                        Column.MODEL_COMPARISON_BASELINE_HAUSDORFF_SEG: baseline_results.hausdorff_seg,
                        # Model comparison of input image with denoised image
                        Column.MODEL_COMPARISON_CORRELATION: denoise_results.correlation,
                        Column.MODEL_COMPARISON_SSIM: denoise_results.ssim,
                        Column.MODEL_COMPARISON_LPIPS: denoise_results.lpips,
                        Column.MODEL_COMPARISON_MEAN_ORIENTATION_DIFFERENCE: denoise_results.mean_orientation_difference,
                        Column.MODEL_COMPARISON_MEAN_RELATIVE_ASPECT_RATIO: denoise_results.mean_relative_aspect_ratio,
                        Column.MODEL_COMPARISON_MEAN_RELATIVE_AREA: denoise_results.mean_relative_area,
                        Column.MODEL_COMPARISON_NUMBER_OF_SEGMENTATIONS_DIFFERENCE: denoise_results.number_of_segmentations_difference,
                        Column.MODEL_COMPARISON_SSIM_SEG: denoise_results.ssim_seg,
                        Column.MODEL_COMPARISON_HAUSDORFF_SEG: denoise_results.hausdorff_seg,
                        # Model comparison of input image with scrambled image
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_CORRELATION: scrambled_image_results.correlation,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_SSIM: scrambled_image_results.ssim,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_LPIPS: scrambled_image_results.lpips,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_MEAN_ORIENTATION_DIFFERENCE: scrambled_image_results.mean_orientation_difference,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_MEAN_RELATIVE_ASPECT_RATIO: scrambled_image_results.mean_relative_aspect_ratio,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_MEAN_RELATIVE_AREA: scrambled_image_results.mean_relative_area,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_NUMBER_OF_SEGMENTATIONS_DIFFERENCE: scrambled_image_results.number_of_segmentations_difference,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_SSIM_SEG: scrambled_image_results.ssim_seg,
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_HAUSDORFF_SEG: scrambled_image_results.hausdorff_seg,
                        # Model comparison of input image with scrambled latent vector
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_CORRELATION: scrambled_latent_results.correlation,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_SSIM: scrambled_latent_results.ssim,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_LPIPS: scrambled_latent_results.lpips,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_MEAN_ORIENTATION_DIFFERENCE: scrambled_latent_results.mean_orientation_difference,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_MEAN_RELATIVE_ASPECT_RATIO: scrambled_latent_results.mean_relative_aspect_ratio,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_MEAN_RELATIVE_AREA: scrambled_latent_results.mean_relative_area,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_NUMBER_OF_SEGMENTATIONS_DIFFERENCE: scrambled_latent_results.number_of_segmentations_difference,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_SSIM_SEG: scrambled_latent_results.ssim_seg,
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_HAUSDORFF_SEG: scrambled_latent_results.hausdorff_seg,
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
