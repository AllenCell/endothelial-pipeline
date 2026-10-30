from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, save_validation_plots: bool = False):
    """
    Calculate structure-aware comparison metrics for DiffAE model runs.

    #diffae #model-comparison #test-ready #gpu

    This workflow evaluates the default DiffAE model run from two model manifests (the default model
    and the CDH5-conditioned model). For each dataset, 100 image crops are randomly sampled from the
    grid-based feature dataframe, and each crop is paired with the same crop at the next timepoint.

    For each crop and each of 10 random seeds, the crop is encoded to a latent vector, noised, and
    denoised. The original image is then compared to:

    - itself (self control)
    - the next timepoint (baseline)
    - the denoised image
    - an image denoised from a scrambled latent vector (scrambled latent control)
    - an image denoised from a latent vector of scrambled pixels (scrambled image control)

    Each comparison reports correlation, SSIM, and LPIPS, as well as segmentation-based metrics
    (mean orientation difference, mean relative aspect ratio, mean relative area, difference in
    number of segmentations, segmentation SSIM, segmentation Hausdorff distance, and segmentation
    mean nearest boundary distance). Results are saved as one parquet file per model manifest, with
    one row for each (random seed, example image) combination, and catalogued in a dataframe
    manifest.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe calculate-structure-aware-comparison-metrics -d
    ```

    To run the workflow for specific datasets or dataset collections:

    ```bash
    uv run endopipe calculate-structure-aware-comparison-metrics \
        --datasets DATASET_NAME DATASET_NAME
    ```

    To run the workflow and save validation plots:

    ```bash
    uv run endopipe calculate-structure-aware-comparison-metrics --save-validation-plots
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will calculate metrics for two example
    crops and two seeds for each model manifest.

    Parameters
    ----------
    datasets
        Datasets to sample example crops from. Defaults to the datasets in
        `STRUCTURE_AWARE_COMPARISON_EXAMPLES`.
    save_validation_plots
        True to save validation plots of each image and its segmentation, False otherwise.
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
        get_refined_segmentation_from_downscaled_image,
        load_nuclei_segmentation_image,
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
        plot_image_validation,
    )
    from endo_pipeline.library.model.model_performance import (
        denoise_with_scrambled_conditioning_input,
        denoise_with_scrambled_latent_vector,
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

    # Use the default structure-aware comparison datasets if no specific datasets are provided
    datasets = datasets or STRUCTURE_AWARE_COMPARISON_EXAMPLES

    models_to_evaluate = [DEFAULT_MODEL_MANIFEST_NAME, CDH5_CONDITIONED_MODEL_MANIFEST_NAME]
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load the grid/patch-based feature dataframes for the selected datasets and randomly sample a
    # subset of crops to use as examples
    df_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)

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
        df = df[columns_to_compute].compute()
        # Exclude crops at the last timepoint, since each example is paired with the same crop at
        # timepoint + 1
        df = df[df[Column.TIMEPOINT] < duration - 1]
        df = df.sample(n=100, replace=False, random_state=RANDOM_SEED)
        all_examples.append(df)

    # Build list of examples paired with their next timepoint counterpart
    example_pairs: list[tuple[ExampleImage, ExampleImage]] = []
    description_prefix = "crop_"
    for df in all_examples:
        for row in df.to_dict("records"):
            example = ExampleImage(
                dataset_name=row[Column.DATASET],
                description=f"{description_prefix}{row[Column.CROP_INDEX]}",
                position=row[Column.POSITION],
                timepoint=row[Column.TIMEPOINT],
                crop_x_start=row[Column.DiffAEData.START_X],
                crop_y_start=row[Column.DiffAEData.START_Y],
            )
            example_pairs.append((example, example._replace(timepoint=example.timepoint + 1)))

    # Expand the centre seed into a symmetric range for select number of seeds.
    num_seeds = 10
    half_range = num_seeds // 2
    seeds_to_evaluate = list(range(RANDOM_SEED - half_range, RANDOM_SEED - half_range + num_seeds))

    # In demo mode, limit to two seeds and two examples
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two seeds and two examples")
        seeds_to_evaluate = seeds_to_evaluate[:2]
        example_pairs = example_pairs[:2]

    for m, model_manifest_name in enumerate(models_to_evaluate):
        # Load model manifest and check that the run is available
        model_manifest = load_model_manifest(model_manifest_name)
        if run_name not in model_manifest.locations:
            raise ValueError(f"Run '{run_name}' not found in manifest '{model_manifest_name}'")

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

        # Create one RNG per seed, shared across examples, so the draws for each example are the
        # same as when iterating over examples for each seed
        rngs = {random_seed: default_rng(seed=random_seed) for random_seed in seeds_to_evaluate}

        for example, next_example in tqdm(
            example_pairs, desc=f"Model {m + 1}/{len(models_to_evaluate)}"
        ):
            # Load transformed conditioning and diffusion examples
            conditioning_ex = load_transformed_conditioning_example_image(example, config)
            diffusion_ex = load_transformed_diffusion_example_image(example, config)
            next_diffusion_ex = load_transformed_diffusion_example_image(next_example, config)

            # Load nuclei segmentation image for the example (used to refine segmentations which
            # are used to compute segmentation-based features).
            img_nuc_seg = load_nuclei_segmentation_image(example, config)

            # Segment the input image once, since it is the reference image in every comparison,
            # and segment the input image at the next timepoint
            diffusion_ex_seg = get_refined_segmentation_from_downscaled_image(
                diffusion_ex, img_nuc_seg
            )
            next_diffusion_ex_seg = get_refined_segmentation_from_downscaled_image(
                next_diffusion_ex, img_nuc_seg
            )

            # Warn if segmentation failed, since the segmentation-based metrics will be NaN
            plot_name = f"{example}_{model_manifest_name}_{run_name}"
            for image_name, seg in [
                ("input", diffusion_ex_seg),
                ("next_timepoint", next_diffusion_ex_seg),
            ]:
                if seg is None:
                    logger.warning(
                        "Segmentation failed for example [ %s ]. "
                        "Setting segmentation metrics to NaN.",
                        f"{plot_name}_{image_name}",
                    )

            # Save validation plots of the input image and the input image at the next timepoint
            if save_validation_plots:
                plot_image_validation(
                    f"{plot_name}_input", diffusion_ex, img_nuc_seg, diffusion_ex_seg
                )
                plot_image_validation(
                    f"{plot_name}_next_timepoint",
                    next_diffusion_ex,
                    img_nuc_seg,
                    next_diffusion_ex_seg,
                )

            # Encode conditioning image to a latent vector
            latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=NUM_GPUS)

            # Calculate seed-independent comparison metrics between the input image vs. itself
            # and the input image at the next timepoint
            self_results = calculator.compute_all_metrics(
                img1=diffusion_ex,
                img2=diffusion_ex,
                img1_seg=diffusion_ex_seg,
                img2_seg=diffusion_ex_seg,
            )
            baseline_results = calculator.compute_all_metrics(
                img1=diffusion_ex,
                img2=next_diffusion_ex,
                img1_seg=diffusion_ex_seg,
                img2_seg=next_diffusion_ex_seg,
            )

            for random_seed in seeds_to_evaluate:
                rng = rngs[random_seed]

                # Denoise from random noise, then generate scrambled latent and scrambled image
                # controls from the same noise
                noise = rng.standard_normal(size=conditioning_ex.shape)
                denoised_ex = generate_from_coords_and_noised_image(
                    model, latent, noise, num_gpus=NUM_GPUS, show_progress=False
                )

                scrambled_latent_control = denoise_with_scrambled_latent_vector(
                    rng, model, [noise], latent, NUM_GPUS, show_progress=False
                )[0]
                scrambled_image_control = denoise_with_scrambled_conditioning_input(
                    rng, model, [noise], conditioning_ex, NUM_GPUS, show_progress=False
                )[0]

                # Segment the denoised image and the scrambled latent and scrambled image controls
                denoised_ex_seg = get_refined_segmentation_from_downscaled_image(
                    denoised_ex, img_nuc_seg
                )
                scrambled_latent_control_seg = get_refined_segmentation_from_downscaled_image(
                    scrambled_latent_control, img_nuc_seg
                )
                scrambled_image_control_seg = get_refined_segmentation_from_downscaled_image(
                    scrambled_image_control, img_nuc_seg
                )

                # Warn if segmentation failed, since the segmentation-based metrics will be NaN
                seed_plot_name = f"{plot_name}_seed{random_seed}"
                for image_name, seg in [
                    ("denoise", denoised_ex_seg),
                    ("scrambled_latent", scrambled_latent_control_seg),
                    ("scrambled_image", scrambled_image_control_seg),
                ]:
                    if seg is None:
                        logger.warning(
                            "Segmentation failed for example [ %s ]. "
                            "Setting segmentation metrics to NaN.",
                            f"{seed_plot_name}_{image_name}",
                        )

                # Save validation plots of the denoised image and the scrambled latent and
                # scrambled image controls
                if save_validation_plots:
                    plot_image_validation(
                        f"{seed_plot_name}_denoise",
                        denoised_ex,
                        img_nuc_seg,
                        denoised_ex_seg,
                    )
                    plot_image_validation(
                        f"{seed_plot_name}_scrambled_latent",
                        scrambled_latent_control,
                        img_nuc_seg,
                        scrambled_latent_control_seg,
                    )
                    plot_image_validation(
                        f"{seed_plot_name}_scrambled_image",
                        scrambled_image_control,
                        img_nuc_seg,
                        scrambled_image_control_seg,
                    )

                # Calculate comparison metrics between the input image vs. the denoised image, and
                # the scrambled latent and scrambled image controls for a given seed
                denoise_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=denoised_ex,
                    img1_seg=diffusion_ex_seg,
                    img2_seg=denoised_ex_seg,
                )
                scrambled_latent_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=scrambled_latent_control,
                    img1_seg=diffusion_ex_seg,
                    img2_seg=scrambled_latent_control_seg,
                )
                scrambled_image_results = calculator.compute_all_metrics(
                    img1=diffusion_ex,
                    img2=scrambled_image_control,
                    img1_seg=diffusion_ex_seg,
                    img2_seg=scrambled_image_control_seg,
                )

                results.append(
                    {
                        # General dataset info
                        Column.DATASET: example.dataset_name,
                        Column.POSITION: example.position,
                        Column.TIMEPOINT: example.timepoint,
                        Column.CROP_INDEX: int(
                            example.description.removeprefix(description_prefix)
                        ),
                        Column.DiffAEData.START_X: example.crop_x_start,
                        Column.DiffAEData.START_Y: example.crop_y_start,
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
                        Column.MODEL_COMPARISON_SELF_MEAN_BOUNDARY_DISTANCE_SEG: self_results.mean_boundary_distance_seg,
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
                        Column.MODEL_COMPARISON_BASELINE_MEAN_BOUNDARY_DISTANCE_SEG: baseline_results.mean_boundary_distance_seg,
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
                        Column.MODEL_COMPARISON_MEAN_BOUNDARY_DISTANCE_SEG: denoise_results.mean_boundary_distance_seg,
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
                        Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_MEAN_BOUNDARY_DISTANCE_SEG: scrambled_image_results.mean_boundary_distance_seg,
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
                        Column.MODEL_COMPARISON_SCRAMBLED_LATENT_MEAN_BOUNDARY_DISTANCE_SEG: scrambled_latent_results.mean_boundary_distance_seg,
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
