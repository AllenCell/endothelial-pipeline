def main() -> None:
    """
    Create the DiffAE model comparison supplemental figure.

    Builds the composite supplemental DiffAE-model figure from two panels:
      * Panel A: cross-model denoising contact sheet showing 3 validation
        examples (rows) denoised by each model in the QC sweep (columns:
        Target VE-cadherin, BF-8, ..., BF-1024, CDH5-512, CDH5-1024).
      * Panel B: the cross-model Rep-2 Pearson-correlation comparison bars.

    The brightfield-conditioned baseline training/eval schematic assets are
    produced by the separate ``supp-fig-diffae-schematic`` workflow.
    """
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.random import default_rng
    from omegaconf import DictConfig, OmegaConf

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_model
    from endo_pipeline.io.mlflow import get_config_path_from_mlflow
    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.model.model_qc import (
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
    )
    from endo_pipeline.library.model.model_qc.results_io import load_results_from_manifests
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_cross_model_comparison_contact_sheet,
        create_rep2_correlation_bar_plot,
    )
    from endo_pipeline.manifests import (
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.examples import (
        EXAMPLE_DIFFAE_TRAINING_SCHEMATIC,
        EXAMPLES_DIFFAE_TRAINING_VALIDATION,
    )
    from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import (
        DIFFAE_ZARR_RESOLUTION_LEVEL,
        PIXEL_SIZE_3i_20x_RESOLUTION_1,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
        DEFAULT_MODEL_QC_LABEL_MAP,
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
        RANDOM_SEED,
    )

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    # Panel A: Cross-model denoising comparison contact sheet.
    # Rows = 3 validation examples (same as before).
    # Columns = Target VE-cadherin + one denoised output per model in
    # DEFAULT_MODEL_QC_LABEL_MAP (BF-8 ... BF-1024, CDH5-512, CDH5-1024).

    example_set = EXAMPLES_DIFFAE_TRAINING_VALIDATION
    examples_to_process = [
        ex for ex in example_set if ex.dataset_name != EXAMPLE_DIFFAE_TRAINING_SCHEMATIC
    ]

    output_path = get_output_path("figure_2_model_qc")

    # Storage for target VE-cadherin crops and per-model denoised outputs
    target_crops: list[np.ndarray] = []  # same three examples for every model
    noise_images: list[np.ndarray] = []  # same three noise images for every modle
    denoised_per_model: list[list[np.ndarray]] = []  # outer=models, inner=examples

    # Column labels: first column is ground truth, then one per model
    col_labels: list[str] = ["Target\nVE-cadherin", *DEFAULT_MODEL_QC_LABEL_MAP.values()]

    # Iterate over every model in the curated QC sweep
    for model_idx, ((manifest_name, run_name), label) in enumerate(
        DEFAULT_MODEL_QC_LABEL_MAP.items()
    ):
        logger.info(f"Panel A: processing model {label} ({manifest_name}/{run_name})")

        # Load and instantiate model
        model_manifest = load_model_manifest(manifest_name)
        model_location = model_manifest.locations[run_name]
        model = load_model(model_location, instantiate=True)
        crop_size = model.hparams.image_shape[-1]  # assumes square crops
        channel_key_for_conditioning = model.hparams.condition_key

        # Get image-preprocessing transforms from the MLflow training config
        ml_flowid = model_location.mlflowid if model_location.mlflowid else None
        if ml_flowid is None:
            raise ValueError(f"MLflow ID is None for {manifest_name}/{run_name}")
        config_path = get_config_path_from_mlflow(ml_flowid)
        model_config = cast(DictConfig, OmegaConf.create(config_path.read_text()))
        transforms = get_image_transforms(model_config)

        model_denoised: list[np.ndarray] = []

        for ex_idx, example in enumerate(examples_to_process):
            dataset_config = load_dataset_config(example.dataset_name)
            zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
            img = load_image(
                zarr_loc,
                level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                timepoints=example.timepoint,
                squeeze=True,
                compute=True,
            )

            data = create_data_dict_loaded_image(img)
            sample = apply_img_transforms(transforms, data)

            # Extract and crop the conditioning input for this model
            transformed_conditioning = get_target_image_from_sample(
                sample, target_key=channel_key_for_conditioning
            )
            conditioning_crop = crop_image(
                transformed_conditioning,
                example.crop_x_start,
                example.crop_y_start,
                crop_size,
            )

            # On the first model pass, also extract target VE-cadherin crops
            # and pre-generate noise images (shared across all models)
            if model_idx == 0:
                transformed_target = get_target_image_from_sample(
                    sample, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
                )
                target_crop = crop_image(
                    transformed_target,
                    example.crop_x_start,
                    example.crop_y_start,
                    crop_size,
                )
                target_crops.append(target_crop.squeeze())
                # Deterministic noise per example, identical across models
                example_rng = default_rng(seed=RANDOM_SEED + ex_idx)
                noise_images.append(example_rng.standard_normal(size=target_crop.shape))

            # Encode conditioning crop and generate denoised image
            latent_vector = get_latent_vector_from_crop(model, conditioning_crop, num_gpus=NUM_GPUS)
            denoised = generate_from_coords_and_noised_image(
                model, latent_vector, noise_images[ex_idx], num_gpus=NUM_GPUS
            )
            model_denoised.append(denoised.squeeze())

        denoised_per_model.append(model_denoised)

    # Build and save the cross-model comparison contact sheet
    scalebar_um = 20
    fig = create_cross_model_comparison_contact_sheet(
        target_crops=target_crops,
        denoised_per_model=denoised_per_model,
        col_labels=col_labels,
        pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
        figure_width=MAX_FIGURE_WIDTH,
        scalebar_um=scalebar_um,
        scalebar_location="lower right",
    )

    contact_sheet_name = "Example_images_across_models"
    save_plot_to_path(
        fig,
        output_path,
        contact_sheet_name,
        file_format=".svg",
        pad_inches=0,
        transparent=True,
    )
    panel_a_svg_path = output_path / f"{contact_sheet_name}.svg"

    # ------------------------------------------------------------------
    # Panel B: quantitative Rep-2 Pearson-correlation sweep across the
    # full DEFAULT_MODEL_QC roster (BF latent 8->1024 + CDH5 controls).
    # Driven entirely by the per-``manifest_name`` dataframe manifests
    # emitted by the ``calculate-model-comparison-metrics`` production
    # workflow -- no GPU work here, just load + aggregate + plot.
    # ------------------------------------------------------------------

    # One dataframe manifest per unique sweep ``manifest_name`` (e.g.
    # baseline + cdh5 positive control), produced by
    # ``calculate-model-comparison-metrics``.  Preserve first-seen order
    # so logging matches the curated sweep ordering.
    unique_manifest_names = list(dict.fromkeys(DEFAULT_MODEL_QC_MANIFEST_NAMES))
    dataframe_manifests = [
        load_dataframe_manifest(f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{mn}")
        for mn in unique_manifest_names
    ]
    for dfm in dataframe_manifests:
        logger.info(
            "Loaded dataframe manifest [ %s ] with %d locations.",
            dfm.name,
            len(dfm.locations),
        )
    all_seed_results, discovered_model_keys, seeds = load_results_from_manifests(
        dataframe_manifests
    )

    # Map (manifest_name, run_name) -> curated label so the bars come out
    # in the publication order regardless of how rows landed in the parquet.
    sweep_label_map = {
        (m, r): lbl
        for m, r, lbl in zip(
            DEFAULT_MODEL_QC_MANIFEST_NAMES,
            DEFAULT_MODEL_QC_RUN_NAMES,
            DEFAULT_MODEL_QC_LABELS,
            strict=True,
        )
    }
    missing = [
        k for k in discovered_model_keys if (k.manifest_name, k.run_name) not in sweep_label_map
    ]
    if missing:
        raise ValueError(
            "Panel B expects only the curated DEFAULT_MODEL_QC sweep models. "
            f"Unexpected entries in dataframe manifest: {missing}"
        )

    discovered = {(k.manifest_name, k.run_name): k for k in discovered_model_keys}
    ordered_pairs = [
        (m, r)
        for m, r in zip(DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True)
        if (m, r) in discovered
    ]
    sweep_model_keys = [discovered[p] for p in ordered_pairs]
    sweep_model_labels = [sweep_label_map[p] for p in ordered_pairs]

    example_sets_for_metrics = {"rep_2_positions"}
    all_metrics, _ = aggregate_seed_metrics(
        all_seed_results, sweep_model_keys, example_sets_for_metrics, seeds
    )
    baseline_data = compute_baseline_data(all_metrics, compute_baseline=False)
    models_data = build_models_data(
        all_metrics, sweep_model_keys, baseline_data, compute_baseline=False
    )

    # Panel B output goes alongside the qualitative per-model PDFs in the
    # parent figure_2_model_qc directory so they composite cleanly in
    # Illustrator.  Format / fonts mirror panel A's export so the bar
    # chart drops into the same panel layout without rescaling.
    panel_b_output_path = get_output_path("figure_2_model_qc")
    panel_b_filename = "Model_QC_Rep2_Correlation_Bars"
    create_rep2_correlation_bar_plot(
        models_data=models_data,
        model_labels=sweep_model_labels,
        output_path=panel_b_output_path,
        filename=panel_b_filename,
        figsize=(MAX_FIGURE_WIDTH - 0.3, 3.6),
        file_format=".svg",
        label_fontsize=FONTSIZE_MEDIUM,
        save_kwargs={"pad_inches": 0, "transparent": True},
    )
    panel_b_svg_path = panel_b_output_path / f"{panel_b_filename}.svg"

    # ------------------------------------------------------------------
    # Composite: panel A (cdh5 contact sheet) above panel B (Rep-2 bars).
    # ------------------------------------------------------------------
    figure_panels = [
        FigurePanel(
            letter="A",
            path=panel_a_svg_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="B",
            path=panel_b_svg_path,
            x_position=0.0,
            y_position=2.2,
            x_offset=0.25,
            y_offset=0.15,
        ),
    ]
    build_figure_from_panels(
        figure_panels,
        panel_b_output_path / "Supplemental_Figure_Diffae_Model.svg",
        width=MAX_FIGURE_WIDTH,
        height=7.3,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
