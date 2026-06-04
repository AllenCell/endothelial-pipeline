def main() -> None:
    """
    Create the VE-cadherin conditioned model validation figure.

    Builds the composite supplemental DiffAE-model figure from two panels:
      * Panel A: the cdh5-conditioned positive-control QC contact sheet.
      * Panel B: the cross-model Rep-2 Pearson-correlation comparison bars.

    The brightfield-conditioned baseline training/eval schematic assets are
    produced by the separate ``supp-fig-diffae-schematic`` workflow.
    """
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
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
        create_rep2_correlation_bar_plot,
        create_validation_examples_contact_sheet,
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
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import (
        DIFFAE_ZARR_RESOLUTION_LEVEL,
        PIXEL_SIZE_3i_20x_RESOLUTION_1,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
        RANDOM_SEED,
    )

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    # Panel A renders the cdh5-conditioned positive-control contact
    # sheet. The loop is retained as a single-element iterable so the
    # per-model setup (manifest load, model instantiation, transforms)
    # stays grouped; the baseline / BF-conditioned schematic assets for
    # the main-text figure-2 diagram live in the ``supp-fig-diffae-schematic``
    # workflow.
    panel_a_svg_path = None
    model_manifest_name = "diffae_cdh5_conditioned"

    rng = default_rng(seed=RANDOM_SEED)

    # Load model manifest and get location for the pinned run_name.
    # Pin the latent-512 cdh5 run explicitly so the panel-A contact sheet
    # is reproducible regardless of new runs landing in the sweep manifest.
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = "20260130_latent_512"
    model_location = model_manifest.locations[run_name]

    # Load model as instantiated Diff AE object. Crop size and the
    # conditioning channel key come straight off ``model.hparams`` so we
    # do not have to re-parse the training config for those two fields.
    model = load_model(model_location, instantiate=True)
    crop_size = model.hparams.image_shape[-1]  # assumes square crops

    # Conditioning vs. diffusion channel keys, e.g.
    # ``condition_key="raw_bf"`` + ``diffusion_key="raw_cdh5"`` means the
    # model was trained to denoise CDH5 images conditioned on the
    # semantic embedding of brightfield images.
    channel_key_for_conditioning_input = model.hparams.condition_key
    label_for_conditioning = (
        "Brightfield" if channel_key_for_conditioning_input == "raw_bf" else "VE-cadherin"
    )

    # The image-preprocessing transforms still need the full training
    # config from MLflow: ``data.train_dataloaders.dataset.transform`` is
    # outside the ``model:`` block and is not persisted into
    # ``model.hparams`` by Lightning's ``save_hyperparameters``.
    ml_flowid = model_location.mlflowid if model_location.mlflowid else None
    if ml_flowid is None:
        raise ValueError(f"Model location MLflow ID is None for model {model_manifest_name}")
    config_path = get_config_path_from_mlflow(ml_flowid)
    model_config = cast(DictConfig, OmegaConf.create(config_path.read_text()))

    cond_crop_list = []
    diffusion_input_crop_list = []
    denoised_image_by_bf_cond_list = []
    denoised_images_by_random_cond_list = []
    denoised_images_by_random_cond_latent_scramble_list = []

    example_set = EXAMPLES_DIFFAE_TRAINING_VALIDATION

    output_path = get_output_path(
        "figure_2_model_qc",
        model_manifest_name,
        run_name,
    )

    for example in example_set:
        dataset_name = example.dataset_name
        logger.info(f"Processing model QC for dataset: {dataset_name}")

        position = example.position
        timepoint = example.timepoint
        start_x = example.crop_x_start
        start_y = example.crop_y_start

        dataset_config = load_dataset_config(dataset_name)
        zarr_loc = get_zarr_location_for_position(dataset_config, position)
        img = load_image(
            zarr_loc,
            level=DIFFAE_ZARR_RESOLUTION_LEVEL,
            timepoints=timepoint,
            squeeze=True,
            compute=True,
        )

        # Get zarr loading dictionary, get image processing steps
        # and apply the transforms for each channel
        data = create_data_dict_loaded_image(img)
        transforms = get_image_transforms(model_config)
        sample = apply_img_transforms(transforms, data)

        # Extract the processed conditioning and diffusion images
        # based on the output key from the transforms
        # Conditioning image can be brightfield or CDH5 depending on model,
        # but diffusion image is always CDH5 in our use case
        transformed_conditioning_input_image = get_target_image_from_sample(
            sample, target_key=channel_key_for_conditioning_input
        )
        transformed_diffusion_input_image = get_target_image_from_sample(
            sample, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
        )

        # Crop
        conditioning_input_crop = crop_image(
            transformed_conditioning_input_image, start_x, start_y, crop_size
        )
        diffusion_input_crop = crop_image(
            transformed_diffusion_input_image, start_x, start_y, crop_size
        )

        # Get latent vector embedding of the crop used for conditioning the denoising process
        conditioning_crop_latent_vector = get_latent_vector_from_crop(
            model, conditioning_input_crop, num_gpus=NUM_GPUS
        )

        # Sample random noise image with fixed seed
        noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

        denoised_image_by_bf_cond = generate_from_coords_and_noised_image(
            model, conditioning_crop_latent_vector, noise_image, num_gpus=NUM_GPUS
        )

        # The schematic example is rendered by the ``supp-fig-diffae-schematic``
        # workflow, not tiled into this contact sheet. Skip it here, but only
        # after the ``noise_image`` draw above so the RNG sequence -- and thus
        # the denoised outputs -- of the remaining examples is unchanged.
        if dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC:
            continue

        # Do the same thing but with the conditioning vector randomly shuffled
        # This is our negative control for the BF conditioning
        latent_vector_scrambled = rng.permuted(conditioning_crop_latent_vector)
        denoised_images_by_random_cond = generate_from_coords_and_noised_image(
            model, latent_vector_scrambled, noise_image, num_gpus=NUM_GPUS
        )

        # Do the same thing but with the conditioning vector retrieved from a
        # randomly shuffled version of the brightfield image
        # This is another negative control for the BF conditioning
        img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
            conditioning_input_crop.shape
        )
        latent_vector_from_img_scrambled = get_latent_vector_from_crop(
            model, img_scrambled, num_gpus=NUM_GPUS
        )
        denoised_images_by_random_cond_latent_scramble = generate_from_coords_and_noised_image(
            model, latent_vector_from_img_scrambled, noise_image, num_gpus=NUM_GPUS
        )

        cond_crop_list.append(conditioning_input_crop.squeeze())
        diffusion_input_crop_list.append(diffusion_input_crop.squeeze())
        denoised_image_by_bf_cond_list.append(denoised_image_by_bf_cond.squeeze())
        denoised_images_by_random_cond_list.append(denoised_images_by_random_cond.squeeze())
        denoised_images_by_random_cond_latent_scramble_list.append(
            denoised_images_by_random_cond_latent_scramble.squeeze()
        )

    # Contact-sheet build + save for the cdh5-conditioned panel A. The
    # figure-styled build is shared with the brightfield schematic
    # workflow via ``create_validation_examples_contact_sheet``.
    scalebar_um = 20
    fig = create_validation_examples_contact_sheet(
        cond_crop_list,
        diffusion_input_crop_list,
        denoised_image_by_bf_cond_list,
        denoised_images_by_random_cond_list,
        denoised_images_by_random_cond_latent_scramble_list,
        label_for_conditioning,
        pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
        figure_width=MAX_FIGURE_WIDTH,
        scalebar_um=scalebar_um,
        scalebar_location="lower right",
    )

    contact_sheet_name = f"Model_QC_Examples_scalebar{scalebar_um}"
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
        label_fontsize=10,
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
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=panel_b_svg_path,
            x_position=0.0,
            y_position=3.4,
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
