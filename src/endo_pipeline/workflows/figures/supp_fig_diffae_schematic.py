def main() -> None:
    """
    Create the DiffAE model training and eval schematic figure assets.

    Uses the brightfield-conditioned baseline model
    (``diffae_baseline_exclude_cell_piling``) to produce two deliverables:

      * The per-channel z-slice + FOV + crop thumbnails for the main-text
        figure-2 training/eval schematic diagram (from the single curated
        schematic example).
      * A figure-styled brightfield QC contact sheet over the remaining
        validation examples (encoder input / target / latent / negative
        controls), matching the layout of the VE-cadherin panel built by
        the ``supp-fig-diffae-model`` workflow.
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
        load_and_preprocess_example_crop,
        run_denoising_experiments,
    )
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_validation_examples_contact_sheet,
    )
    from endo_pipeline.library.visualize.model_training_schematic import (
        create_model_training_schematic_images,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position, load_model_manifest
    from endo_pipeline.settings.examples import (
        EXAMPLE_DIFFAE_TRAINING_SCHEMATIC,
        EXAMPLES_DIFFAE_TRAINING_VALIDATION,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL, PIXEL_SIZE_3i_20x
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        RANDOM_SEED,
    )

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    # Both deliverables come from the brightfield-conditioned baseline model.
    # Pin the latent-512 run explicitly: this is the variant used throughout
    # the figure-2 schematic, and ``get_most_recent_run_name`` would otherwise
    # resolve to a different latent-dimension run in the sweep manifest.
    model_manifest_name = "diffae_baseline_exclude_cell_piling"
    run_name = "20251110_latent_512"
    rng = default_rng(seed=RANDOM_SEED)

    # Load model manifest and get location for the pinned run_name
    model_manifest = load_model_manifest(model_manifest_name)
    model_location = model_manifest.locations[run_name]

    # Load model as instantiated Diff AE object. Crop size and the
    # conditioning channel key come straight off ``model.hparams`` so we
    # do not have to re-parse the training config for those two fields.
    model = load_model(model_location, instantiate=True)
    crop_size = model.hparams.image_shape[-1]  # assumes square crops

    # The baseline model conditions on brightfield, e.g. ``condition_key="raw_bf"``
    # + ``diffusion_key="raw_cdh5"`` means the model was trained to denoise
    # CDH5 images conditioned on the semantic embedding of brightfield images.
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

    output_path = get_output_path(
        "figure_2_model_qc",
        model_manifest_name,
        run_name,
    )

    # All validation examples except the schematic FOV are tiled into the
    # brightfield QC contact sheet; the schematic FOV instead drives the
    # figure-2 training/eval diagram thumbnails.
    cond_crop_list = []
    diffusion_input_crop_list = []
    denoised_by_cond_list = []
    denoised_scrambled_latent_list = []
    denoised_scrambled_input_list = []

    for example in EXAMPLES_DIFFAE_TRAINING_VALIDATION:
        dataset_name = example.dataset_name

        if dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC:
            logger.info(f"Processing training schematic for dataset: {dataset_name}")

            position = example.position
            timepoint = example.timepoint
            start_x = example.crop_x_start
            start_y = example.crop_y_start

            # The schematic thumbnails need the raw z-stack (for the per-z-slice
            # views) alongside the transformed full-FOV images, so this branch
            # loads inline rather than via ``load_and_preprocess_example_crop``
            # (which only returns the two crops).
            dataset_config = load_dataset_config(dataset_name)
            zarr_loc = get_zarr_location_for_position(dataset_config, position)
            img = load_image(
                zarr_loc,
                level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                timepoints=timepoint,
                squeeze=True,
                compute=True,
            )

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
            denoised_image_by_bf_cond = generate_from_coords_and_noised_image(
                model, conditioning_crop_latent_vector, noise_image, num_gpus=NUM_GPUS
            )

            create_model_training_schematic_images(
                dataset_config,
                img,
                position,
                timepoint,
                start_x,
                start_y,
                crop_size,
                transformed_diffusion_input_image,
                transformed_conditioning_input_image,
                conditioning_input_crop,
                diffusion_input_crop,
                denoised_image_by_bf_cond,
                noise_image,
                output_path,
            )
            continue

        # Remaining examples: denoise from pure noise (``noise_levels=[]``) for
        # the contact sheet, including the two negative controls (scrambled
        # latent + scrambled input image). Shares the inference helper with
        # the development model-QC workflow.
        logger.info(f"Processing brightfield QC for dataset: {dataset_name}")
        conditioning_input_crop, diffusion_input_crop = load_and_preprocess_example_crop(
            example,
            model_config,
            crop_size,
            channel_key_for_conditioning_input,
            DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        )
        denoising = run_denoising_experiments(
            model,
            conditioning_input_crop,
            diffusion_input_crop,
            rng,
            noise_levels=[],
            num_gpus=NUM_GPUS,
        )

        cond_crop_list.append(conditioning_input_crop.squeeze())
        diffusion_input_crop_list.append(diffusion_input_crop.squeeze())
        denoised_by_cond_list.append(denoising["denoised_normal"][0].squeeze())
        denoised_scrambled_latent_list.append(
            denoising["denoised_scrambled_embedding"][0].squeeze()
        )
        denoised_scrambled_input_list.append(denoising["denoised_scrambled_input"][0].squeeze())

    # Brightfield QC contact sheet -- same figure-styled build as the
    # VE-cadherin panel in ``supp-fig-diffae-model``.
    scalebar_um = 10
    fig = create_validation_examples_contact_sheet(
        cond_crop_list,
        diffusion_input_crop_list,
        denoised_by_cond_list,
        denoised_scrambled_latent_list,
        denoised_scrambled_input_list,
        label_for_conditioning,
        pixel_size=PIXEL_SIZE_3i_20x,
        figure_width=MAX_FIGURE_WIDTH,
        scalebar_um=scalebar_um,
    )
    save_plot_to_path(
        fig,
        output_path,
        f"Model_QC_Examples_scalebar{scalebar_um}",
        file_format=".svg",
        pad_inches=0,
        transparent=True,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
