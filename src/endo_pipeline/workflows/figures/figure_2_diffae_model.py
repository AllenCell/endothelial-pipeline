def main() -> None:
    """
    Create figure 2 images
    """
    import logging

    import matplotlib.pyplot as plt
    from numpy.random import default_rng

    from endo_pipeline import NUM_GPUS
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
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.process.image_processing import contrast_stretching, crop_image
    from endo_pipeline.library.visualize.figure_utils import (
        add_scalebar,
        make_contact_sheet,
        plot_image_thumbnail,
    )
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
        EXAMPLE_DIFFAE_TRAINING_SCHEMATIC,
        EXAMPLES_DIFFAE_TRAINING_VALIDATION,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import (
        DIFFAE_ZARR_RESOLUTION_LEVEL,
        Z_SLICE_OFFSETS,
        PIXEL_SIZE_3i_20x,
    )
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_PLOT_DIRECTION,
        MODEL_QC_SUBPLOT_KWARGS,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        RANDOM_SEED,
    )

    plt.style.use("endo_pipeline.figure")
    import matplotlib.patches as patches

    logger = logging.getLogger(__name__)

    model_manifest_name: str = "diffae_baseline_exclude_cell_piling"
    run_name: str | None = None
    rng = default_rng(seed=RANDOM_SEED)

    # Load model manifest and get location for run_name
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model_location = model_manifest.locations[run_name_]

    # Model config has info about image processing steps from training
    # Also has the crop size
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    crop_size = model_config.model.image_shape[-1]  # assumes square crops

    # Get the condition and diffusion image keys from model config
    # e.g., model.condition_key = "raw_bf" and model.diffusion_key = "raw_cdh5"
    # means that the model was trained to denoise CDH5 images
    # conditioned on the semantic embedding of brightfield images
    channel_key_for_conditioning_input = model_config.model.condition_key
    label_for_conditioning = (
        "Brightfield" if channel_key_for_conditioning_input == "raw_bf" else "CDH5"
    )

    # Load model as instantiated Diff AE object
    model = load_model(model_location, instantiate=True)

    cond_crop_list = []
    diffusion_input_crop_list = []
    noise_image_list = []
    denoised_image_by_bf_cond_list = []
    denoised_images_by_random_cond_list = []
    denoised_images_by_random_cond_latent_scramble_list = []

    example_set = EXAMPLES_DIFFAE_TRAINING_VALIDATION

    for example in example_set:
        dataset_name = example.dataset_name
        logger.info(f"Processing model QC for dataset: {dataset_name}")

        # Extract position, timepoint, and crop position
        position = example.position
        timepoint = example.timepoint
        start_x = example.crop_x_start
        start_y = example.crop_y_start

        # Get output path for saving figures
        output_path = get_output_path(
            "figure_2_model_qc",
            model_manifest_name,
            run_name_,
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

        # Get zarr loading dictionary, get image processing steps
        # from loaded model config (except cropping step)
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

        if dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC:
            print(img.shape)
            center_slice = dataset_config.center_z_plane[position]
            cdh5_lower_slice = img[0, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
            cdh5_slice = img[0, center_slice, :, :].squeeze()
            cdh5_upper_slice = img[0, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()
            bf_lower_slice = img[1, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
            bf_slice = img[1, center_slice, :, :].squeeze()
            bf_upper_slice = img[1, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()

            for image, image_name, outline_color in zip(
                [
                    cdh5_lower_slice,
                    cdh5_slice,
                    cdh5_upper_slice,
                    bf_lower_slice,
                    bf_slice,
                    bf_upper_slice,
                ],
                [
                    "cdh5_lower_slice",
                    "cdh5_slice",
                    "cdh5_upper_slice",
                    "bf_lower_slice",
                    "bf_slice",
                    "bf_upper_slice",
                ],
                [
                    "white",
                    "white",
                    "white",
                    "black",
                    "black",
                    "black",
                ],
                strict=True,
            ):
                image = contrast_stretching(image)
                plot_image_thumbnail(
                    image,
                    f"{image_name}_{dataset_name}_T{timepoint}",
                    output_path,
                    figsize=(0.7, 0.7),
                    scalebar_size_um=50,
                    pixel_size=PIXEL_SIZE_3i_20x,
                    file_format=".pdf",
                    outline_color=outline_color,
                    bar_padding=30,
                    bar_thickness=20,
                )

            for image, image_name in zip(
                [transformed_diffusion_input_image, transformed_conditioning_input_image],
                ["diffusion_input_FOV", "conditioning_input_FOV"],
                strict=True,
            ):
                fig, ax = plot_image_thumbnail(
                    image.squeeze(),
                    f"{image_name}_{dataset_name}_T{timepoint}",
                    None,
                    figsize=(0.7, 0.7),
                    scalebar_size_um=50,
                    pixel_size=PIXEL_SIZE_3i_20x,
                    file_format=".pdf",
                    bar_thickness=20,
                    bar_padding=30,
                )
                rect = patches.Rectangle(
                    (start_x, start_y),
                    crop_size,
                    crop_size,
                    linewidth=0.5,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax.add_patch(rect)
                save_plot_to_path(fig, output_path, image_name, file_format=".pdf", pad_inches=0)

        # Crop both images to the same region
        conditioning_input_crop = crop_image(
            transformed_conditioning_input_image, start_x, start_y, crop_size
        )
        diffusion_input_crop = crop_image(
            transformed_diffusion_input_image, start_x, start_y, crop_size
        )

        if dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC:
            for image, image_name in zip(
                [conditioning_input_crop, diffusion_input_crop],
                ["conditioning_input_crop", "diffusion_input_crop"],
                strict=True,
            ):
                plot_image_thumbnail(
                    image.squeeze(),
                    f"{image_name}_{dataset_name}_T{timepoint}",
                    output_path,
                    figsize=(0.7, 0.7),
                    scalebar_size_um=10,
                    bar_padding=5,
                    bar_thickness=5,
                    pixel_size=PIXEL_SIZE_3i_20x,
                    file_format=".pdf",
                )

        # Get latent vector embedding of the crop used for
        # conditioning the denoising process
        conditioning_crop_latent_vector = get_latent_vector_from_crop(
            model, conditioning_input_crop, num_gpus=NUM_GPUS
        )

        # Sample random noise image with fixed seed
        noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

        denoised_image_by_bf_cond = generate_from_coords_and_noised_image(
            model, conditioning_crop_latent_vector, noise_image, num_gpus=NUM_GPUS
        )

        if dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC:
            plot_image_thumbnail(
                denoised_image_by_bf_cond.squeeze(),
                f"denoised_image_by_bf_cond_{dataset_name}_T{timepoint}",
                output_path,
                figsize=(0.7, 0.7),
                scalebar_size_um=10,
                bar_padding=5,
                bar_thickness=5,
                pixel_size=PIXEL_SIZE_3i_20x,
                file_format=".pdf",
            )
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
        noise_image_list.append(noise_image.squeeze())
        denoised_image_by_bf_cond_list.append(denoised_image_by_bf_cond.squeeze())
        denoised_images_by_random_cond_list.append(denoised_images_by_random_cond.squeeze())
        denoised_images_by_random_cond_latent_scramble_list.append(
            denoised_images_by_random_cond_latent_scramble.squeeze()
        )

    # Set defaults for plot titles
    CDH5_LABELS = [
        "Original\nVE-cadherin",
        "Noised\nVE-cadherin",
        "VE-cadherin\nembedding",
        "Scrambled\nembedding",
        "Scrambled\ninput image",
    ]

    panels = [
        img
        for img_list in [
            cond_crop_list,
            diffusion_input_crop_list,
            noise_image_list,
            denoised_image_by_bf_cond_list,
            denoised_images_by_random_cond_list,
            denoised_images_by_random_cond_latent_scramble_list,
        ]
        for img in img_list
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=3,
        max_cols=6,
        col_titles=[f"{label_for_conditioning}\ninput", *CDH5_LABELS],
        row_titles=None,
        direction=MODEL_QC_PLOT_DIRECTION,
        font_size=10,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs={"figsize": (MAX_FIGURE_WIDTH, 3.4)},
    )

    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0)
    all_axes = fig.get_axes()
    col_4_pos = all_axes[4].get_position()
    center_x = col_4_pos.x0 + (col_4_pos.width / 2)

    # get first subplot and add scalebar
    scalebar_um = 10
    add_scalebar(
        all_axes[0],
        pixel_size=PIXEL_SIZE_3i_20x,
        scale_bar_um=scalebar_um,
        bar_thickness=3,
        padding=5,
    )

    fig.text(
        x=center_x,
        y=0.97,
        s="Predicted VE-cadherin",
        ha="center",
        fontsize=10,
    )
    save_plot_to_path(
        fig,
        output_path,
        f"Model_QC_Examples_scalebar{scalebar_um}",
        file_format=".pdf",
        pad_inches=0,
        transparent=True,
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
