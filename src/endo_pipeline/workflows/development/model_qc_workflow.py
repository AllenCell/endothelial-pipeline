def main(model_manifest_name, run_name) -> None:
    """QC a newly trained model."""

    from numpy.random import default_rng

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
    from endo_pipeline.io import (
        get_config_dict_from_mlflow,
        get_output_path,
        load_image_from_path,
        load_model,
    )
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
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest

    # In practice, these constants would live in endo_pipeline.settings
    # Set which image to load
    DATASET_NAME = "20250224_20X"
    POSITION = 0
    TIMEPOINT = 0
    # Set where to crop
    START_X = 100
    START_Y = 100
    # Set random seed and instantiate random number generator
    RANDOM_SEED = 47
    rng = default_rng(seed=RANDOM_SEED)
    # Set noise levels for corrupting CDH5
    NOISE_LEVELS = [0.25, 0.5, 0.75]

    # Load Example Data
    dataset_config = load_dataset_config(DATASET_NAME)
    zarr_path = get_zarr_file_for_position(dataset_config, POSITION)
    img = load_image_from_path(zarr_path, level=1, timepoints=TIMEPOINT, squeeze=True, compute=True)

    # Load model manifest and get location for run_name
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model_location = model_manifest.locations[run_name_]

    # Get output path for saving figures
    output_path = get_output_path("model_qc", model_manifest_name, run_name_)  # noqa: F841

    # Model config has info about image processing steps from training
    # Also has the crop size
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    crop_size = model_config.model.image_shape[-1]  # assumes square crops

    # Load model as instantiated Diff AE object
    model = load_model(model_location, instantiate=True)

    # Get zarr loading dictionary, get image processing steps
    # from loaded model config (except cropping step)
    # and apply the transforms for each channel
    data = create_data_dict_loaded_image(img)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    # Extract the processed brightfield vs CDH5 image
    # based on the output key from the transforms
    transformed_bf = get_target_image_from_sample(sample, target_key=f"raw_bf")
    transformed_cdh5 = get_target_image_from_sample(sample, target_key=f"raw_cdh5")

    # Crop both images to the same region
    bf_crop = crop_image(transformed_bf, START_X, START_Y, crop_size)
    cdh5_crop = crop_image(transformed_cdh5, START_X, START_Y, crop_size)

    # Get latent vector embedding of the brightfield crop
    # This is used to condition the denoising process
    bf_crop_latent_vector = get_latent_vector_from_crop(model, bf_crop, num_gpus=NUM_GPUS)

    # Sample random noise image with fixed seed
    noise_image = rng.standard_normal(size=cdh5_crop.shape())

    # Add noise_image to cdh5_crop with increasing weight:
    noisy_cdh5 = [
        add_noise_to_image(cdh5_crop, noise_image, noise_level) for noise_level in NOISE_LEVELS
    ]

    # Reconstruct starting with each noised CDH5, and finally the pure noise
    # conditioned using the embedding of the corresponding brightfield
    # will need to update generate method to do array shaping internally
    images_to_denoise = [*noisy_cdh5, noise_image]
    denoised_images_by_bf_cond = [  # noqa: F841
        generate_from_coords_and_noised_image(
            model, bf_crop_latent_vector, noised_image, num_gpus=NUM_GPUS
        )
        for noised_image in images_to_denoise
    ]

    # Plot these images!

    # Do the same thing but with the conditioning vector randomly shuffled
    # This is our negative control for the BF conditioning
    latent_vector_scrambled = rng.permuted(bf_crop_latent_vector)
    denoised_images_by_random_cond = [  # noqa: F841
        generate_from_coords_and_noised_image(
            model, latent_vector_scrambled, noised_image, num_gpus=NUM_GPUS
        )
        for noised_image in images_to_denoise
    ]

    # Plot these images!
