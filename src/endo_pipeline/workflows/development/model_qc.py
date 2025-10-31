from endo_pipeline.cli import tags
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED

TAGS = ["diffae", tags.TEST_READY, tags.GPU]


def main(
    model_manifest_name: str = "diffae_baseline_exclude_cell_piling",
    run_name: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> None:
    """
    Run quality check assessment of a newly trained Diffusion Autoencoder model.

    This workflow loads a trained DiffAE model specified by the model_manifest_name
    and run_name. It then selects a crop from a specified dataset and position,
    adds varying levels of noise to the ground truth CDH5 image, and uses the model
    to denoise the noised images conditioned on the corresponding conditioning image
    (brightfield or CDH5 depending on the model).

    It also performs negative control experiments by denoising using:
    - A randomly shuffled conditioning vector.
    - A conditioning vector obtained from randomly shuffling the image used for
        conditioning and passing the scrambled image through the encoder.

    The results are visualized in contact sheet figures and saved to an output
    path set inside the workflow based on the model manifest and run names.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Run name within the model manifest to load. If None, the most recent run is used.
    random_seed
        Random seed for reproducibility of noise generation.
    """
    import logging

    from numpy.random import default_rng

    from endo_pipeline import DEMO_MODE, NUM_GPUS
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
        add_noise_to_image,
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
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
    from endo_pipeline.settings import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        DEFAULT_MODEL_ZARR_RESOLUTION_LEVEL,
        MODEL_QC_NOISE_LEVELS,
    )
    from endo_pipeline.settings.examples import MODEL_QC_EXAMPLES
    from endo_pipeline.settings.plot_defaults import (
        MODEL_QC_FIG_KWARGS,
        MODEL_QC_GRIDSPEC_KWARGS,
        MODEL_QC_PLOT_DIRECTION,
        MODEL_QC_SUBPLOT_KWARGS,
    )

    logger = logging.getLogger(__name__)

    # Instantiate random number generator
    rng = default_rng(seed=random_seed)

    # Set defaults for plot titles
    CDH5_LABELS = ["Original CDH5", "Noised CDH5", "Denoised CDH5"]
    NOISE_LABELS = [f"{level * 100:.0f}% Noise" for level in [*MODEL_QC_NOISE_LEVELS, 1]]
    NUM_IMAGES_DENOISED = len(NOISE_LABELS)

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

    # Load Example Data
    if DEMO_MODE:
        logger.info("DEMO MODE: Limiting MODEL_QC_EXAMPLES to first example only.")
        MODEL_QC_EXAMPLES = MODEL_QC_EXAMPLES[:1]

    # Process each dataset
    for example in MODEL_QC_EXAMPLES:
        dataset_name = example["dataset_name"]
        logger.info(f"Processing model QC for dataset: {dataset_name}")

        # Extract position, timepoint, and crop position
        position = example["position"]
        timepoint = example["timepoint"]
        start_x, start_y = example["crop_position"]

        # Get output path for saving figures
        output_path = get_output_path(
            "model_qc",
            model_manifest_name,
            run_name_,
            f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}",
        )

        dataset_config = load_dataset_config(dataset_name)
        zarr_loc = get_zarr_location_for_position(dataset_config, position)
        img = load_image(
            zarr_loc,
            level=DEFAULT_MODEL_ZARR_RESOLUTION_LEVEL,
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

        # Crop both images to the same region
        conditioning_input_crop = crop_image(
            transformed_conditioning_input_image, start_x, start_y, crop_size
        )
        diffusion_input_crop = crop_image(
            transformed_diffusion_input_image, start_x, start_y, crop_size
        )

        # Get latent vector embedding of the crop used for
        # conditioning the denoising process
        conditioning_crop_latent_vector = get_latent_vector_from_crop(
            model, conditioning_input_crop, num_gpus=NUM_GPUS
        )

        # Sample random noise image with fixed seed
        noise_image = rng.standard_normal(size=diffusion_input_crop.shape)

        # Add noise_image to denoising_start_crop with increasing weight:
        noisy_diffusion_input_images = [
            add_noise_to_image(diffusion_input_crop, noise_image, noise_level)
            for noise_level in MODEL_QC_NOISE_LEVELS
        ]

        # Reconstruct starting with each noised ground truth image, and finally
        # the pure noise conditioned using the embedding of the corresponding
        # ground truth image used for conditioning.
        # will need to update generate method to do array shaping internally
        images_to_denoise = [*noisy_diffusion_input_images, noise_image]
        denoised_images_by_bf_cond = [
            generate_from_coords_and_noised_image(
                model, conditioning_crop_latent_vector, noised_image, num_gpus=NUM_GPUS
            )
            for noised_image in images_to_denoise
        ]

        # Plot these images!
        # Prepare arguments for contact sheet
        panels = [
            *[conditioning_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
            *[diffusion_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
            *[img.squeeze() for img in images_to_denoise],
            *[img.squeeze() for img in denoised_images_by_bf_cond],
        ]

        # Make a contact sheet summarizing the results
        fig = make_contact_sheet(
            panels=panels,
            max_rows=NUM_IMAGES_DENOISED,
            max_cols=None,
            col_titles=[f"{label_for_conditioning} Input", *CDH5_LABELS],
            row_titles=NOISE_LABELS,
            direction=MODEL_QC_PLOT_DIRECTION,
            subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
            gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
            fig_kwargs=MODEL_QC_FIG_KWARGS,
        )
        # save the figure
        save_plot_to_path(fig, output_path, "denoising_by_ground_truth_conditioning")

        # Break after the first contact sheet in DEMO_MODE
        if DEMO_MODE:
            break

        # Do the same thing but with the conditioning vector randomly shuffled
        # This is our negative control for the BF conditioning
        latent_vector_scrambled = rng.permuted(conditioning_crop_latent_vector)

        denoised_images_by_random_cond = [
            generate_from_coords_and_noised_image(
                model, latent_vector_scrambled, noised_image, num_gpus=NUM_GPUS
            )
            for noised_image in images_to_denoise
        ]

        # Plot these images!
        # Prepare arguments for contact sheet
        panels = [
            *[diffusion_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
            *[img.squeeze() for img in images_to_denoise],
            *[img.squeeze() for img in denoised_images_by_random_cond],
        ]

        # Make a contact sheet summarizing the results
        fig = make_contact_sheet(
            panels=panels,
            max_rows=NUM_IMAGES_DENOISED,
            max_cols=None,
            col_titles=CDH5_LABELS,
            row_titles=NOISE_LABELS,
            direction=MODEL_QC_PLOT_DIRECTION,
            subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
            gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
            fig_kwargs=MODEL_QC_FIG_KWARGS,
        )
        # save the figure
        save_plot_to_path(fig, output_path, "denoising_by_random_vector_conditioning")

        # Do the same thing but with the conditioning vector retrieved from a
        # randomly shuffled version of the brightfield image
        # This is another negative control for the BF conditioning
        img_scrambled = rng.permuted(conditioning_input_crop.ravel()).reshape(
            conditioning_input_crop.shape
        )
        latent_vector_scrambled = get_latent_vector_from_crop(
            model, img_scrambled, num_gpus=NUM_GPUS
        )

        denoised_images_by_random_cond = [
            generate_from_coords_and_noised_image(
                model, latent_vector_scrambled, noised_image, num_gpus=NUM_GPUS
            )
            for noised_image in images_to_denoise
        ]

        # Plot these images!
        # Prepare arguments for contact sheet
        panels = [
            *[img_scrambled.squeeze()] * NUM_IMAGES_DENOISED,
            *[diffusion_input_crop.squeeze()] * NUM_IMAGES_DENOISED,
            *[img.squeeze() for img in images_to_denoise],
            *[img.squeeze() for img in denoised_images_by_random_cond],
        ]

        # Make a contact sheet summarizing the results
        fig = make_contact_sheet(
            panels=panels,
            max_rows=NUM_IMAGES_DENOISED,
            max_cols=None,
            col_titles=["Scrambled Input", *CDH5_LABELS],
            row_titles=NOISE_LABELS,
            direction=MODEL_QC_PLOT_DIRECTION,
            subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
            gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
            fig_kwargs=MODEL_QC_FIG_KWARGS,
        )
        # save the figure
        save_plot_to_path(fig, output_path, "denoising_by_conditioning_on_scrambled_image")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
