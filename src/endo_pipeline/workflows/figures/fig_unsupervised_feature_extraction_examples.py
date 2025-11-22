# def main() -> None:
from pathlib import Path

from numpy.random import default_rng

# import numpy as np
# from skimage.exposure import rescale_intensity
# from skimage.morphology import binary_dilation
from endo_pipeline import NUM_GPUS
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_config_dict_from_mlflow, get_output_path, load_image, load_model

# from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
from endo_pipeline.library.model.diffae.generate_image import (
    add_noise_to_image,
    generate_from_coords_and_noised_image,
)
from endo_pipeline.library.process.image_processing import crop_image

# from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
# from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
#     calculate_derived_data_dynamics_dependent,
# )
# from endo_pipeline.library.process.general_image_preprocessing import (
#     DIMENSION_ORDER,
#     save_image_output,
# )
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail

# from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (  # apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)

# from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
#     get_seg_feat_plot_args,
#     hist_2d_of_feats,
#     mark_parallel,
#     mark_perpendicular,
# )
from endo_pipeline.manifests import (
    get_most_recent_run_name,
    get_zarr_location_for_position,
    load_model_manifest,
)
from endo_pipeline.settings import (
    DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
    DEFAULT_MODEL_ZARR_RESOLUTION_LEVEL,
    RANDOM_SEED,
)
from endo_pipeline.settings.examples import (
    CDH5_SEG_FIG_EXAMPLE,
    MODEL_QC_EXAMPLES_REP_2_POSITIONS,
    MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
    MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
)

# TAGS = []


# from endo_pipeline.settings.plot_defaults import (
#     MODEL_QC_FIG_KWARGS,
#     MODEL_QC_GRIDSPEC_KWARGS,
#     MODEL_QC_PLOT_DIRECTION,
#     MODEL_QC_SUBPLOT_KWARGS,
# )


# FOR LIBRARY FILE
def save_transform_example_intermediates(output_dir: Path, transforms: list, sample: dict) -> dict:
    for i, func in enumerate(transforms):
        # apply the next transform function in the list
        sample = func(sample)
        transform_name = func.__class__.__name__

        # make a subdirectory for each transform
        output_subdir = output_dir / f"step_{i}_{transform_name}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        # iterate through the channels in the sample image
        for channel_name, image_array in sample.items():
            panel_size = (2, 2)
            image_array = image_array.squeeze()

            # save the image for each channel
            if image_array.ndim == 2:
                filename = f"step_{i}_{transform_name}_{channel_name}"
                plot_image_thumbnail(
                    image_array,
                    filename,
                    output_subdir,
                    figsize=panel_size,
                    show_plot=False,
                )
            # if the image is a 3D image then save each z-slice separately
            elif image_array.ndim == 3:
                for z_index in range(image_array.shape[0]):
                    filename = f"step_{i}_{transform_name}_{channel_name}_Z{z_index}"
                    plot_image_thumbnail(
                        image_array[z_index],
                        filename,
                        output_subdir,
                        figsize=panel_size,
                        show_plot=False,
                    )
    return sample


output_dir = get_output_path(__file__, "A_preproc_examples")

dataset_name = CDH5_SEG_FIG_EXAMPLE.dataset_name
position = CDH5_SEG_FIG_EXAMPLE.position
timepoint = CDH5_SEG_FIG_EXAMPLE.timepoint

model_manifest_name = "diffae_baseline_exclude_cell_piling"
run_name = "20251110_latent_512"

# call "get_image_preprocessing_examples"


def get_image_preprocessing_examples(dataset_name: str, postion: int, timepoint: int) -> None:
    pass


output_dir = get_output_path(__file__, "B_model_architect_examples")


# Load model manifest and get location for run_name
model_manifest = load_model_manifest(model_manifest_name)
run_name = get_most_recent_run_name(model_manifest) if run_name is None else run_name
model_location = model_manifest.locations[run_name]

# Model config has info about image processing steps from training
# Also has the crop size
model_config = get_config_dict_from_mlflow(model_location.mlflowid)
transforms = get_image_transforms(model_config)
channel_key_for_conditioning_input = model_config.model.condition_key
crop_size = model_config.model.image_shape[-1]  # assumes square crops

# Load model as instantiated Diff AE object to create images with
model = load_model(model_location, instantiate=True)


# Instantiate random number generator
rng = default_rng(seed=RANDOM_SEED)
noise_levels = (1,)


panel_size = (2, 2)

example_sets = [
    MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
    MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
    MODEL_QC_EXAMPLES_REP_2_POSITIONS,
]
example_set_labels = ["training_positions", "validation_positions", "rep_2_positions"]

for example_set, example_set_label in zip(example_sets, example_set_labels):
    for example in example_set:

        # Make subdirectory for output so we know what it belonged to afterwards
        output_subdir = output_dir / example_set_label
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Extract dataset, position, timepoint, and crop position
        dataset_name = example.dataset_name
        position = example.position
        timepoint = example.timepoint
        start_x = example.crop_x_start
        start_y = example.crop_y_start

        # Load the image for the specified dataset, position, timepoint
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

        # apply the transforms and save a thumbnail image after each step
        sample = save_transform_example_intermediates(output_dir, transforms, data)

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
            for noise_level in noise_levels
        ]

        # Reconstruct starting with each noised ground truth image, and finally
        # the pure noise conditioned using the embedding of the corresponding
        # ground truth image used for conditioning.
        # will need to update generate method to do array shaping internally
        images_to_denoise = [noise_image]
        denoised_images_by_bf_cond = [
            generate_from_coords_and_noised_image(
                model, conditioning_crop_latent_vector, noised_image, num_gpus=NUM_GPUS
            )
            for noised_image in images_to_denoise
        ]

        for noise_level, input_img, output_img in zip(
            *(noise_levels, images_to_denoise, denoised_images_by_bf_cond), strict=True
        ):
            noise_level = round(noise_level * 100)
            input_img = input_img.squeeze()
            output_img = output_img.squeeze()

            plot_image_thumbnail(
                input_img,
                f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}_NL{noise_level}_input_image",
                output_dir,
                figsize=panel_size,
                show_plot=False,
            )
            plot_image_thumbnail(
                output_img,
                f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}_NL{noise_level}_output_image",
                output_dir,
                figsize=panel_size,
                show_plot=False,
            )

        plot_image_thumbnail(
            diffusion_input_crop.squeeze(),
            f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}_NL{noise_level}_ground_truth",
            output_dir,
            figsize=panel_size,
            show_plot=False,
        )

        plot_image_thumbnail(
            conditioning_input_crop.squeeze(),
            f"{dataset_name}_P{position}_T{timepoint}_X{start_x}_Y{start_y}_NL{noise_level}_conditioning_input",
            output_dir,
            figsize=panel_size,
            show_plot=False,
        )
# call "get_unsupervised_feature_extraction_imaging_panels"

# get_unsupervised_feature_extraction_imaging_panels()


# get_standard_dev_projection_example()

# get_standard_dev_projection_example_crops()

# get_pure_noise_and_reconstruction_example()
