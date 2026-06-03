from pathlib import Path

import numpy as np
from matplotlib import patches

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import contrast_stretching
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.settings.image_data import Z_SLICE_OFFSETS, PIXEL_SIZE_3i_20x_RESOLUTION_1


def plot_model_crop_thumbnails(
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    denoised_image_by_bf_cond: np.ndarray,
    dataset_name: str,
    timepoint: int,
    output_path: Path,
) -> None:
    """Plot conditioning input, diffusion input, and denoised crop thumbnails.

    Parameters
    ----------
    conditioning_input_crop
        Cropped conditioning input image.
    diffusion_input_crop
        Cropped diffusion input image.
    denoised_image_by_bf_cond
        Denoised image conditioned on brightfield.
    dataset_name
        Name of the dataset (used in filename).
    timepoint
        Timepoint index (used in filename).
    output_path
        Directory to save the output thumbnails.
    """
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
            scalebar_size_um=20,
            bar_padding=5,
            bar_thickness=5,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            scalebar_location="lower right",
        )

    plot_image_thumbnail(
        denoised_image_by_bf_cond.squeeze(),
        f"denoised_image_by_bf_cond_{dataset_name}_T{timepoint}",
        output_path,
        figsize=(0.7, 0.7),
        scalebar_size_um=20,
        bar_padding=5,
        bar_thickness=5,
        pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
        file_format=".svg",
        scalebar_location="lower right",
    )


def create_model_training_schematic_images(
    dataset_config: DatasetConfig,
    z_stack_img: np.ndarray,
    position: int,
    timepoint: int,
    start_x: int,
    start_y: int,
    crop_size: int,
    transformed_diffusion_input_image: np.ndarray,
    transformed_conditioning_input_image: np.ndarray,
    conditioning_input_crop: np.ndarray,
    diffusion_input_crop: np.ndarray,
    denoised_image_by_bf_cond: np.ndarray,
    noise_image: np.ndarray,
    output_path: Path,
) -> None:
    center_slice = (
        dataset_config.center_z_plane[position]
        if dataset_config.center_z_plane is not None
        else None
    )
    if center_slice is None:
        raise ValueError(f"center_slice is None for position {position}")
    cdh5_lower_slice = z_stack_img[0, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
    cdh5_slice = z_stack_img[0, center_slice, :, :].squeeze()
    cdh5_upper_slice = z_stack_img[0, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()
    bf_lower_slice = z_stack_img[1, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
    bf_slice = z_stack_img[1, center_slice, :, :].squeeze()
    bf_upper_slice = z_stack_img[1, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()

    slice_info = [
        (cdh5_lower_slice, "cdh5_lower_slice", "white"),
        (cdh5_slice, "cdh5_slice", "white"),
        (cdh5_upper_slice, "cdh5_upper_slice", "white"),
        (bf_lower_slice, "bf_lower_slice", "black"),
        (bf_slice, "bf_slice", "black"),
        (bf_upper_slice, "bf_upper_slice", "black"),
    ]

    for image, image_name, outline_color in slice_info:
        image = contrast_stretching(image)
        plot_image_thumbnail(
            image,
            f"{image_name}_{dataset_config.name}_T{timepoint}",
            output_path,
            figsize=(0.7, 0.7),
            scalebar_size_um=100,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            outline_color=outline_color,
            bar_padding=30,
            bar_thickness=20,
            scalebar_location="lower right",
        )

    for image, image_name in zip(
        [transformed_diffusion_input_image, transformed_conditioning_input_image],
        ["diffusion_input_FOV", "conditioning_input_FOV"],
        strict=True,
    ):
        fig, ax = plot_image_thumbnail(
            image.squeeze(),
            f"{image_name}_{dataset_config.name}_T{timepoint}",
            None,
            figsize=(0.7, 0.7),
            scalebar_size_um=100,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            bar_thickness=20,
            bar_padding=30,
            scalebar_location="lower right",
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
        save_plot_to_path(fig, output_path, image_name, file_format=".svg", pad_inches=0)

    plot_model_crop_thumbnails(
        conditioning_input_crop=conditioning_input_crop,
        diffusion_input_crop=diffusion_input_crop,
        denoised_image_by_bf_cond=denoised_image_by_bf_cond,
        dataset_name=dataset_config.name,
        timepoint=timepoint,
        output_path=output_path,
    )
    plot_image_thumbnail(
        noise_image.squeeze(),
        "noise_image",
        output_path,
        figsize=(0.7, 0.7),
        scalebar_size_um=20,
        bar_padding=5,
        bar_thickness=5,
        pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
        file_format=".svg",
        scalebar_location="lower right",
    )
