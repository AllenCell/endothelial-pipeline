import logging
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes
from skimage import transform
from skimage.exposure import rescale_intensity
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac

from endo_pipeline.configs import get_unannotated_positions, load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.process.z_stack_selection import get_plane_indices
from endo_pipeline.manifests import ImageLocation, get_zarr_location_for_position
from endo_pipeline.settings import ZARR_BRIGHTFIELD_CHANNEL, ZARR_EGFP_CHANNEL

logger = logging.getLogger(__name__)

DatasetPair = namedtuple("DatasetPair", ["target", "moving"])

# Sift registration ============================================================


def preprocess_image_for_sift(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for SIFT feature detection with percentile clipping
    and 0-1 normalization.

    Parameters
    ----------
    img
        The input image.

    Returns
    -------
    :
        The preprocessed image.
    """

    img = np.clip(img, np.percentile(img, 10), np.percentile(img, 99))
    img = (img - img.min()) / (img.max() - img.min())
    return img


def get_sift_keypoints_and_descriptors(
    image: np.ndarray, upsampling: int = 1, sigma_min: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect SIFT keypoints and descriptors in the given image.

    Parameters
    ----------
    image
        The input image.
    upsampling
        The number of times to upsample the image before detecting keypoints.
    sigma_min
        The minimum standard deviation for Gaussian smoothing.

    Returns
    -------
    :
        The detected keypoints.
    :
        The descriptors for the detected keypoints.
    """

    extractor = SIFT(upsampling=upsampling, sigma_min=sigma_min)
    extractor.detect_and_extract(image)
    keypoints = cast(np.ndarray, extractor.keypoints)
    descriptors = cast(np.ndarray, extractor.descriptors)

    return keypoints, descriptors


def sift_registration(
    image_target: np.ndarray,
    image_moving: np.ndarray,
    output_dir: Path,
    min_samples: int = 4,
    residual_threshold: int = 10,
    max_trials: int = 1000,
) -> transform.SimilarityTransform:
    """
    Register a moving image to a target image using SIFT keypoint matching and RANSAC.

    Parameters
    ----------
    image_target
        The reference image used for registration.
    image_moving
        The image that will be registered to the target image.
    output_dir
        Directory to save visualizations of keypoints.
    min_samples
        Minimum number of samples for RANSAC.
    residual_threshold
        Max distance for RANSAC inliers. Adjust based on image resolution/expected error.
    max_trials
        Maximum RANSAC iterations.
    """

    image_target = preprocess_image_for_sift(image_target)
    image_moving = preprocess_image_for_sift(image_moving)

    keypoints_target, descriptors_target = get_sift_keypoints_and_descriptors(image_target)
    keypoints_moving, descriptors_moving = get_sift_keypoints_and_descriptors(image_moving)

    # Brute force matching
    matches = match_descriptors(descriptors_moving, descriptors_target)
    logger.debug("Matches found: [ %d ]", len(matches))

    if len(matches) < min_samples:
        logger.error("Not enough matches found for RANSAC after cross-checking")
        raise RuntimeError("Not enough matches found for RANSAC after cross-checking")

    # Prepare data for RANSAC which expects (x, y)
    src = keypoints_moving[matches[:, 0]][:, ::-1]
    dst = keypoints_target[matches[:, 1]][:, ::-1]

    # Visualize keypoints
    plot_detected_keypoints(image_moving, src[:, ::-1], output_dir / "moving_keypoints")
    plot_detected_keypoints(image_target, dst[:, ::-1], output_dir / "target_keypoints")

    logger.info("Starting estimation of transformation using RANSAC")

    try:
        model_robust, inliers = ransac(
            (src, dst),
            transform.SimilarityTransform,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )

        if model_robust is None or sum(inliers) < 1:
            logger.error(
                "RANSAC failed to find a robust model. Inliers: [ %d ]",
                sum(inliers) if inliers is not None else 0,
            )
            raise RuntimeError("RANSAC failed to find a robust model")

        logger.info(
            "RANSAC transformation estimated successfully. Inliers: [ %d/%d ]",
            sum(inliers),
            len(matches),
        )
        logger.debug("RANSAC model parameters: \n%s", model_robust.params)

        # Set rotation and scaling to identity
        model_robust.params[:2, :2] = np.eye(2)
        logger.debug(
            "RANSAC model parameters (rotation and scaling to identity): \n%s", model_robust.params
        )
    except Exception as exception:
        logger.error("Error occurred during RANSAC transformation estimation")
        raise exception

    return model_robust


# Moving image processing ======================================================


def warp_moving_image(
    model: transform.ProjectiveTransform, target_image: np.ndarray, moving_image: np.ndarray
) -> np.ndarray:
    """
    Warp moving image to align with target image using the provided transformation model.

    Parameters
    ----------
    model
        The transformation model.
    target_image
        The target image to which the moving image will be aligned.
    moving_image
        The moving image to be warped.

    Returns
    -------
    :
        The warped moving image aligned to the target image.
    """

    warp = partial(
        transform.warp,
        inverse_map=model.inverse,
        output_shape=target_image.shape[-2:],
        order=3,
        mode="constant",
        cval=np.nan,
    )

    if len(target_image.shape) == 3:
        aligned_moving = np.stack([warp(moving_image[i]) for i in range(len(moving_image))])
    else:
        aligned_moving = warp(moving_image)

    return aligned_moving


def resize_moving_image(
    moving_image: np.ndarray, resize_factor: float | tuple[float, float, float]
) -> np.ndarray:
    """
    Resize image by given resizing factor.

    Parameters
    ----------
    moving_image
        The moving image to be resized.
    resize_factor
        The factor by which to resize the moving image.

    Returns
    -------
    :
        Resized moving image.
    """

    if np.all(resize_factor == 1.0):
        return moving_image

    return transform.rescale(moving_image, resize_factor, order=3)


def crop_images_to_overlap(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN values in the XY border of either of the passed images.

    We assume that the XY locations of the NaN values are the same across all
    Z slices if the images are 3D.

    Parameters
    ----------
    crop1
        The first image crop.
    crop2
        The second image crop.

    Returns
    -------
    :
        Tuple of the cropped images with NaN values removed.
    """

    slices_to_check = [image1[0], image2[0]] if len(image1.shape) == 3 else [image1, image2]
    valid_mask = ~np.isnan(np.stack(slices_to_check, axis=0)).any(axis=0)
    overlap = np.where(valid_mask)
    y_start, y_end = overlap[0].min(), overlap[0].max()
    x_start, x_end = overlap[1].min(), overlap[1].max()

    return (
        image1[..., y_start:y_end, x_start:x_end],
        image2[..., y_start:y_end, x_start:x_end],
    )


# Paired image alignment =======================================================


def build_live_fixed_dataset_pairs(datasets: list[str]) -> list[DatasetPair]:
    """
    Organize list of datasets into live ("target") and fixed ("moving") pairs.

    Iterate through the list of provided datasets to identify pairs using the
    target flag "PreFixation" and the moving flag "PostFixation".

    Parameters
    ----------
    datasets
        List of datasets to pair.

    Returns
    -------
    :
        List of paired "target" and "moving" dataset pairs.
    """

    target_flag = "PreFixation"
    moving_flag = "PostFixation"

    available_datasets = set(datasets)
    dataset_pairs = []

    while len(available_datasets) > 0:
        current_dataset = available_datasets.pop()

        # If neither flag is found in the name of the dataset, skip pairing.
        if target_flag not in current_dataset and moving_flag not in current_dataset:
            logger.warning(
                "Dataset [ %s ] does not have pairing flag. Skipping pairing.",
                current_dataset,
            )
            continue

        # If target flag found in dataset, attempt to pair to moving dataset.
        if target_flag in current_dataset:
            paired_dataset = current_dataset.replace(target_flag, moving_flag)

            if paired_dataset not in available_datasets:
                logger.warning(
                    "Paired moving dataset [ %s ] not found in given dataset list. "
                    "Skipping pairing for target dataset [ %s ].",
                    paired_dataset,
                    current_dataset,
                )
                continue

            available_datasets.remove(paired_dataset)
            dataset_pairs.append(DatasetPair(target=current_dataset, moving=paired_dataset))

        # If moving flag found in dataset, attempt to pair to target dataset.
        if moving_flag in current_dataset:
            paired_dataset = current_dataset.replace(moving_flag, target_flag)

            if paired_dataset not in available_datasets:
                logger.warning(
                    "Paired target dataset [ %s ] not found in given dataset list. "
                    "Skipping pairing for moving dataset [ %s ].",
                    paired_dataset,
                    current_dataset,
                )
                continue

            available_datasets.remove(paired_dataset)
            dataset_pairs.append(DatasetPair(target=paired_dataset, moving=current_dataset))

    return dataset_pairs


def align_dataset_pair(
    moving_image_location: ImageLocation,
    target_image_location: ImageLocation,
    moving_z_slices: list[int],
    target_z_slices: list[int],
    resolution_level: int,
    output_dir: Path,
) -> dict[str, str]:
    """
    Align moving images to target images using blob detection and registration.

    Parameters
    ----------
    moving_image_location
        Location of the moving image.
    target_image_location
        Location of the target image.
    moving_z_slices
        List of z-slices to load from the moving image.
    target_z_slices
        List of z-slices to load from the target image.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    output_dir
        Directory to save the aligned images.

    Returns
    -------
    :
        DataFrame containing paths to the aligned images.
    """

    # Load image readers to check dimensions and physical pixel sizes.
    target_image_reader = load_image(target_image_location, read=False)
    moving_image_reader = load_image(moving_image_location, read=False)

    if target_image_reader.shape[:-3] != moving_image_reader.shape[:-3]:
        logger.error(
            "The moving and target image must have the same non-spatial dimensions. "
            "Target shape: [ %s ], Moving shape: [ %s ]",
            target_image_reader.shape[:-3],
            moving_image_reader.shape[:-3],
        )
        raise ValueError("The moving and target image must have the same non-spatial dimensions.")

    # Initialize output dictionary of aligned file paths.
    aligned_files: dict[str, str] = {}

    # Use original image names as base for output file names.
    base_moving_name = moving_image_location.path.name.split(".")[0]
    base_target_name = target_image_location.path.name.split(".")[0]

    # Check physical pixel sizes in loaded images.
    moving_image_pixel_sizes: PhysicalPixelSizes = moving_image_reader.physical_pixel_sizes
    target_image_pixel_sizes: PhysicalPixelSizes = target_image_reader.physical_pixel_sizes

    if moving_image_pixel_sizes.X is None:
        logger.error("Physical pixel sizes for moving image not set: %s", moving_image_pixel_sizes)
        raise ValueError("Physical pixel sizes for moving image are not set.")

    if target_image_pixel_sizes.X is None:
        logger.error("Physical pixel sizes for target image not set: %s ", target_image_pixel_sizes)
        raise ValueError("Physical pixel sizes for target image are not set.")

    # Calculate rescaling factor from physical sizes.
    rescale_factor = target_image_pixel_sizes.X / moving_image_pixel_sizes.X
    logger.debug("Moving image rescale factor set to [ %f ]", rescale_factor)

    # Load image data
    target_image = load_image(target_image_location, level=resolution_level, compute=True)
    moving_image = load_image(moving_image_location, level=resolution_level, compute=True)

    # Get separate GFP and BF images for selected z slices.
    timepoint = 0
    target_gfp_image = target_image[timepoint, ZARR_EGFP_CHANNEL, target_z_slices, :, :]
    moving_gfp_image = moving_image[timepoint, ZARR_EGFP_CHANNEL, moving_z_slices, :, :]
    target_bf_image = target_image[timepoint, ZARR_BRIGHTFIELD_CHANNEL, target_z_slices, :, :]
    moving_bf_image = moving_image[timepoint, ZARR_BRIGHTFIELD_CHANNEL, moving_z_slices, :, :]

    # Resize moving image.
    moving_gfp_image = resize_moving_image(moving_gfp_image, rescale_factor)
    moving_bf_image = resize_moving_image(moving_bf_image, (1, rescale_factor, rescale_factor))

    # Apply standard deviation projection on GFP images to get alignment model.
    target_gfp_projection = target_gfp_image.std(0)
    moving_gfp_projection = moving_gfp_image.std(0)
    model = sift_registration(target_gfp_projection, moving_gfp_projection, output_dir)

    # Warp and crop the GFP images.
    moving_gfp_image = warp_moving_image(model, target_gfp_image, moving_gfp_image)
    moving_gfp_image, target_gfp_image = crop_images_to_overlap(moving_gfp_image, target_gfp_image)

    # Save the aligned GFP images
    moving_gfp_save_path = output_dir / f"{base_moving_name}_moving_gfp.ome.tiff"
    target_gfp_save_path = output_dir / f"{base_target_name}_target_gfp.ome.tiff"
    OmeTiffWriter.save(uri=moving_gfp_save_path, data=moving_gfp_image)
    OmeTiffWriter.save(uri=target_gfp_save_path, data=target_gfp_image)
    aligned_files["moving_gfp"] = moving_gfp_save_path.as_posix()
    aligned_files["target_gfp"] = target_gfp_save_path.as_posix()

    # Save overlay of GFP images.
    plot_aligned_overlay(
        moving_gfp_image, target_gfp_image, output_dir / f"{base_target_name}_overlay.png"
    )

    # Warp and crop the BF images.
    moving_bf_image = warp_moving_image(model, target_bf_image, moving_bf_image)
    moving_bf_image, target_bf_image = crop_images_to_overlap(moving_bf_image, target_bf_image)

    # Save the aligned BF images.
    moving_bf_save_path = output_dir / f"{base_moving_name}_moving_bf.ome.tiff"
    target_bf_save_path = output_dir / f"{base_target_name}_target_bf.ome.tiff"
    OmeTiffWriter.save(uri=moving_bf_save_path, data=moving_bf_image)
    OmeTiffWriter.save(uri=target_bf_save_path, data=target_bf_image)
    aligned_files["moving_bf"] = moving_bf_save_path.as_posix()
    aligned_files["target_bf"] = target_bf_save_path.as_posix()

    # Save combined image.
    target_bf_projection = target_bf_image.std(0)
    moving_bf_projection = moving_bf_image.std(0)
    combined_bf_image = np.stack([target_bf_projection, moving_bf_projection], axis=0)[:, None]
    combined_bf_save_path = output_dir / f"{base_target_name}_aligned_paired_bf.ome.tiff"
    OmeTiffWriter.save(uri=combined_bf_save_path, data=combined_bf_image)
    aligned_files["combined_bf"] = combined_bf_save_path.as_posix()

    return aligned_files


def align_all_positions_for_dataset_pair(
    dataset_pair: DatasetPair,
    resolution_level: int,
    z_slice_offsets: tuple[int, int],
    output_dir: Path,
    num_positions_to_align: int | None = None,
) -> pd.DataFrame:
    """
    Align all positions of the moving dataset to the target dataset.

    Parameters
    ----------
    dataset_pair
        Names of the target and moving datasets.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    output_dir
        The directory where the aligned images will be saved.
    num_positions_to_align
        Number of positions in the dataset to process for alignment.

    Returns
    -------
    :
        DataFrame containing the paths to the aligned images.
    """

    # Load dataset configs for each dataset.
    moving_dataset_config = load_dataset_config(dataset_pair.moving)
    target_dataset_config = load_dataset_config(dataset_pair.target)

    # Ensure that the two datasets have the same positions.
    if set(moving_dataset_config.zarr_positions) != set(target_dataset_config.zarr_positions):
        logger.error(
            "Positions in moving dataset '%s' %s do not match positions in target dataset '%s' %s",
            dataset_pair.moving,
            moving_dataset_config.zarr_positions,
            dataset_pair.target,
            target_dataset_config.zarr_positions,
        )
        raise ValueError("Datasets must have the same positions.")

    # Get include positions for moving and target datasets
    moving_include_positions = get_unannotated_positions(moving_dataset_config)
    target_include_positions = get_unannotated_positions(target_dataset_config)
    include_positions = set(moving_include_positions) & set(target_include_positions)

    aligned_image_paths = []
    aligned_positions_count = 0

    # Iterate through each position to align images.
    for position in moving_dataset_config.zarr_positions:
        if position not in include_positions:
            logger.warning(
                "Position [ %d ] skipped because it is annotated with a known imaging artifact",
                position,
            )
            continue

        moving_image_location = get_zarr_location_for_position(moving_dataset_config, position)
        target_image_location = get_zarr_location_for_position(target_dataset_config, position)

        logger.debug(
            "Aligning moving image [ %s ] to target image [ %s ]",
            moving_image_location.path.name,
            target_image_location.path.name,
        )

        moving_z_slices = get_plane_indices(
            moving_dataset_config,
            position,
            lower_offset=z_slice_offsets[0],
            upper_offset=z_slice_offsets[1],
        )
        target_z_slices = get_plane_indices(
            target_dataset_config,
            position,
            lower_offset=z_slice_offsets[0],
            upper_offset=z_slice_offsets[1],
        )

        paths = align_dataset_pair(
            moving_image_location,
            target_image_location,
            moving_z_slices,
            target_z_slices,
            resolution_level,
            output_dir,
        )

        aligned_image_paths.append(paths)
        aligned_positions_count += 1

        if num_positions_to_align is not None and aligned_positions_count >= num_positions_to_align:
            break

    return pd.DataFrame(aligned_image_paths)


# Image registration visualization =============================================


def get_normalized_projection_for_overlay(image: np.ndarray) -> np.ndarray:
    """Project and normalize the image for overlay visualization."""

    projection = image.max(0)
    projection = np.clip(projection, np.percentile(projection, 10), np.percentile(projection, 99))
    return rescale_intensity(projection, out_range="uint8")


def plot_aligned_overlay(
    moving_image: np.ndarray, target_image: np.ndarray, output_path: Path
) -> None:
    """Save overlay of aligned moving and target GFP images."""

    moving_projection = get_normalized_projection_for_overlay(moving_image)
    target_projection = get_normalized_projection_for_overlay(target_image)
    overlay = np.stack([moving_projection, target_projection, target_projection], axis=-1)

    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_detected_keypoints(image: np.ndarray, keypoints: np.ndarray, output_path: Path) -> None:
    """Plot and save detected keypoints on the image."""

    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:, 1], keypoints[:, 0], s=0.1, c="red", marker="o")
    plt.axis("off")
    plt.savefig(output_path, dpi=300)
    plt.close()
