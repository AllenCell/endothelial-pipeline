import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes
from monai.inferers import SlidingWindowSplitter
from skimage import transform as tf
from skimage.exposure import rescale_intensity
from skimage.feature import SIFT, match_descriptors
from skimage.measure import block_reduce, ransac
from tqdm import tqdm, trange

from endo_pipeline.configs import (
    get_available_zarr_files,
    get_datasets_in_collection,
    get_position_integer_from_zarr_file_path,
    load_dataset_config,
)
from endo_pipeline.io import load_image_from_path
from endo_pipeline.library.model import get_include_positions, get_z_slice_bounds_per_position
from endo_pipeline.library.process.cdh5_preprocessing import preprocess

FLUOR_CHANNEL = 0
BF_CHANNEL = 1

logger = logging.getLogger(__name__)


def visualize_keypoints(image: np.ndarray, keypoints: np.ndarray, savepath: str) -> None:
    """
    Visualize the detected keypoints on the image.

    Parameters
    ----------
    image
        The input image.
    keypoints
        The coordinates of the detected keypoints.
    savepath
        The path where the visualization will be saved.
    """
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:, 1], keypoints[:, 0], s=0.1, c="red", marker="o")
    plt.axis("off")
    plt.savefig(savepath, dpi=300)
    plt.close()


def sift_preprocess(img: np.ndarray) -> np.ndarray:
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


def template_matching(
    image: np.ndarray, template: np.ndarray, scale: int = 3
) -> tuple[np.ndarray, float]:
    """
    Register a small moving image to a larger target image via sliding window correlation.

    Note that the moving image is assumed to be smaller than the target image.

    Parameters
    ----------
    image
        The target image used for registration.
    template
        The moving image that will be registered to the target image.
    scale
        The scale factor for downsampling the images before registration.
    """
    # Resize image to current scale
    downsampled_image = block_reduce(image, (scale, scale), np.max)
    downsampled_template = block_reduce(template, (scale, scale), np.max)

    downsampled_image = preprocess(downsampled_image).astype(np.float32)
    downsampled_template = preprocess(downsampled_template).astype(np.float32)

    # Ensure resized target image is larger than resized moving
    if (
        downsampled_image.shape[0] < downsampled_template.shape[0]
        or downsampled_image.shape[1] < downsampled_template.shape[1]
    ):
        raise ValueError(
            "Target image is smaller than moving image. Resized target shape:",
            f"{downsampled_image.shape}, moving shape: {downsampled_template.shape}",
        )
    # Perform template matching
    result = cv2.matchTemplate(downsampled_image, downsampled_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    print(f"Best score: {max_val}, Location: {max_loc}")
    return np.array(max_loc) * scale, max_val


def template_registration(
    image_target: np.ndarray,
    image_moving: np.ndarray,
    scale: int = 3,
    template_shape: None | Sequence[int] = None,
) -> tf.SimilarityTransform:
    """
    Register a moving image to a target image using template matching.

    Parameters
    ----------
    image_target
        The reference image used for registration.
    image_moving
        The image that will be registered to the target image.
    scale
        The scale factor for downsampling the images before registration.
    template_shape
        Optional, the shape of the template used for registration.
    """
    template_shape = template_shape or image_moving.shape[-2:]
    splitter = SlidingWindowSplitter(patch_size=template_shape, overlap=0.1, pad_mode=None)
    # Register each template to the target image
    best_transform = None
    best_score = 0.0
    for template in tqdm(splitter(torch.from_numpy(image_moving[None, None]))):
        transform, score = template_matching(
            image_target, template[0].numpy().squeeze(), scale=scale
        )
        if score > best_score:
            best_score = score
            best_transform = transform

    return tf.SimilarityTransform(
        translation=best_transform,
    )


def _get_sift(
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
    min_samples: int = 4,
    residual_threshold: int = 10,
    max_trials: int = 1000,
    visualize_keypoints_dir: str | None = None,
) -> tf.SimilarityTransform | None:
    """
    Register a moving image to a target image using SIFT keypoint matching and RANSAC.

    Parameters
    ----------
    image_target
        The reference image used for registration.
    image_moving
        The image that will be registered to the target image.
    min_samples
        Minimum number of samples for RANSAC.
    residual_threshold
        Max distance for RANSAC inliers. Adjust based on image resolution/expected error.
    max_trials
        Maximum RANSAC iterations.
    visualize_keypoints_dir
        Directory to save visualizations of keypoints. If None, no visualizations are saved.
    """
    image_target = sift_preprocess(image_target)
    image_moving = sift_preprocess(image_moving)

    keypoints_target, descriptors_target = _get_sift(image_target)
    keypoints_moving, descriptors_moving = _get_sift(image_moving)

    # brute force matching
    matches = match_descriptors(descriptors_moving, descriptors_target)
    logger.debug("Matches found: [ %d ]", len(matches))

    if len(matches) < min_samples:
        logger.warning("Not enough matches found for RANSAC after cross-checking.")
        return None

    # Prepare data for RANSAC
    src = keypoints_moving[matches[:, 0]][:, ::-1]  # RANSAC expects (x, y)
    dst = keypoints_target[matches[:, 1]][:, ::-1]  # RANSAC expects (x, y)

    if visualize_keypoints_dir:
        visualize_keypoints(image_moving, src[:, ::-1], "moving_matched")
        visualize_keypoints(image_target, dst[:, ::-1], "target_matched")

    logger.info("Estimating transformation using RANSAC...")
    try:
        model_robust, inliers = ransac(
            (src, dst),
            tf.SimilarityTransform,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )

        if model_robust is None or sum(inliers) < 1:
            logger.warning(
                "RANSAC failed to find a robust model. Inliers: [ %d ]",
                sum(inliers) if inliers is not None else 0,
            )
            return None

        logger.info(
            "RANSAC transformation estimated successfully. Inliers: [ %d/%d ]",
            sum(inliers),
            len(matches),
        )
        logger.debug("RANSAC model parameters: \n%s", model_robust.params)

        model_robust.params[:2, :2] = np.eye(2)  # Set rotation and scaling to identity
        logger.debug(
            "Estimated model parameters after setting rotation and scaling to identity: \n%s",
            model_robust.params,
        )

    except Exception as e:
        logger.warning("Error during RANSAC transformation estimation: [ %s ]", e)
        return None
    return model_robust


def warp(
    model: tf.ProjectiveTransform, image_target: np.ndarray, image_moving: np.ndarray
) -> np.ndarray:
    """
    Warp the moving image to align with the target image using the provided transformation model.

    Parameters
    ----------
    model
        The transformation model.
    image_target
        The target image to which the moving image will be aligned.
    image_moving
        The moving image to be warped.

    Returns
    -------
    :
        The warped moving image aligned to the target image.
    """
    print("Warping image...")
    warp_transform = partial(
        tf.warp,
        inverse_map=model.inverse,
        output_shape=image_target.shape[-2:],
        order=3,
        mode="constant",
        cval=np.nan,
    )
    is_3d = len(image_target.shape) == 3
    if is_3d:
        aligned_moving = np.stack(
            [warp_transform(image_moving[i]) for i in trange(len(image_moving))]
        )
    else:
        aligned_moving = warp_transform(image_moving)
    return aligned_moving


def resize_moving(image_moving: np.ndarray, resize_factor: float | Sequence[float]) -> np.ndarray:
    """
    Resize the moving image to match the target image dimensions.

    Parameters
    ----------
    image_moving
        The moving image to be resized.
    resize_factor
        The factor by which to resize the moving image.

    Returns
    -------
    :
        Resized moving image.
    """
    if np.all(resize_factor == 1.0):
        return image_moving
    resized_image_moving = tf.rescale(image_moving, resize_factor, order=3)
    return resized_image_moving


def crop_to_overlap(crop1: np.ndarray, crop2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN values present in the XY border of either of the passed images.

    It is assumed that the XY locations of the NaN values are the same across all
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
    slices_to_check = [crop1[0], crop2[0]] if len(crop1.shape) == 3 else [crop1, crop2]
    valid_mask = ~np.isnan(np.stack(slices_to_check, axis=0)).any(axis=0)
    overlap = np.where(valid_mask)
    y_start, y_end = overlap[0].min(), overlap[0].max()
    x_start, x_end = overlap[1].min(), overlap[1].max()
    return (
        crop1[..., y_start:y_end, x_start:x_end],
        crop2[..., y_start:y_end, x_start:x_end],
    )


def overlay_normalize(img: np.ndarray) -> np.ndarray:
    """Project and normalize the image for overlay visualization."""
    proj = img.max(0)
    proj = np.clip(proj, np.percentile(proj, 10), np.percentile(proj, 99))
    proj = rescale_intensity(proj, out_range="uint8")
    return proj


def save_overlay(moving: np.ndarray, target: np.ndarray, savepath: str | Path) -> None:
    """Save overlay of aligned moving and target fluorescent images."""
    moving_proj = overlay_normalize(moving)
    target_proj = overlay_normalize(target)
    overlay = np.stack([moving_proj, target_proj, target_proj], axis=-1)
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(savepath, dpi=300)
    plt.close()


def align(
    moving_image_path: str | Path,
    target_image_path: str | Path,
    moving_z_slices: list[int],
    target_z_slices: list[int],
    resolution_level: int,
    savedir: Path,
    alignment_method: Literal["sift", "template"],
    align_fluo: bool = True,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Align a moving image to a target image using blob detection and registration.

    **Alignment methods**

    Options for alignment methods are "sift" or "template", passed in via the ``alignment_method``
    input. The method "sift" is recommended for the 20X pre/post fixation datasets, while the
    method "template" is recommended for the 20X/40X datasets.

    Parameters
    ----------
    moving_image_path
        Path to the moving image.
    target_image_path
        Path to the target image.
    moving_z_slices
        List of z-slices to load from the moving image.
    target_z_slices
        List of z-slices to load from the target image.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    savedir
        Directory to save the aligned images.
    alignment_method
        The method used for alignment.
    align_fluo
        If True, align the fluorescent channel. If False, do not align the fluorescent channel.
    **alignment_kwargs
        Additional arguments for the alignment function.

    Returns
    -------
    :
        DataFrame containing the paths to the aligned images.
    """
    if alignment_method not in ["sift", "template"]:
        logger.error(
            "Invalid alignment method: [ %s ]. Choose 'sift' or 'template'.", alignment_method
        )
        raise ValueError("Invalid alignment method. Choose 'sift' or 'template'.")

    # load images, both brightfield and fluorescent channels
    image_target = BioImage(target_image_path)
    image_moving = BioImage(moving_image_path)

    if image_target.shape[:-3] != image_moving.shape[:-3]:
        logger.error(
            "The moving and target image must have the same non-spatial dimensions. "
            "Target shape: [ %s ], Moving shape: [ %s ]",
            image_target.shape[:-3],
            image_moving.shape[:-3],
        )
        raise ValueError("The moving and target image must have the same non-spatial dimensions.")

    aligned_files: dict[str, list[str]] = {"target": [], "moving": []}

    if align_fluo:
        aligned_files["target_fluo"] = []
        aligned_files["moving_fluo"] = []

    for scene in range(len(image_target.scenes)):
        image_target.set_scene(scene)
        image_moving.set_scene(scene)
        for t in range(image_target.dims["T"][0]):
            target_fluo = image_target.get_image_dask_data(
                "ZYX", C=FLUOR_CHANNEL, T=t, resolution=resolution_level, Z=target_z_slices
            ).compute()
            moving_fluo = image_moving.get_image_dask_data(
                "ZYX", C=FLUOR_CHANNEL, T=t, resolution=resolution_level, Z=moving_z_slices
            ).compute()

            if not align_fluo:
                target_fluo = target_fluo.std(0)
                moving_fluo = moving_fluo.std(0)

            # assume isotropic in xy
            img_moving_physical_size: PhysicalPixelSizes = image_moving.physical_pixel_sizes
            img_target_physical_size: PhysicalPixelSizes = image_target.physical_pixel_sizes
            # to deal with type error with .X being float or None
            if img_moving_physical_size.X is None or img_target_physical_size.X is None:
                logger.error(
                    "Physical pixel sizes for moving or target image are not set. "
                    "Moving size: [ %s ], Target size: [ %s ]",
                    img_moving_physical_size,
                    img_target_physical_size,
                )
                raise ValueError("Physical pixel sizes for moving or target image are not set.")
            # will only continue if the physical pixel sizes are set
            rescale_factor = img_target_physical_size.X / img_moving_physical_size.X
            logger.debug(
                "Rescale factor for scene [ %d ], time [ %d ]: [ %f ]", scene, t, rescale_factor
            )
            moving_fluo = resize_moving(moving_fluo, rescale_factor)

            target_projection = target_fluo.std(0) if align_fluo else target_fluo
            moving_projection = moving_fluo.std(0) if align_fluo else moving_fluo

            # apply alignment method
            # add type: ignore[arg-type] to avoid type errors with the kwargs
            if alignment_method == "sift":
                model = sift_registration(
                    target_projection,
                    moving_projection,
                    **alignment_kwargs,  # type: ignore[arg-type]
                )
            elif alignment_method == "template":
                model = template_registration(
                    target_projection,
                    moving_projection,
                    **alignment_kwargs,  # type: ignore[arg-type]
                )
            if model is None:
                continue

            target_bf = image_target.get_image_dask_data(
                "ZYX", C=BF_CHANNEL, T=t, Z=target_z_slices
            ).compute()
            moving_bf = image_moving.get_image_dask_data(
                "ZYX", C=BF_CHANNEL, T=t, Z=moving_z_slices
            ).compute()
            moving_bf = resize_moving(moving_bf, (1, rescale_factor, rescale_factor))

            base_moving_path = Path(moving_image_path).name.split(".")[0]
            base_target_path = Path(target_image_path).name.split(".")[0]

            if align_fluo:
                moving_fluo = warp(model, target_fluo, moving_fluo)
                moving_fluo, target_fluo = crop_to_overlap(moving_fluo, target_fluo)

                # Save the aligned images
                moving_save_path = (
                    savedir / f"{base_moving_path}_{scene}_{t}_moving_fluo.ome.tiff"
                ).as_posix()
                target_save_path = (
                    savedir / f"{base_target_path}_{scene}_{t}_target_fluo.ome.tiff"
                ).as_posix()
                OmeTiffWriter.save(uri=moving_save_path, data=moving_fluo)
                OmeTiffWriter.save(uri=target_save_path, data=target_fluo)

                save_overlay(
                    moving_fluo,
                    target_fluo,
                    savedir / f"{base_target_path}_{scene}_{t}_overlay.png",
                )
                aligned_files["target_fluo"].append(target_save_path)
                aligned_files["moving_fluo"].append(moving_save_path)

            aligned_moving = warp(model, target_bf, moving_bf)
            aligned_moving, target_bf = crop_to_overlap(aligned_moving, target_bf)
            # Save the aligned images
            moving_save_path = (
                savedir / f"{base_moving_path}_{scene}_{t}_moving_bf.ome.tiff"
            ).as_posix()
            target_save_path = (
                savedir / f"{base_target_path}_{scene}_{t}_target_bf.ome.tiff"
            ).as_posix()
            OmeTiffWriter.save(uri=moving_save_path, data=aligned_moving)
            OmeTiffWriter.save(uri=target_save_path, data=target_bf)
            aligned_files["moving"].append(moving_save_path)
            aligned_files["target"].append(target_save_path)
    return pd.DataFrame(aligned_files)


def align_all_positions(
    target_dataset_name: str,
    moving_dataset_name: str,
    resolution_level: int,
    z_slice_offsets: tuple[int, int] | None,
    savedir: Path,
    alignment_method: Literal["sift", "template"],
    align_fluo: bool = True,
    num_positions_to_align: int | None = None,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Align all positions of the moving dataset to the target dataset.

    **Alignment methods**

    Options for alignment methods are "sift" or "template", passed in via the ``alignment_method``
    input. The method "sift" is recommended for the 20X pre/post fixation datasets, while the
    method "template" is recommended for the 20X/40X datasets.

    Parameters
    ----------
    target_dataset_name
        The name of the target dataset.
    moving_dataset_name
        The name of the moving dataset.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    savedir
        The directory where the aligned images will be saved.
    alignment_method
        The method used for alignment.
    align_fluo
        If True, align the fluorescent channel. If False, do not align the fluorescent channel.
    num_positions_to_align
        Optional, the number of positions in the dataset to process for alignment.
    **alignment_kwargs
        Additional arguments for the alignment function.

    Returns
    -------
    data
        DataFrame containing the paths to the aligned images.
    """
    if alignment_method not in ["sift", "template"]:
        logger.error(
            "Invalid alignment method: [ %s ]. Choose 'sift' or 'template'.", alignment_method
        )
        raise ValueError("Invalid alignment method. Choose 'sift' or 'template'.")

    # get list of zarr files in each dataset
    moving_dataset_config = load_dataset_config(moving_dataset_name)
    target_dataset_config = load_dataset_config(target_dataset_name)
    moving_zarr_files = sorted(get_available_zarr_files(moving_dataset_config))
    target_zarr_files = sorted(get_available_zarr_files(target_dataset_config))

    # get image loading args for moving and target datasets
    moving_z_slice = get_z_slice_bounds_per_position(moving_dataset_config, z_slice_offsets)
    moving_include_pos = get_include_positions(moving_dataset_config)
    target_z_slice = get_z_slice_bounds_per_position(target_dataset_config, z_slice_offsets)
    target_include_pos = get_include_positions(target_dataset_config)

    data_list = []
    position_counter = 0
    for moving, target in zip(moving_zarr_files, target_zarr_files, strict=True):
        logger.debug(
            "Aligning moving image [ %s ] to target image [ %s ]",
            moving,
            target,
        )
        position = get_position_integer_from_zarr_file_path(moving)
        if position not in moving_include_pos or position not in target_include_pos:
            logger.warning(
                "Skipping position [ %d ] as it has been annotated for a known imaging artifact.",
                position,
            )
            continue
        moving_z_slices = list(
            range(moving_z_slice[position]["z_start"], moving_z_slice[position]["z_stop"] + 1)
        )
        target_z_slices = list(
            range(target_z_slice[position]["z_start"], target_z_slice[position]["z_stop"] + 1)
        )
        df_position = align(
            moving,
            target,
            moving_z_slices,
            target_z_slices,
            resolution_level,
            savedir,
            align_fluo=align_fluo,
            alignment_method=alignment_method,
            **alignment_kwargs,
        )
        data_list.append(df_position)
        position_counter += 1
        if num_positions_to_align is not None and position_counter >= num_positions_to_align:
            break
    data = pd.concat(data_list, ignore_index=True)
    return data


def _get_concat_path(row: dict[str, str], savedir: Path) -> Path:
    base_image_path = Path(row["target"]).name.split(".")[0]
    return savedir / f"{base_image_path.replace('_target', '_aligned_paired')}.ome.tiff"


def get_paired_dataset_dict(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
) -> dict[str, list[str]]:
    """
    Get a dictionary of paired datasets for alignment with correct
    'target' and 'moving' labels.

    Parameters
    ----------
    dataset_pair_type
        The type of dataset pair to align, either "live_fixed" or "20X_40X".

    Returns
    -------
    :
        Dictionary with keys "target" and "moving" containing lists of dataset names.
    """

    if dataset_pair_type not in ["live_fixed", "20X_40X"]:
        logger.error(
            "Invalid dataset pair type: [ %s ]. Choose 'live_fixed' or '20X_40X'.",
            dataset_pair_type,
        )
        raise ValueError("Invalid dataset pair type. Choose 'live_fixed' or '20X_40X'.")

    # Get the list of datasets of the specified pair type.
    dataset_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")

    # Set dataset name flags for setting
    # "target" and "moving" images for alignment.
    if dataset_pair_type == "live_fixed":
        # for live/fixed pairs, the "target" image
        # for alignment is the pre-fixation (live) image
        # and the "moving" image is the post-fixation (fixed) image.
        target_flag = "PreFixation"
        moving_flag = "PostFixation"
    else:
        # for 20X/40X pairs, the "target" image is the 20X image
        # and the "moving" image is the 40x image.
        target_flag = "20X"
        moving_flag = "40X"
    dataset_pairs = {
        "target": [dataset_name for dataset_name in dataset_list if target_flag in dataset_name],
        "moving": [dataset_name for dataset_name in dataset_list if moving_flag in dataset_name],
    }
    return dataset_pairs


def align_and_save_paired_images(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
    resolution_level: int,
    z_slice_offsets: tuple[int, int] | None,
    save_path: Path,
    num_datasets_to_align: int | None = None,
    num_positions_to_align: int | None = None,
) -> pd.DataFrame:
    """
    Align and save all paired images from the specified dataset pair type.

    **Z-stack offsets**

    The ``z_slice_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_slice_offsets`` is provided, it limits the number of z-slices to load,
    by slicing about a global center (annotated in the dataset config). If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    Parameters
    ----------
    dataset_pair_type
        The type of dataset pair to align.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    save_path
        The directory where the aligned images will be saved.
    num_datasets_to_align
        The number of datasets to process for alignment.
        Use None to align all datasets.
    num_positions_to_align
        The number of positions in the dataset to process for alignment.
        Use None to align all positions.

    Returns
    -------
    :
        DataFrame containing the paths to the aligned images.
    """

    dataset_pairs = get_paired_dataset_dict(dataset_pair_type)

    # Note that the "target" key refers to the image being used as
    # the reference image for alignment, and the "moving" key
    # refers to the image being aligned to the target image.
    target_datasets = dataset_pairs["target"]
    moving_datasets = dataset_pairs["moving"]

    if dataset_pair_type == "live_fixed":
        alignment_method: Literal["sift", "template"] = "sift"
    else:
        alignment_method = "template"

    df_list = []

    for index, (target, moving) in enumerate(zip(target_datasets, moving_datasets, strict=True)):
        if num_datasets_to_align is not None and index > num_datasets_to_align:
            break

        df_ = align_all_positions(
            target,
            moving,
            resolution_level,
            z_slice_offsets,
            save_path,
            alignment_method=alignment_method,
            num_positions_to_align=num_positions_to_align,
        )
        df_["target_dataset"] = target
        df_list.append(df_)

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["target", "moving"])
    logger.debug("Found %d pairs of images to save.", len(df))
    return df


def concat_and_save_aligned_image_pairs(
    row: dict[str, str], savedir: Path, overwrite_images: bool = True
) -> Path:
    """
    Concatenate the aligned target and moving images into a single OME-TIFF file
    and save it to the specified directory.

    Parameters
    ----------
    row
        A row (in dict form) of a DataFrame containing paths to the target and moving images.
    savedir
        The directory where the concatenated image will be saved.
    overwrite_images
        Overwrite existing images if True, return existing file if False.
    """
    save_path = _get_concat_path(row, savedir)
    if save_path.exists() and not overwrite_images:
        logger.debug("Returning existing file at: [ %s ]", save_path)
        return save_path

    # load the aligned brightfield images (squeeze out C and T dims)
    target_3d_stack = load_image_from_path(Path(row["target"]), squeeze=True)
    moving_3d_stack = load_image_from_path(Path(row["moving"]), squeeze=True)

    # take the std projection of each 3D stack over Z
    target_proj = target_3d_stack.std(0)
    moving_proj = moving_3d_stack.std(0)

    # concatenate along a new axis
    concatenated_images = np.stack([target_proj, moving_proj], axis=0)[:, None]

    # save the concatenated image as a multi-channel OME-TIFF
    OmeTiffWriter.save(uri=save_path, data=concatenated_images)
    logger.debug("Saving concatenated image to: [ %s ]", save_path)

    return save_path
