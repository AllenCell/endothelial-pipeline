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
from endo_pipeline.library.model import get_include_positions, get_z_offset_information
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
    Register a small moving image to a larger fixed image via sliding window correlation.

    Note that the moving image is assumed to be smaller than the fixed image.

    Parameters
    ----------
    image
        The fixed image used for registration.
    template
        The moving image that will be registered to the fixed image.
    scale
        The scale factor for downsampling the images before registration.
    """
    # Resize image to current scale
    downsampled_image = block_reduce(image, (scale, scale), np.max)
    downsampled_template = block_reduce(template, (scale, scale), np.max)

    downsampled_image = preprocess(downsampled_image).astype(np.float32)
    downsampled_template = preprocess(downsampled_template).astype(np.float32)

    # Ensure resized fixed image is larger than resized moving
    if (
        downsampled_image.shape[0] < downsampled_template.shape[0]
        or downsampled_image.shape[1] < downsampled_template.shape[1]
    ):
        raise ValueError(
            "Fixed image is smaller than moving image. Resized fixed shape:",
            f"{downsampled_image.shape}, moving shape: {downsampled_template.shape}",
        )
    # Perform template matching
    result = cv2.matchTemplate(downsampled_image, downsampled_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    print(f"Best score: {max_val}, Location: {max_loc}")
    return np.array(max_loc) * scale, max_val


def template_registration(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    scale: int = 3,
    template_shape: None | Sequence[int] = None,
) -> tf.SimilarityTransform:
    """
    Register a moving image to a fixed image using template matching.

    Parameters
    ----------
    image_fixed
        The reference image used for registration.
    image_moving
        The image that will be registered to the fixed image.
    scale
        The scale factor for downsampling the images before registration.
    template_shape
        Optional, the shape of the template used for registration.
    """
    template_shape = template_shape or image_moving.shape[-2:]
    splitter = SlidingWindowSplitter(patch_size=template_shape, overlap=0.1, pad_mode=None)
    # Register each template to the fixed image
    best_transform = None
    best_score = 0.0
    for template in tqdm(splitter(torch.from_numpy(image_moving[None, None]))):
        transform, score = template_matching(
            image_fixed, template[0].numpy().squeeze(), scale=scale
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
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    min_samples: int = 4,
    residual_threshold: int = 10,
    max_trials: int = 1000,
    visualize_keypoints_dir: str | None = None,
) -> tf.SimilarityTransform | None:
    """
    Register a moving image to a fixed image using SIFT keypoint matching and RANSAC.

    Parameters
    ----------
    image_fixed
        The reference image used for registration.
    image_moving
        The image that will be registered to the fixed image.
    min_samples
        Minimum number of samples for RANSAC.
    residual_threshold
        Max distance for RANSAC inliers. Adjust based on image resolution/expected error.
    max_trials
        Maximum RANSAC iterations.
    visualize_keypoints_dir
        Directory to save visualizations of keypoints. If None, no visualizations are saved.
    """
    image_fixed = sift_preprocess(image_fixed)
    image_moving = sift_preprocess(image_moving)

    keypoints_fixed, descriptors_fixed = _get_sift(image_fixed)
    keypoints_moving, descriptors_moving = _get_sift(image_moving)

    # brute force matching
    matches = match_descriptors(descriptors_moving, descriptors_fixed)
    logger.debug("Matches found: [ %d ]", len(matches))

    if len(matches) < min_samples:
        logger.warning("Not enough matches found for RANSAC after cross-checking.")
        return None

    # Prepare data for RANSAC
    src = keypoints_moving[matches[:, 0]][:, ::-1]  # RANSAC expects (x, y)
    dst = keypoints_fixed[matches[:, 1]][:, ::-1]  # RANSAC expects (x, y)

    if visualize_keypoints_dir:
        visualize_keypoints(image_moving, src[:, ::-1], "moving_matched")
        visualize_keypoints(image_fixed, dst[:, ::-1], "fixed_matched")

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
    model: tf.ProjectiveTransform, image_fixed: np.ndarray, image_moving: np.ndarray
) -> np.ndarray:
    """
    Warp the moving image to align with the fixed image using the provided transformation model.

    Parameters
    ----------
    model
        The transformation model.
    image_fixed
        The fixed image to which the moving image will be aligned.
    image_moving
        The moving image to be warped.

    Returns
    -------
    :
        The warped moving image aligned to the fixed image.
    """
    print("Warping image...")
    warp_transform = partial(
        tf.warp,
        inverse_map=model.inverse,
        output_shape=image_fixed.shape[-2:],
        order=3,
        mode="constant",
        cval=np.nan,
    )
    is_3d = len(image_fixed.shape) == 3
    if is_3d:
        aligned_moving = np.stack(
            [warp_transform(image_moving[i]) for i in trange(len(image_moving))]
        )
    else:
        aligned_moving = warp_transform(image_moving)
    return aligned_moving


def resize_moving(image_moving: np.ndarray, resize_factor: float | Sequence[float]) -> np.ndarray:
    """
    Resize the moving image to match the fixed image dimensions.

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


def save_overlay(moving: np.ndarray, fixed: np.ndarray, savepath: str | Path) -> None:
    """Save overlay of aligned moving and fixed fluorescent images."""
    moving_proj = overlay_normalize(moving)
    fixed_proj = overlay_normalize(fixed)
    overlay = np.stack([moving_proj, fixed_proj, fixed_proj], axis=-1)
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(savepath, dpi=300)
    plt.close()


def align(
    moving_image_path: str | Path,
    fixed_image_path: str | Path,
    moving_z_slices: list[int],
    fixed_z_slices: list[int],
    resolution_level: int,
    savedir: Path,
    alignment_method: Literal["sift", "template"],
    align_fluo: bool = True,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Align a moving image to a fixed image using blob detection and registration.

    **Alignment methods**

    Options for alignment methods are "sift" or "template", passed in via the ``alignment_method``
    input. The method "sift" is recommended for the 20X pre/post fixation datasets, while the
    method "template" is recommended for the 20X/40X datasets.

    Parameters
    ----------
    moving_image_path
        Path to the moving image.
    fixed_image_path
        Path to the fixed image.
    moving_z_slices
        List of z-slices to load from the moving image.
    fixed_z_slices
        List of z-slices to load from the fixed image.
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

    # load images
    image_fixed = BioImage(fixed_image_path)
    image_moving = BioImage(moving_image_path)

    if image_fixed.shape[:-3] != image_moving.shape[:-3]:
        logger.error(
            "The moving and fixed image must have the same non-spatial dimensions. "
            "Fixed shape: [ %s ], Moving shape: [ %s ]",
            image_fixed.shape[:-3],
            image_moving.shape[:-3],
        )
        raise ValueError("The moving and fixed image must have the same non-spatial dimensions.")

    aligned_files: dict[str, list[str]] = {"fixed": [], "moving": []}

    if align_fluo:
        aligned_files["fixed_fluo"] = []
        aligned_files["moving_fluo"] = []

    for scene in range(len(image_fixed.scenes)):
        image_fixed.set_scene(scene)
        image_moving.set_scene(scene)
        for t in range(image_fixed.dims["T"][0]):
            fixed_fluo = image_fixed.get_image_dask_data(
                "ZYX", C=FLUOR_CHANNEL, T=t, resolution=resolution_level, Z=fixed_z_slices
            ).compute()
            moving_fluo = image_moving.get_image_dask_data(
                "ZYX", C=FLUOR_CHANNEL, T=t, resolution=resolution_level, Z=moving_z_slices
            ).compute()

            if not align_fluo:
                fixed_fluo = fixed_fluo.std(0)
                moving_fluo = moving_fluo.std(0)

            # assume isotropic in xy
            img_moving_physical_size: PhysicalPixelSizes = image_moving.physical_pixel_sizes
            img_fixed_physical_size: PhysicalPixelSizes = image_fixed.physical_pixel_sizes
            # to deal with type error with .X being float or None
            if img_moving_physical_size.X is None or img_fixed_physical_size.X is None:
                logger.error(
                    "Physical pixel sizes for moving or fixed image are not set. "
                    "Moving size: [ %s ], Fixed size: [ %s ]",
                    img_moving_physical_size,
                    img_fixed_physical_size,
                )
                raise ValueError("Physical pixel sizes for moving or fixed image are not set.")
            # will only continue if the physical pixel sizes are set
            rescale_factor = img_fixed_physical_size.X / img_moving_physical_size.X
            logger.debug(
                "Rescale factor for scene [ %d ], time [ %d ]: [ %f ]", scene, t, rescale_factor
            )
            moving_fluo = resize_moving(moving_fluo, rescale_factor)

            fixed_projection = fixed_fluo.std(0) if align_fluo else fixed_fluo
            moving_projection = moving_fluo.std(0) if align_fluo else moving_fluo

            # apply alignment method
            # add type: ignore[arg-type] to avoid type errors with the kwargs
            if alignment_method == "sift":
                model = sift_registration(
                    fixed_projection,
                    moving_projection,
                    **alignment_kwargs,  # type: ignore[arg-type]
                )
            elif alignment_method == "template":
                model = template_registration(
                    fixed_projection,
                    moving_projection,
                    **alignment_kwargs,  # type: ignore[arg-type]
                )
            if model is None:
                continue

            fixed_bf = image_fixed.get_image_dask_data(
                "ZYX", C=BF_CHANNEL, T=t, Z=fixed_z_slices
            ).compute()
            moving_bf = image_moving.get_image_dask_data(
                "ZYX", C=BF_CHANNEL, T=t, Z=moving_z_slices
            ).compute()
            moving_bf = resize_moving(moving_bf, (1, rescale_factor, rescale_factor))

            base_moving_path = Path(moving_image_path).name.split(".")[0]
            base_fixed_path = Path(fixed_image_path).name.split(".")[0]

            if align_fluo:
                moving_fluo = warp(model, fixed_fluo, moving_fluo)
                moving_fluo, fixed_fluo = crop_to_overlap(moving_fluo, fixed_fluo)

                # Save the aligned images
                moving_save_path = (
                    savedir / f"{base_moving_path}_{scene}_{t}_moving_fluo.ome.tiff"
                ).as_posix()
                fixed_save_path = (
                    savedir / f"{base_fixed_path}_{scene}_{t}_fixed_fluo.ome.tiff"
                ).as_posix()
                OmeTiffWriter.save(uri=moving_save_path, data=moving_fluo)
                OmeTiffWriter.save(uri=fixed_save_path, data=fixed_fluo)

                save_overlay(
                    moving_fluo,
                    fixed_fluo,
                    savedir / f"{base_fixed_path}_{scene}_{t}_overlay.png",
                )
                aligned_files["fixed_fluo"].append(fixed_save_path)
                aligned_files["moving_fluo"].append(moving_save_path)

            aligned_moving = warp(model, fixed_bf, moving_bf)
            aligned_moving, fixed_bf = crop_to_overlap(aligned_moving, fixed_bf)
            # Save the aligned images
            moving_save_path = (
                savedir / f"{base_moving_path}_{scene}_{t}_moving_bf.ome.tiff"
            ).as_posix()
            fixed_save_path = (
                savedir / f"{base_fixed_path}_{scene}_{t}_fixed_bf.ome.tiff"
            ).as_posix()
            OmeTiffWriter.save(uri=moving_save_path, data=aligned_moving)
            OmeTiffWriter.save(uri=fixed_save_path, data=fixed_bf)
            aligned_files["moving"].append(moving_save_path)
            aligned_files["fixed"].append(fixed_save_path)
    return pd.DataFrame(aligned_files)


def align_all_positions(
    fixed_dataset_name: str,
    moving_dataset_name: str,
    resolution_level: int,
    z_stack_offsets: tuple[int, int] | None,
    slice_by_global_center: bool,
    savedir: Path,
    alignment_method: Literal["sift", "template"],
    align_fluo: bool = True,
    num_positions_to_align: int | None = None,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Align all positions of the moving dataset to the fixed dataset.

    **Alignment methods**

    Options for alignment methods are "sift" or "template", passed in via the ``alignment_method``
    input. The method "sift" is recommended for the 20X pre/post fixation datasets, while the
    method "template" is recommended for the 20X/40X datasets.

    Parameters
    ----------
    fixed_dataset_name
        The name of the fixed dataset.
    moving_dataset_name
        The name of the moving dataset.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center
        Get global center plane per position for z-slicing if True, use offsets directly if False.
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
    fixed_dataset_config = load_dataset_config(fixed_dataset_name)
    moving_zarr_files = sorted(get_available_zarr_files(moving_dataset_config))
    fixed_zarr_files = sorted(get_available_zarr_files(fixed_dataset_config))

    # get image loading args for moving and fixed datasets
    moving_z_slice = get_z_offset_information(
        moving_dataset_config, z_stack_offsets, slice_by_global_center
    )
    moving_include_pos = get_include_positions(moving_dataset_config)
    fixed_z_slice = get_z_offset_information(
        fixed_dataset_config, z_stack_offsets, slice_by_global_center
    )
    fixed_include_pos = get_include_positions(fixed_dataset_config)

    data_list = []
    position_counter = 0
    for moving, fixed in zip(moving_zarr_files, fixed_zarr_files, strict=True):
        logger.debug(
            "Aligning moving image [ %s ] to fixed image [ %s ]",
            moving,
            fixed,
        )
        position = get_position_integer_from_zarr_file_path(moving)
        if position not in moving_include_pos or position not in fixed_include_pos:
            logger.warning(
                "Skipping position [ %d ] as it has been annotated for a known imaging artifact.",
                position,
            )
            continue
        moving_z_slices = list(
            range(moving_z_slice[position]["z_start"], moving_z_slice[position]["z_stop"] + 1)
        )
        fixed_z_slices = list(
            range(fixed_z_slice[position]["z_start"], fixed_z_slice[position]["z_stop"] + 1)
        )
        data_list.append(
            align(
                moving,
                fixed,
                moving_z_slices,
                fixed_z_slices,
                resolution_level,
                savedir,
                align_fluo=align_fluo,
                alignment_method=alignment_method,
                **alignment_kwargs,
            )
        )
        position_counter += 1
        if num_positions_to_align is not None and position_counter >= num_positions_to_align:
            break
    data = pd.concat(data_list, ignore_index=True)
    return data


def _get_concat_path(row: dict[str, str], savedir: Path) -> Path:
    base_image_path = Path(row["fixed"]).name.split(".")[0]
    return savedir / f"{base_image_path.replace('_fixed', '')}.ome.tiff"


def _get_paired_dataset_dict(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
) -> dict[str, list[str]]:

    # Get the list of datasets of the specified pair type.
    dataset_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")

    # Set dataset name flags for setting
    # "fixed" and "moving" images for alignment.
    if dataset_pair_type == "live_fixed":
        # for live/fixed pairs, the "fixed" image
        # for alignment is the pre-fixation (live) image
        # and the "moving" image is the post-fixation (fixed) image.
        fixed_flag = "PreFixation"
        moving_flag = "PostFixation"
    else:
        # for 20X/40X pairs, the "fixed" image is the 20X image
        # and the "moving" image is the 40x image.
        fixed_flag = "20X"
        moving_flag = "40X"
    dataset_pairs = {
        "fixed": [dataset_name for dataset_name in dataset_list if fixed_flag in dataset_name],
        "moving": [dataset_name for dataset_name in dataset_list if moving_flag in dataset_name],
    }
    return dataset_pairs


def align_and_save_paired_images(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
    resolution_level: int,
    z_stack_offsets: tuple[int, int] | None,
    slice_by_global_center: bool,
    save_path: Path,
    testing_mode: bool = False,
) -> pd.DataFrame:
    """
    Align and save all paired images from the specified dataset pair type.

    **Z-stack offsets**

    The ``z_stack_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_stack_offsets`` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    If ``slice_by_global_center`` is set to True, the z-slice range is calculated based on
    the global center plane for the given position. In this case, ``z_stack_offsets`` should
    indicate the number of slices to include below and above the center plane. Else, the
    ``z_stack_offsets`` are used directly as the range bounds.

    **Workflow testing**

    If ``testing_mode`` is set to True, the function will only align the first pair of images
    and save them to the specified `save_path`. This is useful for testing the basic function
    of this method without processing the entire dataset.

    Parameters
    ----------
    dataset_pair_type
        The type of dataset pair to align.
    resolution_level
        The resolution level of the zarr files to load for alignment.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center
        Get global center plane per position for z-slicing if True, use offsets directly if False.
    save_path
        The directory where the aligned images will be saved.
    testing_mode
        If True, only the first pair of images will be aligned and saved.


    Returns
    -------
    :
        DataFrame containing the paths to the aligned images.
    """

    dataset_pairs = _get_paired_dataset_dict(dataset_pair_type)

    # Note that the "fixed" key refers to the image being used as
    # the reference image for alignment, and the "moving" key
    # refers to the image being aligned to the fixed image.
    # That is, "fixed" here does not refer to the image being fixed.
    fixed_datasets = dataset_pairs["fixed"]
    moving_datasets = dataset_pairs["moving"]

    if dataset_pair_type == "live_fixed":
        alignment_method: Literal["sift", "template"] = "sift"
    else:
        alignment_method = "template"

    df_list = []
    for fixed, moving in zip(fixed_datasets, moving_datasets, strict=True):
        if testing_mode:
            logger.warning(
                "Testing mode is enabled. Only the first two pairs of images "
                "from the first dataset pair will be aligned and saved."
            )

        df_list.append(
            align_all_positions(
                fixed,
                moving,
                resolution_level,
                z_stack_offsets,
                slice_by_global_center,
                save_path,
                alignment_method=alignment_method,
                num_positions_to_align=4 if testing_mode else None,
            )
        )

        if testing_mode:
            break

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["fixed", "moving"])
    logger.debug("Found %d pairs of images to save.", len(df))
    return df


def concat_and_save_aligned_image_pairs(row: dict[str, str], savedir: Path) -> Path:
    """
    Concatenate the aligned fixed and moving images into a single OME-TIFF file
    and save it to the specified directory.

    Parameters
    ----------
    row
        A row (in dict form) of a DataFrame containing paths to the fixed and moving images.
    savedir
        The directory where the concatenated image will be saved.

    Returns
    -------
    :
        The path to the saved concatenated image.
    """
    save_path = _get_concat_path(row, savedir)
    if save_path.exists():
        logger.debug("Returning existing file at: [ %s ]", save_path)
        return save_path
    # take standard deviation projection here to allow concatenation with different z-axis sizes
    fixed = BioImage(row["fixed"]).data.squeeze().max(0)
    moving = BioImage(row["moving"]).data.squeeze().max(0)

    out = np.stack([fixed, moving], axis=0)[:, None]

    OmeTiffWriter.save(uri=save_path, data=out)
    logger.debug("Saving concatenated image to: [ %s ]", save_path)
    return save_path
