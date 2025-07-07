from functools import partial
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from cyto_dl.api import CytoDLModel
from monai.inferers import SlidingWindowSplitter
from skimage import transform as tf
from skimage.exposure import rescale_intensity
from skimage.feature import SIFT, match_descriptors
from skimage.measure import block_reduce, ransac
from tqdm import tqdm, trange

from cellsmap.util.manifest_io import load_pca_model
from cellsmap.util.manifest_preprocessing import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import load_model_config
from src.endo_pipeline.configs.dataset_io import get_zarr_path, update_dataset_config
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import project_manifest_to_pcs
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.process.cdh5_preprocessing import preprocess
from src.endo_pipeline.workflows.apply_diffae_model import generate_overrides

FLUOR_CHANNEL = 0
BF_CHANNEL = 1


def visualize_keypoints(image: np.ndarray, keypoints: np.ndarray, savepath: str) -> None:
    """
    Visualizes the detected keypoints on the image.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    keypoints : np.ndarray
        The coordinates of the detected keypoints.
    """
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:, 1], keypoints[:, 0], s=0.1, c="red", marker="o")
    plt.axis("off")
    plt.savefig(savepath, dpi=300)
    plt.close()


def sift_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for SIFT feature detection with percentile clipping and 0-1 normalization
    Parameters
    ----------
    img : np.ndarray
        The input image.
    Returns
    -------
    img : np.ndarray
        The preprocessed image.
    """
    img = np.clip(img, np.percentile(img, 10), np.percentile(img, 99))
    img = (img - img.min()) / (img.max() - img.min())
    return img


def template_matching(
    image: np.ndarray, template: np.ndarray, scale: int = 3
) -> tuple[np.ndarray, float]:
    """
    Register a small moving image to a larger fixed image using a multi-scale sliding window correlation. NOTE that the moving image is assumed to be smaller than the fixed image.
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
            f"Fixed image is smaller than moving image. Resized fixed shape: {downsampled_image.shape}, moving shape: {downsampled_template.shape}"
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
    Registers a moving image to a fixed image using template matching
    Parameters
    ----------
    image_fixed : np.ndarray
        The reference image used for registration.
    image_moving : np.ndarray
        The image that will be registered to the fixed image.
    scale : int
        The scale factor for downsampling the images before registration.
    template_shape : tuple
        The shape of the template used for registration. If None, the shape of the moving image will be used.
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
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Detects SIFT keypoints and descriptors in the given image.
    Parameters
    ----------
    image : np.ndarray
        The input image.
    upsampling : int
        The number of times to upsample the image before detecting keypoints.
    sigma_min : int
        The minimum standard deviation for Gaussian smoothing.
    Returns
    -------
    keypoints : np.ndarray
        The detected keypoints.
    descriptors : np.ndarray
        The descriptors for the detected keypoints.
    """
    extractor = SIFT(upsampling=upsampling, sigma_min=sigma_min)
    extractor.detect_and_extract(image)
    return extractor.keypoints, extractor.descriptors


def sift_registration(
    image_fixed: np.ndarray,
    image_moving: np.ndarray,
    min_samples: int = 4,
    residual_threshold: int = 10,
    max_trials: int = 1000,
    visualize_keypoints_dir: str | None = None,
) -> tf.SimilarityTransform | None:
    """
    Registers a moving image to a fixed image using SIFT keypoint matching and RANSAC. Returns a similarity transform if successful, otherwise None.
    Parameters
    ----------
    image_fixed : np.ndarray
        The reference image used for registration.
    image_moving : np.ndarray
        The image that will be registered to the fixed image.
    min_samples : int
        Minimum number of samples for RANSAC.
    residual_threshold : int
        Max distance for RANSAC inliers. Adjust based on image resolution/expected error.
    max_trials : int
        Maximum RANSAC iterations.
    visualize_keypoints : bool
        Whether to visualize the detected keypoints.
    """
    image_fixed = sift_preprocess(image_fixed)
    image_moving = sift_preprocess(image_moving)

    keypoints_fixed, descriptors_fixed = _get_sift(image_fixed)
    keypoints_moving, descriptors_moving = _get_sift(image_moving)

    # brute force matching
    matches = match_descriptors(descriptors_moving, descriptors_fixed)
    print(f"Matches found: {len(matches)}")

    if len(matches) < min_samples:
        print("Error: Not enough matches found after cross-checking.")
        return None

    # Prepare data for RANSAC
    src = keypoints_moving[matches[:, 0]][:, ::-1]  # RANSAC expects (x, y)
    dst = keypoints_fixed[matches[:, 1]][:, ::-1]  # RANSAC expects (x, y)

    if visualize_keypoints_dir:
        visualize_keypoints(image_moving, src[:, ::-1], "moving_matched")
        visualize_keypoints(image_fixed, dst[:, ::-1], "fixed_matched")

    print("Estimating transformation using RANSAC...")
    try:
        model_robust, inliers = ransac(
            (src, dst),
            tf.SimilarityTransform,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )

        if model_robust is None or sum(inliers) < 1:
            print(
                f"RANSAC failed to find a robust model. Inliers: {sum(inliers) if inliers is not None else 0}"
            )
            return None

        print(f"RANSAC successful. Inliers: {sum(inliers)}/{len(matches)}")
        print(model_robust.params)

        model_robust.params[:2, :2] = np.eye(2)  # Set rotation and scaling to identity
        print("Estimated Transform:")
        print(model_robust.params)

    except Exception as e:
        print(f"Error during RANSAC transformation estimation: {e}")
        return None
    return model_robust


def warp(
    model: tf.ProjectiveTransform, image_fixed: np.ndarray, image_moving: np.ndarray
) -> np.ndarray:
    """
    Warps the moving image to align with the fixed image using the provided transformation model.
    Parameters
    ----------
    model (skimage.transform.SimilarityTransform): The transformation model.
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
    Resizes the moving image to match the fixed image dimensions.

    Args:
        image_moving (np.ndarray): The moving image to be resized.
        resize_factor (float): The factor by which to resize the moving image.

    Returns:
        np.ndarray: Resized moving image.
    """
    if np.all(resize_factor == 1.0):
        return image_moving
    resized_image_moving = tf.rescale(image_moving, resize_factor, order=3)
    return resized_image_moving


def crop_to_overlap(crop1: np.ndarray, crop2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN values present in the XY border of either of the passed images. It is assumed that
    the XY locations of the NaN values are the same across all Z slices if the images are 3D.
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
    """
    Project and normalize the image for overlay visualization.
    """
    proj = img.max(0)
    proj = np.clip(proj, np.percentile(proj, 10), np.percentile(proj, 99))
    proj = rescale_intensity(proj, out_range="uint8")
    return proj


def save_overlay(moving: np.ndarray, fixed: np.ndarray, savepath: str | Path) -> None:
    """
    Save overlay of aligned moving and fixed fluorescent images.
    """
    moving_proj = overlay_normalize(moving)
    fixed_proj = overlay_normalize(fixed)
    overlay = np.stack([moving_proj, fixed_proj, fixed_proj], axis=-1)
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(savepath, dpi=300)
    plt.close()


def crop_to_overlap(crop1: np.ndarray, crop2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN values present in the XY border of either of the passed images. It is assumed that
    the XY locations of the NaN values are the same across all Z slices if the images are 3D.
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


def align(
    moving_image_path: str | Path,
    fixed_image_path: str | Path,
    savedir: Path,
    alignment_method: str,
    align_fluo: bool = True,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Aligns a moving image to a fixed image using blob detection and registration.

    Parameters
    ----------
    moving_image_path (str):
        Path to the moving image.
    fixed_image_path (str):
        Path to the fixed image.
    savedir (Path):
        Directory to save the aligned images.
    alignment_method (str):
        The method used for alignment. Options are "sift" or "template". "sift" is recommended for the 20x pre/post fixation datasets, while "template" is recommended for the 20x/40x datasets.
    align_fluo (bool):
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    **alignment_kwargs (Dict[str, Any]):
        Additional arguments for the alignment function.

    Returns:
    --------
    pd.DataFrame: DataFrame containing the paths to the aligned images.
    """
    print(f"Registering {moving_image_path} to {fixed_image_path} using {alignment_method}")
    image_fixed = BioImage(fixed_image_path)
    image_moving = BioImage(moving_image_path)

    if image_fixed.shape[:-3] != image_moving.shape[:-3]:
        raise ValueError("The moving and fixed image must have the same non-spatial dimensions.")

    alignment_func = {
        "sift": sift_registration,
        "template": template_registration,
    }[alignment_method]

    aligned_files: dict[str, list[str]] = {"fixed": [], "moving": []}

    if align_fluo:
        aligned_files["fixed_fluo"] = []
        aligned_files["moving_fluo"] = []

    for scene in range(len(image_fixed.scenes)):
        image_fixed.set_scene(scene)
        image_moving.set_scene(scene)
        for t in range(image_fixed.dims["T"][0]):
            fixed_fluo = image_fixed.get_image_dask_data("ZYX", C=FLUOR_CHANNEL, T=t).compute()
            moving_fluo = image_moving.get_image_dask_data("ZYX", C=FLUOR_CHANNEL, T=t).compute()

            if not align_fluo:
                fixed_fluo = fixed_fluo.std(0)
                moving_fluo = moving_fluo.std(0)

            # assume isotropic in xy
            rescale_factor: float = (
                image_moving.physical_pixel_sizes.X / image_fixed.physical_pixel_sizes.X
            )
            print(f"Rescale factor: {rescale_factor}")
            moving_fluo = resize_moving(moving_fluo, rescale_factor)

            fixed_projection = fixed_fluo.std(0) if align_fluo else fixed_fluo
            moving_projection = moving_fluo.std(0) if align_fluo else moving_fluo

            model = alignment_func(fixed_projection, moving_projection, **alignment_kwargs)
            if model is None:
                continue

            fixed_bf = image_fixed.get_image_dask_data("ZYX", C=BF_CHANNEL, T=t).compute()
            moving_bf = image_moving.get_image_dask_data("ZYX", C=BF_CHANNEL, T=t).compute()
            moving_bf = resize_moving(moving_bf, (1, rescale_factor, rescale_factor))

            if align_fluo:
                moving_fluo = warp(model, fixed_fluo, moving_fluo)
                moving_fluo, fixed_fluo = crop_to_overlap(moving_fluo, fixed_fluo)

                # Save the aligned images
                moving_save_path = str(
                    savedir / f"{Path(moving_image_path).stem}_{scene}_{t}_moving_fluo.ome.tiff"
                )
                fixed_save_path = str(
                    savedir / f"{Path(fixed_image_path).stem}_{scene}_{t}_fixed_fluo.ome.tiff"
                )
                OmeTiffWriter.save(uri=moving_save_path, data=moving_fluo)
                OmeTiffWriter.save(uri=fixed_save_path, data=fixed_fluo)

                save_overlay(
                    moving_fluo,
                    fixed_fluo,
                    savedir / f"{Path(fixed_image_path).stem}_{scene}_{t}_overlay.png",
                )
                aligned_files["fixed_fluo"].append(fixed_save_path)
                aligned_files["moving_fluo"].append(moving_save_path)

            aligned_moving = warp(model, fixed_bf, moving_bf)
            aligned_moving, fixed_bf = crop_to_overlap(aligned_moving, fixed_bf)
            # Save the aligned images
            moving_save_path = str(
                savedir / f"{Path(moving_image_path).stem}_{scene}_{t}_moving_bf.ome.tiff"
            )
            fixed_save_path = str(
                savedir / f"{Path(fixed_image_path).stem}_{scene}_{t}_fixed_bf.ome.tiff"
            )
            OmeTiffWriter.save(uri=moving_save_path, data=aligned_moving)
            OmeTiffWriter.save(uri=fixed_save_path, data=fixed_bf)
            aligned_files["moving"].append(moving_save_path)
            aligned_files["fixed"].append(fixed_save_path)
    return pd.DataFrame(aligned_files)


def align_all_positions(
    fixed_dataset_name: str,
    moving_dataset_name: str,
    savedir: Path,
    alignment_method: str,
    align_fluo: bool = True,
    **alignment_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Aligns all positions of the moving dataset to the fixed dataset.

    Parameters
    ----------
    fixed_dataset_name : str
        The name of the fixed dataset.
    moving_dataset_name : str
        The name of the moving dataset.
    savedir : Path
        The directory where the aligned images will be saved.
    alignment_method : str
        The method used for alignment. Options are "sift" or "template". "sift" is recommended for the 20x pre/post fixation datasets, while "template" is recommended for the 20x/40x datasets.
    align_fluo : bool
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    **alignment_kwargs : Dict[str, Any]
        Additional arguments for the alignment function.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the paths to the aligned images.
    """
    savedir.mkdir(parents=True, exist_ok=True)
    moving_zarr_files = sorted(get_zarr_path(moving_dataset_name).values())
    fixed_zarr_files = sorted(get_zarr_path(fixed_dataset_name).values())
    data = pd.concat(
        [
            align(
                moving,
                fixed,
                savedir,
                align_fluo=align_fluo,
                alignment_method=alignment_method,
                **alignment_kwargs,
            )
            for moving, fixed in zip(moving_zarr_files, fixed_zarr_files, strict=True)
        ]
    )
    return data


def plot_paired_features(
    fixed_features: pd.DataFrame,
    fixed_name: str,
    moving_features: pd.DataFrame,
    moving_name: str,
    save_path: Path,
    pca_dir: None | str | Path,
) -> None:
    """
    Plot the PCA features of the fixed and moving images
    """
    pca = load_pca_model(str(pca_dir)) if pca_dir else fit_pca()

    fixed_features = project_manifest_to_pcs(fixed_features, pca, overwrite_feature_columns=False)
    moving_features = project_manifest_to_pcs(moving_features, pca, overwrite_feature_columns=False)

    n_pcs = len([c for c in fixed_features.columns if c.startswith("pc")])

    fig, ax = plt.subplots(1, n_pcs, figsize=(n_pcs * 4, 4))
    for i in range(n_pcs):
        r = np.corrcoef(fixed_features[f"pc{i+1}"], moving_features[f"pc{i+1}"])[0, 1]
        ax[i].scatter(fixed_features[f"pc{i+1}"], moving_features[f"pc{i+1}"], alpha=0.1, s=3)
        ax[i].set_xlabel(fixed_name)
        ax[i].set_ylabel(moving_name)
        ax[i].set_title(f"PC{i+1} r^2: {r**2:.2f}", fontsize=6)
        min_ = min(fixed_features[f"pc{i+1}"].min(), moving_features[f"pc{i+1}"].min())
        max_ = max(fixed_features[f"pc{i+1}"].max(), moving_features[f"pc{i+1}"].max())
        ax[i].plot([min_, max_], [min_, max_], "r--")
        ax[i].set_xlim(min_, max_)
        ax[i].set_ylim(min_, max_)
        ax[i].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(save_path / "paired_features.png", dpi=300)
    fig.clf()
    plt.close(fig)


def add_fmsid_to_config(
    prediction_path: str, dataset_name: str, mlflow_id: str, model_path: Path
) -> None:
    """
    Upload path to FMS and add the FMS ID to the dataset config file for the given dataset.

    Parameters
    ----------
    prediction_path : str
        Path to the prediction file.
    dataset_name : str
        Name of the dataset to update in config
    mlflow_id : str
        MLflow ID of the model used for prediction.
    model_path : Path
        Path to the model directory. Used for extracting the commit hash.
    """
    file_id = save_file_to_fms(
        prediction_path,
        dataset_name,
        get_cytodl_commit_hash(mlflow_id, model_path),
        misc_notes="",
        mlflow_run_id=mlflow_id,
    )

    update_dataset_config(
        dataset_name,
        {"diffae_manifest_fmsid": file_id},
    )


def compare_paired_features(
    model_name: str,
    fixed_dataset_name: str,
    moving_dataset_name: str,
    alignment_method: str,
    pca_dir: str | None,
    align_fluo: bool = True,
    align_only: bool = False,
    overrides: dict[str, Any] = {},
    **alignment_kwargs: dict[str, Any],
) -> None:
    """
    Compare the features of two paired datasets using a trained model through registration, crop extraction, and PCA

    Parameters
    ----------
    model_name : str
        The name of the trained model.
    fixed_dataset_name : str
        Dataset name to use as the fixed images (i.e. the reference against which the moving images are registered)
    moving_dataset_name : str
        Dataset name to use as the moving images (i.e. the images to be registered to the fixed images)
    alignment_method : str
        The method used for alignment. Options are "sift" or "template". "sift" is recommended for the 20x pre/post fixation datasets, while "template" is recommended for the 20x/40x datasets.
    align_fluo : bool
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    pca_dir : str | None
        Path to the PCA model directory. If None, PCA will be calculated from existing features
    overrides : Union[str, Dict], optional
        Overrides for the model configuration, by default {}. One relevant override is `model.spatial_inferer.splitter.overlap`, which determines the percent overlap of patches extracted during sliding window inference and can increase the number of samples used for the dataset comparison.
    **alignment_kwargs : Dict[str, Any]
        Additional arguments for the alignment function.
    """
    mlflow_id = load_model_config(model_name).mlflow_run_id
    model_path = Path(get_output_path(f"models/{model_name}"))
    path_dict = download_model(mlflow_id, model_path)

    save_path = model_path / f"{fixed_dataset_name}_vs_{moving_dataset_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    data_save_path = save_path / f"aligned_{fixed_dataset_name}_vs_{moving_dataset_name}.csv"

    if not data_save_path.exists():
        data = align_all_positions(
            fixed_dataset_name,
            moving_dataset_name,
            save_path,
            alignment_method,
            align_fluo,
            **alignment_kwargs,
        )
        # channel used for inference is in the aligned images, which are single channel
        data["channel"] = 0
        data.to_csv(data_save_path, index=False)

    if align_only:
        print(
            f"Aligned images saved to {save_path}. Skipping feature extraction and PCA projection."
        )
        return

    # apply on fixed images
    fixed_overrides = overrides.copy()  # copy to avoid overriding the original
    fixed_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "fixed"})
    fixed_overrides = generate_overrides(
        fixed_overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=fixed_dataset_name,
        model_name=model_name,
    )

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])
    model.override_config(fixed_overrides)
    model.predict()

    # apply on moving images
    overrides.update({"data.predict_dataloaders.dataset.img_path_column": "moving"})
    overrides = generate_overrides(
        overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=moving_dataset_name,
        model_name=model_name,
    )
    model.override_config(overrides)
    model.predict()

    # compare paired features
    fixed_features_path = str(
        save_path / f"predict_{fixed_dataset_name}_{model_name}_features.parquet"
    )
    add_fmsid_to_config(
        fixed_features_path,
        fixed_dataset_name,
        mlflow_id,
        model_path,
    )
    moving_features_path = str(
        save_path / f"predict_{moving_dataset_name}_{model_name}_features.parquet"
    )
    add_fmsid_to_config(
        moving_features_path,
        moving_dataset_name,
        mlflow_id,
        model_path,
    )

    # load features for comparison
    fixed_features = pd.read_parquet(fixed_features_path)
    moving_features = pd.read_parquet(moving_features_path)

    plot_paired_features(
        fixed_features,
        fixed_dataset_name,
        moving_features,
        moving_dataset_name,
        save_path,
        pca_dir,
    )


def main(
    pca_dir: str | None = None,
    fixed_finetuned_model_name: str = "diffae_finetuned_for_fixed",
    model_name: str = "diffae_04_10",
    align_only: bool = False,
) -> None:
    """ "
    Main function to compare paired features of fixed and moving images using a trained model.
    Parameters
    ----------"
    pca_dir : str | None
        Path to the PCA model directory. If None, PCA will be calculated from existing features"
    """
    overrides = {"model.spatial_inferer.splitter.overlap": 0.9}

    datasets_live_fixed = {
        "fixed": [
            "20250214_pairedPreFixation",
        ],
        "moving": [
            "20250214_pairedPostFixation",
        ],
    }
    for fixed, moving in zip(
        datasets_live_fixed["fixed"], datasets_live_fixed["moving"], strict=True
    ):
        compare_paired_features(
            # use model finetuned for fixation
            fixed_finetuned_model_name,
            fixed,
            moving,
            alignment_method="sift",
            pca_dir=pca_dir,
            overrides=overrides,
            align_only=align_only,
        )

    datasets_20x_40x = {
        "fixed": ["20250110_paired20X", "20250227_paired20X", "20250228_paired20X"],
        "moving": ["20250110_paired40X", "20250227_paired40X", "20250228_paired40X"],
    }
    for fixed, moving in zip(datasets_20x_40x["fixed"], datasets_20x_40x["moving"], strict=True):
        compare_paired_features(
            model_name,
            fixed,
            moving,
            alignment_method="template",
            pca_dir=pca_dir,
            overrides=overrides,
            align_only=align_only,
        )


if __name__ == "__main__":
    fire.Fire(main)
