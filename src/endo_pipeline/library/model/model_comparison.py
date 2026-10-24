"""Methods for calculating model comparison metrics."""

import logging
from typing import Literal, NamedTuple

import colorcet
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.stats import circmean, pearsonr
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from skimage.metrics import hausdorff_distance
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import find_boundaries
from skimage.transform import pyramid_expand

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, load_image
from endo_pipeline.library.process.cdh5_preprocessing import (
    generate_segmentations,
    get_thresholds,
    preprocess,
    split_multinucleate_regions,
)
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_image_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
    DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
)

logger = logging.getLogger(__name__)


def _get_refined_segmentation_from_downscaled_image(
    cell_boundaries_image: "NDArray", nuclei_segmentations: "NDArray"
) -> "NDArray":
    """
    Get refined segmentation for the given image using classic segmentation
    and splitting multinucleate regions.

    Parameters
    ----------
    cell_boundaries_image
        Downscaled image of cell boundaries.
    nuclei_segmentations
        Downscaled image of nuclei segmentations.

    Returns
    -------
    :
        Refined segmentation mask.
    """
    # Determine the upscale factor required to restore the image to native resolution based on the
    # resolution level of the ZARR dataset. Cell segmentation expects images at native resolution.
    upscale_factor = 2**DIFFAE_ZARR_RESOLUTION_LEVEL

    # Squeeze out dimensions of size 1
    cell_boundaries_image = cell_boundaries_image.squeeze()
    nuclei_segmentations = nuclei_segmentations.squeeze()
    upscaled_cell_boundaries_image = pyramid_expand(cell_boundaries_image, upscale=upscale_factor)
    upscaled_nuclei_segmentations = pyramid_expand(nuclei_segmentations, upscale=upscale_factor)
    upscaled_nuclei_segmentations = np.around(upscaled_nuclei_segmentations).astype(int)

    # Perform classic segmentation on the upscaled images and refine the segmentation by splitting
    # multinucleate regions.
    processed_img = preprocess(upscaled_cell_boundaries_image)
    hyst, hyst_clean, hyst_removed = get_thresholds(processed_img)
    seg2_lab_no_mask_merge, _ = generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed, 80
    )
    refined_segmentation, _ = split_multinucleate_regions(
        cell_segmentations=seg2_lab_no_mask_merge,
        nuclei_segmentations=upscaled_nuclei_segmentations,
        cell_boundary_thresh=hyst,
        cell_boundary_image=processed_img,
    )
    return refined_segmentation


class ModelComparisonMetrics(NamedTuple):
    """Container for image similarity metrics used for model comparison."""

    correlation: float
    """Pearson correlation coefficient between the two images (range [-1, 1])."""

    ssim: float
    """Structural Similarity Index Measure (SSIM) score (range [0, 1], 1 = identical)."""

    lpips: float
    """Learned Perceptual Image Patch Similarity (LPIPS) score (lower is better, 0 = identical)."""

    hausdorff: float
    """Hausdorff distance between the two images (lower is better, 0 = identical)."""

    mean_orientation_difference: float
    """Mean orientation (deg) difference between the two images (lower is better, 0 = identical)."""

    mean_relative_aspect_ratio: float
    """Mean elongation of image 2 relative to mean elongation of image 1 (lower is better, 1 = identical)."""

    mean_relative_area: float
    """Mean area of image 2 relative to mean area of image 1 (lower is better, 1 = identical)."""

    number_of_segmentations_difference: int
    """Difference in the number of segmentations (img2 - img1)."""

    ssim_seg: float
    """Structural Similarity Index Measure (SSIM) score between two segmentation images(range [0, 1], 1 = identical)."""

    hausdorff_seg: float
    """Hausdorff distance between two segmentation images (lower is better, 0 = identical)."""


class ModelComparisonMetricsCalculator:
    """
    Singleton for calculating model comparison metrics between two images.

    For LPIPS (Learned Perceptual Image Patch Similarity) calculations, the
    underlying ``torchmetrics`` model is created when the class is first
    instantiated to avoid unnecessary GPU memory allocation if the metric is not
    called. On subsequent instantiations, this same instance is used.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            import torch
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            cls._instance = super().__new__(cls)
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.model = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
            cls.model = cls.model.to(cls.device)

            logger.info("Initialized model for calculating LPIPS")

        return cls._instance

    @classmethod
    def compute_correlation(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute Pearson correlation coefficient between two images.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.

        Returns
        -------
        :
            Pearson correlation coefficient in [-1, 1].
        """

        corr, _ = pearsonr(img1.ravel(), img2.ravel())
        return float(corr)  # type: ignore[reportArgumentType]

    @classmethod
    def compute_ssim(
        cls, img1: "NDArray", img2: "NDArray", data_range: float | None = None
    ) -> float:
        """
        Compute SSIM (Structural Similarity Index) between two images.

        When dynamic range of images is not given, assume range to be 2.0 since
        our images are normalized between -1 and 1.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.
        data_range
            Dynamic range of the input images.

        Returns
        -------
        :
            SSIM score in [0, 1] where 1 = identical.
        """

        if img1.ndim > 2:
            img1 = img1.squeeze()

        if img2.ndim > 2:
            img2 = img2.squeeze()

        if data_range is None:
            data_range = 2.0

        return float(ssim(img1, img2, data_range=data_range))  # type: ignore[reportArgumentType]

    @classmethod
    def compute_lpips(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.

        Returns
        -------
        :
            LPIPS score where lower = more similar and 0 = identical.
        """

        import torch

        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

        img1_t = torch.from_numpy(img1_norm).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        img2_t = torch.from_numpy(img2_norm).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

        img1_t = img1_t.to(cls.device)
        img2_t = img2_t.to(cls.device)

        with torch.no_grad():
            score = cls.model(img1_t, img2_t)

        return score.item()

    @classmethod
    def compute_orientation(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Run segmentation on and compute mean orientations of two images.

        Parameters
        ----------
        img1
            First image to compute mean orientation for.
        img2
            Second image to compute mean orientation for.
        img_nuc_seg
            Nuclei segmentation image (used to refine segmentations).

        Returns
        -------
        :
            Orientations for image 1 and image 2.
        """
        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        props1 = regionprops(img1)
        props2 = regionprops(img2)

        # Compute mean orientation for each image using circular mean to account for angle wrapping
        orientation_max = np.pi / 2
        orientation_min = -np.pi / 2
        mean_orientation1 = circmean(
            [prop.orientation for prop in props1],
            high=orientation_max,
            low=orientation_min,  # type: ignore[reportArgumentType]
        )
        mean_orientation2 = circmean(
            [prop.orientation for prop in props2],
            high=orientation_max,
            low=orientation_min,  # type: ignore[reportArgumentType]
        )

        # NOTE: we do not use the existing get_smallest_angle_difference method because it requires
        # the angle range to be a full 2*pi, and here we have a range of -pi/2 to pi/2.
        # Unwrap orientations to ensure continuity for angle difference calculation
        unwrapped_orientations = np.unwrap(
            [mean_orientation1, mean_orientation2], period=orientation_max - orientation_min
        )
        mean_orientation_difference = np.abs(np.diff(unwrapped_orientations))

        return float(mean_orientation_difference)

    @classmethod
    def compute_relative_aspect_ratio(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute mean aspect ratio of image 2 relative to mean aspect ratio of image 1.

        Parameters
        ----------
        img1
            First image to compute mean aspect ratio for.
        img2
            Second image to compute mean aspect ratio for.

        Returns
        -------
        :
            Mean aspect ratio of image 2 relative to mean aspect ratio of image 1.
        """
        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        props1 = regionprops(img1)
        props2 = regionprops(img2)

        mean_aspect_ratio1 = np.mean(
            [
                prop.major_axis_length / prop.minor_axis_length
                for prop in props1
                if prop.minor_axis_length > 0
            ]
        )
        mean_aspect_ratio2 = np.mean(
            [
                prop.major_axis_length / prop.minor_axis_length
                for prop in props2
                if prop.minor_axis_length > 0
            ]
        )

        mean_aspect_ratio_normalized = mean_aspect_ratio2 / mean_aspect_ratio1

        return float(mean_aspect_ratio_normalized)

    @classmethod
    def compute_relative_area(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute mean area of image 2 relative to mean area of image 1.

        Parameters
        ----------
        img1
            First image to compute mean area for.
        img2
            Second image to compute mean area for.

        Returns
        -------
        :
            Mean area of image 2 relative to mean area of image 1.
        """
        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        props1 = regionprops(img1)
        props2 = regionprops(img2)

        mean_area1 = np.mean([prop.area for prop in props1])
        mean_area2 = np.mean([prop.area for prop in props2])

        mean_area_normalized = mean_area2 / mean_area1

        return float(mean_area_normalized)

    @classmethod
    def compute_number_of_segmentations_difference(cls, img1: "NDArray", img2: "NDArray") -> int:
        """
        Compute the difference in the number of segmentations between two images defined as
        img2 number of segmentations - img1 number of segmentations.

        Parameters
        ----------
        img1
            First image to compute number of segmentations for.
        img2
            Second image to compute number of segmentations for.

        Returns
        -------
        :
            Difference in the number of segmentations (img2 - img1).
        """
        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        num_segmentations1 = np.count_nonzero(np.unique(img1))
        num_segmentations2 = np.count_nonzero(np.unique(img2))

        return num_segmentations2 - num_segmentations1

    @classmethod
    def compute_hausdorff(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute the Hausdorff distance between two images.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.

        Returns
        -------
        :
            Hausdorff distance between image 1 and image 2.
        """

        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        hausdorff = hausdorff_distance(img1, img2)

        if not isinstance(hausdorff, float):
            hausdorff = np.nan

        return float(hausdorff)

    @classmethod
    def compute_all_metrics(
        cls,
        img1: NDArray,
        img2: NDArray,
        img_nuc_seg: NDArray,
        example_name: str,
    ) -> ModelComparisonMetrics:
        """
        Compute correlation, SSIM, and LPIPS between two images.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.
        img_nuc_seg
            Nuclei segmentation image.

        Returns
        -------
        :
            Computed correlation, SSIM, and LPIPS metrics.
        """

        img1_seg = _get_refined_segmentation_from_downscaled_image(img1, img_nuc_seg)
        img2_seg = _get_refined_segmentation_from_downscaled_image(img2, img_nuc_seg)

        # Save the refined segmentations for validation
        plot_segmentation_validation(example_name, img1, img2, img1_seg, img2_seg, img_nuc_seg)

        return ModelComparisonMetrics(
            correlation=cls.compute_correlation(img1, img2),
            ssim=cls.compute_ssim(img1, img2),
            lpips=cls.compute_lpips(img1, img2),
            hausdorff=cls.compute_hausdorff(img1, img2),
            mean_orientation_difference=cls.compute_orientation(img1_seg, img2_seg),
            mean_relative_aspect_ratio=cls.compute_relative_aspect_ratio(img1_seg, img2_seg),
            mean_relative_area=cls.compute_relative_area(img1_seg, img2_seg),
            number_of_segmentations_difference=cls.compute_number_of_segmentations_difference(
                img1_seg, img2_seg
            ),
            ssim_seg=cls.compute_ssim(find_boundaries(img1_seg), find_boundaries(img2_seg)),
            hausdorff_seg=cls.compute_hausdorff(
                find_boundaries(img1_seg), find_boundaries(img2_seg)
            ),
        )


def load_nuclei_segmentation_image(
    example: "ExampleImage", model_config: "DictConfig"
) -> "NDArray":
    """
    Load nuclei segmentation image for select example.

    Parameters
    ----------
    example
        Example image metadata.
    model_config
        Model configuration instance.

    Returns
    -------
    :
        Nuclei segmentation image.
    """
    dataset_config = load_dataset_config(example.dataset_name)
    nuc_manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_config, example.position)
    nuc_seg = load_image(
        nuc_location,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=example.timepoint,
        squeeze=False,
        compute=True,
    )
    nuc_seg = nuc_seg.squeeze(axis=(DIMENSION_ORDER.index("T"), DIMENSION_ORDER.index("C")))

    crop_size = model_config.model.image_shape[-1]
    return crop_image(nuc_seg, example.crop_x_start, example.crop_y_start, crop_size)


def load_transformed_example_image(
    example: "ExampleImage", model_config: "DictConfig", target_key: str
) -> "NDArray":
    """
    Load transformed and cropped image for select example.

    Parameters
    ----------
    example
        Example image metadata.
    model_config
        Model configuration instance.
    target_key
        Sample dictionary target image key.

    Returns
    -------
    :
        Transformed and cropped example image
    """

    # Load image for select dataset at given position and timepoint
    dataset_config = load_dataset_config(example.dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
    img = load_image(
        zarr_loc,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=example.timepoint,
        squeeze=True,
        compute=True,
    )

    # Extract transformation steps and apply to image
    data = create_data_dict_loaded_image(img)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    # Extract the target image and crop
    crop_size = model_config.model.image_shape[-1]
    conditioning_img = get_target_image_from_sample(sample, target_key)
    return crop_image(conditioning_img, example.crop_x_start, example.crop_y_start, crop_size)


def load_transformed_conditioning_example_image(
    example: "ExampleImage",
    model_config: "DictConfig",
) -> np.ndarray:
    """
    Load transformed and cropped conditioning image for select example.

    Parameters
    ----------
    example
        Example image metadata.
    model_config
        Model configuration instance.

    Returns
    -------
    :
        Transformed and cropped conditioning image.
    """

    target_key = model_config.model.condition_key
    return load_transformed_example_image(example, model_config, target_key)


def load_transformed_diffusion_example_image(
    example: "ExampleImage",
    model_config: "DictConfig",
) -> np.ndarray:
    """
    Load transformed and cropped diffusion image for select example.

    Parameters
    ----------
    example
        Example image metadata.
    model_config
        Model configuration instance.

    Returns
    -------
    :
        Transformed and cropped diffusion image.
    """

    target_key = DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
    return load_transformed_example_image(example, model_config, target_key)


def load_model_comparison_metrics(
    model_runs: list[tuple[str, str]],
    example_groups: list[Literal["training", "validation", "replicate"]] | None = None,
) -> pd.DataFrame:
    """
    Load all model comparison metric for given model runs.

    Parameters
    ----------
    model_runs
        List of model runs as (model_manifest_name, run_name).
    example_groups
        List of example groups to include. Use None to include all groups.

    Returns
    -------
    :
        Dataframe containing all loaded model comparison metrics.
    """

    unique_manifest_names = {model_run[0] for model_run in model_runs}

    manifests = {
        manifest_name: load_dataframe_manifest(
            f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{manifest_name}"
        )
        for manifest_name in unique_manifest_names
    }

    all_metrics = []

    for manifest_name, run_name in model_runs:
        location = get_dataframe_location_for_dataset(manifests[manifest_name], run_name)
        all_metrics.append(load_dataframe(location))

    all_metrics_df = pd.concat(all_metrics)

    if example_groups is not None:
        all_metrics_df = all_metrics_df[
            all_metrics_df[Column.MODEL_COMPARISON_EXAMPLE_GROUP].isin(example_groups)
        ]

    return all_metrics_df


def aggregate_model_comparison_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate model comparison metrics across examples and seeds.

    Means and standard deviations are taken across examples and random seeds. An
    extra column "N" is added containing the total number of examples and seeds.

    Parameters
    ----------
    metrics_df
        Dataframe containing all model comparison metrics.

    Returns
    -------
    :
        Dataframe containing aggregated model comparison metrics.
    """

    metrics_columns = [
        Column.MODEL_COMPARISON_BASELINE_CORRELATION,
        Column.MODEL_COMPARISON_BASELINE_SSIM,
        Column.MODEL_COMPARISON_BASELINE_LPIPS,
        Column.MODEL_COMPARISON_CORRELATION,
        Column.MODEL_COMPARISON_SSIM,
        Column.MODEL_COMPARISON_LPIPS,
    ]

    metadata_columns = [
        Column.DiffAEData.MODEL_MANIFEST,
        Column.DiffAEData.MODEL_RUN,
        Column.MODEL_COMPARISON_EXAMPLE_GROUP,
    ]

    aggregate_means = metrics_df.groupby(metadata_columns)[metrics_columns].mean().reset_index()
    aggregate_stdevs = metrics_df.groupby(metadata_columns)[metrics_columns].std().reset_index()
    aggregate_counts = (
        metrics_df.groupby(metadata_columns)[Column.EXAMPLE_KEY]
        .count()
        .reset_index()
        .rename(columns={Column.EXAMPLE_KEY: "N"})
    )

    return aggregate_means.merge(
        aggregate_stdevs, on=metadata_columns, suffixes=("_mean", "_stdev")
    ).merge(aggregate_counts)


def group_aggregate_model_comparison_metrics(aggregate_df: pd.DataFrame) -> dict:
    """
    Restructure entries in aggregate dataframe into nested dictionary.

    Dictionary is in the form:

    ```
    {
        example_group: {
            (model_manifest_name, run_name): {
                metric: value,
                metric: value
            },
            (model_manifest_name, run_name): {
                metric: value,
                metric: value
            }
        },
        example_group: {
            (model_manifest_name, run_name): {
                metric: value,
                metric: value
            }
        }
    }
    ```

    Parameters
    ----------
    aggregate_df
        Dataframe containing all aggregated model comparison metrics.

    Returns
    -------
    :
        Nested dictionary containing aggregated model comparison metrics.
    """

    aggregate_by_group = {}

    for group_name, group_df in aggregate_df.groupby(Column.MODEL_COMPARISON_EXAMPLE_GROUP):
        aggregate_by_group[group_name] = group_df.set_index(
            [Column.DiffAEData.MODEL_MANIFEST, Column.DiffAEData.MODEL_RUN]
        ).to_dict("index")

    return aggregate_by_group


def plot_segmentation_validation(
    example_name: str,
    img1: np.ndarray,
    img2: np.ndarray,
    img1_seg: np.ndarray,
    img2_seg: np.ndarray,
    img_nuc_seg: np.ndarray,
) -> None:
    """
    Plot segmentation validation images.

    Parameters
    ----------
    example_name
        Name of the example.
    img1
        First image.
    img2
        Second image.
    img1_seg
        Segmentation of the first image.
    img2_seg
        Segmentation of the second image.
    img_nuc_seg
        Nuclear segmentation.
    """
    output_path = get_output_path(__file__, "segmentation_validation")
    cmap = colorcet.cm.glasbey_bw
    plot_image_thumbnail(
        img1.squeeze(),
        f"{example_name}_img1",
        output_path=output_path,
        figsize=(2, 2),
    )
    plot_image_thumbnail(
        img2.squeeze(),
        f"{example_name}_img2",
        output_path=output_path,
        figsize=(2, 2),
    )
    plot_image_thumbnail(
        img1_seg.squeeze(),
        f"{example_name}_img1_seg",
        output_path=output_path,
        figsize=(2, 2),
        image_colormap=cmap,
    )
    plot_image_thumbnail(
        img2_seg.squeeze(),
        f"{example_name}_img2_seg",
        output_path=output_path,
        figsize=(2, 2),
        image_colormap=cmap,
    )
    overlay = label2rgb(
        label=img_nuc_seg.squeeze(),
        image=rescale_intensity(img1.squeeze(), out_range="uint8"),
        bg_label=0,
        alpha=0.5,
    )
    plot_image_thumbnail(
        overlay,
        f"{example_name}_nuc_seg_on_img1",
        output_path=output_path,
        figsize=(2, 2),
        image_colormap=cmap,
    )
    overlay = label2rgb(
        label=img_nuc_seg.squeeze(),
        image=rescale_intensity(img2.squeeze(), out_range="uint8"),
        bg_label=0,
        alpha=0.5,
    )
    plot_image_thumbnail(
        overlay,
        f"{example_name}_nuc_seg_on_img2",
        output_path=output_path,
        figsize=(2, 2),
        image_colormap=cmap,
    )
