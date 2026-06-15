"""Methods for calculating model comparison metrics."""

import logging
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    apply_img_transforms,
    create_data_dict_loaded_image,
    get_image_transforms,
    get_target_image_from_sample,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
    DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
)

logger = logging.getLogger(__name__)


class ModelComparisonMetrics(NamedTuple):
    """Container for image similarity metrics used for model comparison."""

    correlation: float
    """Pearson correlation coefficient between the two images (range [-1, 1])."""

    ssim: float
    """Structural Similarity Index Measure (SSIM) score (range [0, 1], 1 = identical)."""

    lpips: float
    """Learned Perceptual Image Patch Similarity (LPIPS) score (lower is better, 0 = identical)."""


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
        return float(corr)

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

        return float(ssim(img1, img2, data_range=data_range))

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
    def compute_all_metrics(
        cls,
        img1: NDArray,
        img2: NDArray,
    ) -> ModelComparisonMetrics:
        """
        Compute correlation, SSIM, and LPIPS between two images.

        Parameters
        ----------
        img1
            First image to compare.
        img2
            Second image to compare.

        Returns
        -------
        :
            Computed correlation, SSIM, and LPIPS metrics.
        """

        return ModelComparisonMetrics(
            correlation=cls.compute_correlation(img1, img2),
            ssim=cls.compute_lpips(img1, img2),
            lpips=cls.compute_lpips(img1, img2),
        )


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


def compute_denoising_metrics(
    ground_truth: NDArray,
    denoised_images: list[NDArray],
    compute_all_noise_levels: bool = False,
) -> tuple[list[dict] | None, dict]:
    """Compute image quality metrics for denoised images.

    Parameters
    ----------
    ground_truth
        Squeezed ground-truth image.
    denoised_images
        Denoised outputs at successive noise levels (100 % noise is last).
    lpips_calculator
        Pre-initialised calculator.  A new one is created when ``None``.
    compute_all_noise_levels
        If ``True``, return metrics for every noise level.  Otherwise only
        the 100 % noise level (last entry) is evaluated.

    Returns
    -------
    metrics_list
        Per-noise-level metric dicts, or ``None`` when
        *compute_all_noise_levels* is ``False``.
    metrics_100
        Metric dict for the 100 % noise level.
    """

    metrics_calculator = ModelComparisonMetricsCalculator()

    if compute_all_noise_levels:
        metrics = [
            metrics_calculator.compute_all_metrics(ground_truth, img.squeeze())._asdict()
            for img in denoised_images
        ]
        metrics_100 = metrics[-1]
    else:
        denoised_100 = denoised_images[-1].squeeze()
        metrics_100 = metrics_calculator.compute_all_metrics(ground_truth, denoised_100)._asdict()
        metrics = None

    return metrics, metrics_100


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

    Means and standard deviations are taken across examples and random seeds. A
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


def group_aggregate_model_comparison_metrics(
    aggregate_df: pd.DataFrame,
) -> dict[str, dict[tuple[str, str], dict[str, int | float]]]:
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
