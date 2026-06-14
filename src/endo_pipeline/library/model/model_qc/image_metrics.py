"""Image similarity metrics for model QC evaluation.

Provides Pearson correlation, SSIM, and LPIPS computations used by the
model QC pipeline to compare ground-truth and reconstructed images.
"""

from numpy.typing import NDArray
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from endo_pipeline.library.model.model_comparison import LPIPSCalculator, ModelComparisonMetrics

# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------


def compute_correlation(img1: NDArray, img2: NDArray) -> float:
    """Compute Pearson correlation coefficient between two images.

    Parameters
    ----------
    img1
        First Image/Crop to compare
    img2
        Second Image/Crop to compare

    Returns
    -------
        Pearson correlation coefficient in [-1, 1].
    """
    corr, _ = pearsonr(img1.ravel(), img2.ravel())
    return float(corr)


def compute_ssim(
    img1: NDArray,
    img2: NDArray,
    data_range: float | None = None,
) -> float:
    """Compute SSIM (Structural Similarity Index) between two images.

    Parameters
    ----------
    img1
        First Image/Crop to compare
    img2
        Second Image/Crop to compare
    data_range
        Dynamic range of the input images.  When ``None`` the range is
        assumed to be 2.0 (since the images in our pipeline
        are normalised to [-1, 1]).

    Returns
    -------
        SSIM score in [0, 1] (1 = identical).
    """
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()

    if data_range is None:
        data_range = 2.0

    return float(ssim(img1, img2, data_range=data_range))


# ---------------------------------------------------------------------------
# Composite helpers
# ---------------------------------------------------------------------------


def compute_all_metrics(
    img1: NDArray,
    img2: NDArray,
    lpips_calculator: LPIPSCalculator | None = None,
) -> ModelComparisonMetrics:
    """Compute correlation, SSIM, and LPIPS between two images.

    Parameters
    ----------
    img1
        First Image/Crop to compare
    img2
        Second Image/Crop to compare
    lpips_calculator
        Pre-initialised calculator.  A new one is created when ``None``.

    Returns
    -------
    ImageMetrics with ``correlation``, ``ssim``, and ``lpips`` fields.
    """
    corr = compute_correlation(img1, img2)
    ssim_score = compute_ssim(img1, img2)

    if lpips_calculator is None:
        lpips_calculator = LPIPSCalculator()
    lpips_score = lpips_calculator.compute(img1, img2)

    return ModelComparisonMetrics(correlation=corr, ssim=ssim_score, lpips=lpips_score)


def compute_denoising_metrics(
    ground_truth: NDArray,
    denoised_images: list[NDArray],
    lpips_calculator: LPIPSCalculator | None = None,
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
    if lpips_calculator is None:
        lpips_calculator = LPIPSCalculator()

    if compute_all_noise_levels:
        metrics = [
            compute_all_metrics(ground_truth, img.squeeze(), lpips_calculator)._asdict()
            for img in denoised_images
        ]
        metrics_100 = metrics[-1]
    else:
        denoised_100 = denoised_images[-1].squeeze()
        metrics_100 = compute_all_metrics(ground_truth, denoised_100, lpips_calculator)._asdict()
        metrics = None

    return metrics, metrics_100
