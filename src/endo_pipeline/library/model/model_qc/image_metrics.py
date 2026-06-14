"""Image similarity metrics for model QC evaluation.

Provides Pearson correlation, SSIM, and LPIPS computations used by the
model QC pipeline to compare ground-truth and reconstructed images.
"""

from typing import TYPE_CHECKING, Literal

from numpy.typing import NDArray
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from endo_pipeline.library.model.model_comparison import ModelComparisonMetrics

if TYPE_CHECKING:
    import torch


class LPIPSCalculator:
    """Lazily-initialized LPIPS (Learned Perceptual Image Patch Similarity) calculator.

    The underlying ``torchmetrics`` model is created on first use to prevent
    unnecessary GPU memory allocation when the metric is not explicitly called.
    """

    def __init__(self, net_type: Literal["vgg", "alex"] = "vgg", device: str | None = None):
        """Initialise the LPIPS calculator.

        Parameters
        ----------
        net_type
            The backbone network architecture used to extract features for
            the similarity calculation.
        device
            Torch device string (e.g., 'cuda', 'cpu'). If ``None``, the device
            is auto-detected based on CUDA availability.
        """
        self.net_type = net_type
        self._device = device
        self._model: torch.nn.Module | None = None

    @property
    def device(self) -> str:
        """Return the device string, auto-detecting on first access."""
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def model(self) -> "torch.nn.Module":
        """Lazily initialise and return the underlying LPIPS model.

        The ``torchmetrics`` LPIPS model instance is created on first
        access and moved to the configured device.
        """
        if self._model is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self._model = LearnedPerceptualImagePatchSimilarity(
                net_type=self.net_type, normalize=True
            )
            self._model = self._model.to(self.device)
        return self._model

    def compute(self, img1: NDArray, img2: NDArray) -> float:
        """Compute LPIPS between two images.

        Parameters
        ----------
        img1
            First Image/Crop to compare
        img2
            Second Image/Crop to compare

        Returns
        -------
            LPIPS score (lower is better, 0 = identical)
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

        img1_t = img1_t.to(self.device)
        img2_t = img2_t.to(self.device)

        with torch.no_grad():
            score = self.model(img1_t, img2_t)
        return score.item()


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
