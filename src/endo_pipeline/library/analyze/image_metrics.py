"""
Image similarity metrics for comparing ground truth and reconstructed images.

This module provides functions for computing various image quality metrics:
- Pearson correlation coefficient
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import NDArray
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

if TYPE_CHECKING:
    import torch


@dataclass
class ImageMetrics:
    """Container for image similarity metrics."""

    correlation: float
    ssim: float
    lpips: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "correlation": self.correlation,
            "ssim": self.ssim,
            "lpips": self.lpips,
        }


class LPIPSCalculator:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) calculator.

    Lazily initializes the LPIPS model on first use to avoid unnecessary GPU memory
    allocation when LPIPS is not needed.

    Parameters
    ----------
    net_type : str
        Network type for LPIPS ('vgg' or 'alex'). Default is 'vgg'.
    device : str | None
        Device to run LPIPS on ('cuda', 'cpu', or None for auto-detect).
    """

    def __init__(self, net_type: str = "vgg", device: str | None = None):
        self.net_type = net_type
        self._device = device
        self._model: torch.nn.Module | None = None

    @property
    def device(self) -> str:
        """Get the device to use for LPIPS computation."""
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        """Lazily initialize and return the LPIPS model."""
        if self._model is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self._model = LearnedPerceptualImagePatchSimilarity(
                net_type=self.net_type, normalize=True
            )
            self._model = self._model.to(self.device)
        return self._model

    def compute(self, img1: NDArray, img2: NDArray) -> float:
        """
        Compute LPIPS between two images.

        Parameters
        ----------
        img1, img2 : NDArray
            Images to compare (should be 2D arrays).

        Returns
        -------
        float
            LPIPS score (lower is better, 0 = identical).
        """
        import torch

        # Ensure images are 2D
        if img1.ndim > 2:
            img1 = img1.squeeze()
        if img2.ndim > 2:
            img2 = img2.squeeze()

        # Normalize images to [0, 1] range
        img1_normalized = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_normalized = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

        # Convert to torch tensors [B, C, H, W] format
        img1_tensor = torch.from_numpy(img1_normalized).float().unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_normalized).float().unsqueeze(0).unsqueeze(0)

        # Convert to 3-channel (repeat grayscale across channels)
        img1_tensor = img1_tensor.repeat(1, 3, 1, 1)
        img2_tensor = img2_tensor.repeat(1, 3, 1, 1)

        # Move to device
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)

        with torch.no_grad():
            lpips_score = self.model(img1_tensor, img2_tensor)

        return lpips_score.item()


def compute_correlation(img1: NDArray, img2: NDArray) -> float:
    """
    Compute Pearson correlation coefficient between two images.

    Parameters
    ----------
    img1, img2 : NDArray
        Images to compare.

    Returns
    -------
    float
        Pearson correlation coefficient between -1 and 1.
    """
    corr, _ = pearsonr(img1.ravel(), img2.ravel())
    return float(corr)


def compute_ssim(img1: NDArray, img2: NDArray) -> float:
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Parameters
    ----------
    img1, img2 : NDArray
        Images to compare (should be 2D arrays).

    Returns
    -------
    float
        SSIM score between 0 and 1 (1 = identical).
    """
    # Ensure images are 2D
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()

    # Compute SSIM with appropriate data range
    data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    return float(ssim(img1, img2, data_range=data_range))


def compute_all_metrics(
    img1: NDArray,
    img2: NDArray,
    lpips_calculator: LPIPSCalculator | None = None,
) -> ImageMetrics:
    """
    Compute all image similarity metrics between two images.

    Parameters
    ----------
    img1, img2 : NDArray
        Images to compare.
    lpips_calculator : LPIPSCalculator | None
        Pre-initialized LPIPS calculator. If None, a new one will be created.

    Returns
    -------
    ImageMetrics
        Container with correlation, SSIM, and LPIPS scores.
    """
    corr = compute_correlation(img1, img2)
    ssim_score = compute_ssim(img1, img2)

    if lpips_calculator is None:
        lpips_calculator = LPIPSCalculator()
    lpips_score = lpips_calculator.compute(img1, img2)

    return ImageMetrics(correlation=corr, ssim=ssim_score, lpips=lpips_score)
