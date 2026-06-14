"""Methods for calculating model comparison metrics."""

from typing import NamedTuple


class ModelComparisonMetrics(NamedTuple):
    """Container for image similarity metrics used for model comparison."""

    correlation: float
    """Pearson correlation coefficient between the two images (range [-1, 1])."""

    ssim: float
    """Structural Similarity Index Measure (SSIM) score (range [0, 1], 1 = identical)."""

    lpips: float
    """Learned Perceptual Image Patch Similarity (LPIPS) score (lower is better, 0 = identical)."""
