"""Methods for calculating model comparison metrics."""

import logging
from typing import NamedTuple

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ModelComparisonMetrics(NamedTuple):
    """Container for image similarity metrics used for model comparison."""

    correlation: float
    """Pearson correlation coefficient between the two images (range [-1, 1])."""

    ssim: float
    """Structural Similarity Index Measure (SSIM) score (range [0, 1], 1 = identical)."""

    lpips: float
    """Learned Perceptual Image Patch Similarity (LPIPS) score (lower is better, 0 = identical)."""


class LPIPSCalculator:
    """
    Singleton for LPIPS (Learned Perceptual Image Patch Similarity) calculator.

    The underlying ``torchmetrics`` model is created when the class is first
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
    def compute(cls, img1: "NDArray", img2: "NDArray") -> float:
        """
        Compute LPIPS between two images.

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
