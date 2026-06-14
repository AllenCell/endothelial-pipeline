"""Methods for calculating model comparison metrics."""

from typing import TYPE_CHECKING, Literal, NamedTuple

from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


class ModelComparisonMetrics(NamedTuple):
    """Container for image similarity metrics used for model comparison."""

    correlation: float
    """Pearson correlation coefficient between the two images (range [-1, 1])."""

    ssim: float
    """Structural Similarity Index Measure (SSIM) score (range [0, 1], 1 = identical)."""

    lpips: float
    """Learned Perceptual Image Patch Similarity (LPIPS) score (lower is better, 0 = identical)."""


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
