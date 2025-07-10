from dataclasses import dataclass
from typing import Literal


@dataclass
class SINDyRegressionParameters:
    """
    Parameters for polynomial SINDy-based regression on
    Kramers-Moyal drift and diffusion coefficients.
    """

    drift_feat_degree: int = 4
    """Polynomial degree for feature variable expansion of drift coefficient."""

    diffusion_feat_degree: int = 4
    """Polynomial degree for feature variable expansion of diffusion coefficient."""

    drift_param_degree: int = 6
    """Polynomial degree for control parameter expansion of drift coefficient."""

    diffusion_param_degree: int = 6
    """Polynomial degree for control parameter expansion of diffusion coefficient."""


@dataclass
class KramersMoyalParameters:
    """Parameters for estimation of 2D Kramers-Moyal coefficients."""

    num_bins: list[int] = [75, 75]
    """Number of bins to use along each dimension."""

    bandwidth: float = 0.075
    """Bandwidth for kernel regression."""

    kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular", "quartic"] = "gaussian"
    """
    Kernel type to use (e.g., 'gaussian', 'uniform', 'epanechnikov').

    These are the kernels found in library.analyze.kramersmoyal.km_kernels.
    """


@dataclass
class DynamicsConfig:
    """Config for 2D dynamics analysis pipeline."""

    name: str
    """Name of the configuration."""

    pcs_to_analyze: list[str] = [0, 1]
    """List of principal components to include in analysis."""

    dt: float = 5
    """Time interval between frames minutes."""

    kramers_moyal: KramersMoyalParameters = KramersMoyalParameters()
    """Parameters for Kramers-Moyal coefficient estimation."""

    sindy_parameters: SINDyRegressionParameters = SINDyRegressionParameters()
    """Configuration for polynomial regression using SINDy."""

    num_points_pplane: int = 50
    """Number of points to use along both dimensions for grid in phase plane plots."""

    num_bins_histogram: int[50, 50]
    """Number of bins in each dimension to use for plotting stationary histogram."""

    num_bins_landscape: int = [60, 60]
    """Number of bins in each dimension to use for plotting probability landscape."""

    shear_stress_range: list[float] = [4, 30]
    """Range of shear stress values to use for plotting fixed points and landscape as a function of shear stress."""

    num_shear_fixed_points: int = 30
    """Number of shear stress values to sweep for plotting fixed points within shear_stress_range."""

    num_shear_landscape: int = 15
    """Number of shear stress values to sweep for plotting probability landscape within shear_stress_range."""

    quiver_downsample_factor: int = 10
    """Downsample factor for quiver plot on top of landscape to reduce number of arrows."""

    norm_vectors: bool = False
    """Whether to normalize vectors in quiver plot on top of landscape."""
