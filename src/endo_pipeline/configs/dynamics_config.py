from dataclasses import dataclass, field
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

    num_bins: list[int] = field(default_factory=lambda: [70, 70])
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

    pcs_to_analyze: list[str] = field(default_factory=lambda: [0, 1])
    """
    List of principal components to include in analysis.

    This list should contain the indices of the principal components
    that are to be analyzed. For example, [0, 1] means the first two
    principal components will be included in the analysis.
    """

    dt: float = 5
    """
    Time interval between frames in preferred units.

    Default is 5, which corresponds to 5 minutes between frames.
    """

    kramers_moyal: KramersMoyalParameters = field(default_factory=KramersMoyalParameters)
    """Parameters for Kramers-Moyal coefficient estimation."""

    sindy_parameters: SINDyRegressionParameters = field(default_factory=SINDyRegressionParameters)
    """Configuration for polynomial regression using SINDy."""

    num_points_pplane: int = 50
    """Number of points to use along both dimensions for grid in phase plane plots."""

    num_bins_histogram: list[int] = field(default_factory=lambda: [50, 50])
    """Number of bins in each dimension to use for plotting stationary histogram."""

    num_bins_landscape: list[int] = field(default_factory=lambda: [60, 60])
    """Number of bins in each dimension to use for plotting probability landscape."""

    shear_stress_range: list[float] = field(default_factory=lambda: [4.0, 30.0])
    """Range of shear stress values to use for plotting fixed points and landscape as a function of shear stress."""

    num_shear_fixed_points: int = 30
    """Number of shear stress values to sweep for plotting fixed points within shear_stress_range."""

    num_shear_landscape: int = 15
    """Number of shear stress values to sweep for plotting probability landscape within shear_stress_range."""

    quiver_downsample_factor: int = 10
    """Downsample factor for quiver plot on top of landscape to reduce number of arrows."""

    norm_vectors: bool = False
    """Whether to normalize vectors in quiver plot on top of landscape."""
