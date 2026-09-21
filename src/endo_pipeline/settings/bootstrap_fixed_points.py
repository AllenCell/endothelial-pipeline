"""Workflow settings for bootstrapping fixed points from 3D flow field analysis."""

NUM_BOOTSTRAP_ITERATIONS: int = 500
"""Number of bootstrap iterations for fixed point confidence interval estimation
in 3D flow field analysis."""

BATCH_SIZE_SCALING_FACTOR: float = 4
"""Factor to used determine the number of batches for parallel processing in bootstrapping."""

BOOTSTRAP_MATCH_RADIUS: float = 0.3
"""Radius threshold for matching bootstrapped fixed points to baseline fixed
points in 3D flow field analysis."""

FP_CI_LOWER_PERCENTILE: float = 5
"""Lower percentile for fixed point confidence interval estimation in 3D flow
field analysis."""

FP_CI_UPPER_PERCENTILE: float = 95
"""Upper percentile for fixed point confidence interval estimation in 3D flow
field analysis."""

BOOTSTRAP_THRESHOLD: float = 0.4
"""Threshold for high confidence fixed points."""

SDE_ALPHA_ITO: float = 0.0
"""SDE interpretation parameter corresponding to the Ito interpretation."""

SDE_ALPHA_STRATONOVICH: float = 0.5
"""SDE interpretation parameter corresponding to the Stratonovich interpretation."""

ALPHA_CORRECTED_MANIFEST_SUFFIX: str = "alpha_corrected"
"""Manifest name suffix for bootstrap results computed with a corrected drift."""

ITO_EDGE_COLOR: str = "black"
"""Marker edge color for Ito-interpretation fixed points in comparison plots."""

ALPHA_CORRECTED_EDGE_COLOR: str = "red"
"""Marker edge color for alpha-corrected fixed points in comparison plots."""
