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
