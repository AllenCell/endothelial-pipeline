"""Workflow settings for bootstrapping fixed points from 3D flow field analysis."""

NUM_BOOTSTRAP_ITERATIONS: int = 100
"""Number of bootstrap iterations for fixed point confidence interval estimation
in 3D flow field analysis."""

BATCH_SIZE_SCALING_FACTOR: float = 4
"""Factor to used determine the number of batches for parallel processing in bootstrapping."""

BOOTSTRAP_MATCH_RADIUS: float = 0.1
"""Radius threshold for matching bootstrapped fixed points to baseline fixed
points in 3D flow field analysis."""

FP_CI_LOWER_PERCENTILE: float = 0.05
"""Lower percentile for fixed point confidence interval estimation in 3D flow
field analysis."""

FP_CI_UPPER_PERCENTILE: float = 0.95
"""Upper percentile for fixed point confidence interval estimation in 3D flow
field analysis."""
