"""Global constants for mean squared displacement (MSD) analysis."""

MAX_MSD_LAG: int = 24
"""Maximum time lag (in number of frames) to consider for mean squared
displacement calculation."""

MSD_Y_AXIS_LIMITS: tuple[float, float] = (2e-3, 1e0)
"""Axes limits for mean squared displacement plots."""
