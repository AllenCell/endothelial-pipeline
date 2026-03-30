"""Global settings for track statistics analyses (e.g., MSD, long-timescale statistics)."""

MAX_MSD_LAG: int = 24
"""Maximum time lag (in number of frames) to consider for mean squared
displacement calculation."""

MSD_Y_AXIS_LIMITS: tuple[float, float] = (2e-3, 1e0)
"""Axes limits for mean squared displacement plots."""

LONG_TRACK_THRESHOLD_LENGTH: int = 150
"""
Minimum track length (in number of timepoints) to include in analyses of
long-timescale statistics (e.g., mean squared displacement) in dynamics
workflows.
"""
