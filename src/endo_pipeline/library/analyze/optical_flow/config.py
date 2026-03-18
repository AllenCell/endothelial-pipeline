import logging

from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings.workflow_defaults import (
    OPTICAL_FLOW_CHANNEL_ATTACHMENT,
    OPTICAL_FLOW_CHANNEL_PERCENTILE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COHERENCE_BOX_SIZES: tuple[int, ...] = (
    1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
)
"""Non-overlapping box sizes (in pixels) for multi-scale coherence."""

DEMO_SCAN_N_CROPS: int = 6
DEMO_SCAN_N_PAIRS: int = 10
QUIVER_GRID_DIVISIONS: int = 8


# ---------------------------------------------------------------------------
# Channel-aware parameter resolution
# ---------------------------------------------------------------------------
def resolve_percentile(channel: list, explicit: int | None = None) -> int:
    """Return the intensity percentile for thresholding.

    If *explicit* is ``None``, look up the channel in the built-in table
    (EGFP → 95, BF → 0); otherwise use the given value directly.

    Parameters
    ----------
    channel
        Single-element list containing the channel name (e.g. ``["BF"]``).
    explicit
        Override value.  When provided, the channel table is ignored.

    Returns
    -------
        Percentile used to compute the intensity threshold.

    """
    if len(channel) != 1:
        raise ValueError(f"Optical flow operates on a single channel, got {channel}.")
    if explicit is not None:
        return explicit
    ch = channel[0]
    if ch not in OPTICAL_FLOW_CHANNEL_PERCENTILE:
        raise ValueError(
            f"Unknown channel {ch!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_PERCENTILE)}. "
            "Pass an explicit percentile to override."
        )
    return OPTICAL_FLOW_CHANNEL_PERCENTILE[ch]


def resolve_attachment(channel: list, explicit: float | None = None) -> float:
    """Return the TVL1 attachment (lambda) for a given channel.

    If *explicit* is None, look up the channel in the built-in table
    (EGFP -> 7.5, BF -> 25.0), else use the given value.

    Parameters
    ----------
    channel
        Single-element list with channel name.
    explicit
        Override value.  When provided, the channel table is ignored.

    Returns
    -------
        Attachment value to pass to :func:`compute_tvl1`.
    """
    if len(channel) != 1:
        raise ValueError(f"Optical flow operates on a single channel, got {channel}.")
    if explicit is not None:
        return float(explicit)
    ch = channel[0]
    if ch not in OPTICAL_FLOW_CHANNEL_ATTACHMENT:
        raise ValueError(
            f"Unknown channel {ch!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_ATTACHMENT)}. "
            "Pass an explicit attachment to override."
        )
    return OPTICAL_FLOW_CHANNEL_ATTACHMENT[ch]


# ---------------------------------------------------------------------------
# Annotation exclusion
# ---------------------------------------------------------------------------
def default_annotations_to_exclude(
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> list[TimepointAnnotation]:
    """Build the default timepoint-annotation exclusion list.

    Returns nine quality annotations (scope errors, temporary artifacts,
    XY/Z shifts, unfed) and, depending on the boolean flags, up to two
    lifecycle annotations (``CELL_PILING``, ``NOT_STEADY_STATE``).

    Note: the caller can set ``include_all=True`` to skip this function
    and disable all annotation filtering.

    Parameters
    ----------
    include_cell_piling
        When ``False`` (default), timepoints annotated as
        :attr:`TimepointAnnotation.CELL_PILING` are added to the
        exclusion list.
    include_pre_steady_state
        When ``False`` (default), timepoints annotated as
        :attr:`TimepointAnnotation.NOT_STEADY_STATE` are added to the
        exclusion list.

    Returns
    -------
        Annotations whose timepoints should be filtered out.
    """
    excl: list[TimepointAnnotation] = [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
        TimepointAnnotation.GFP_SCOPE_ERROR,
        TimepointAnnotation.UNFED,
        TimepointAnnotation.XY_SHIFT,
        TimepointAnnotation.Z_SHIFT,
    ]
    if not include_pre_steady_state:
        excl.append(TimepointAnnotation.NOT_STEADY_STATE)
    if not include_cell_piling:
        excl.append(TimepointAnnotation.CELL_PILING)
    return excl
