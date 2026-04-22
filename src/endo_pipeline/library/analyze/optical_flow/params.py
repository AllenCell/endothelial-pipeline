import logging

from endo_pipeline.configs import TimepointAnnotation, get_subset_of_timepoint_annotations
from endo_pipeline.settings.optical_flow import (
    OPTICAL_FLOW_CHANNEL_ATTACHMENT,
    OPTICAL_FLOW_CHANNEL_PERCENTILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel-aware parameter resolution
# ---------------------------------------------------------------------------
def resolve_percentile(channel: str, explicit: int | None = None) -> int:
    """Return the intensity percentile for thresholding.

    If *explicit* is ``None``, look up the channel in the built-in table
    (EGFP → 95, BF → 0); otherwise use the given value directly.

    Parameters
    ----------
    channel
        Channel name (e.g. ``"BF"``).
    explicit
        Override value.  When provided, the channel table is ignored.

    Returns
    -------
    :
        Percentile used to compute the intensity threshold.

    """
    if explicit is not None:
        return explicit
    if channel not in OPTICAL_FLOW_CHANNEL_PERCENTILE:
        raise ValueError(
            f"Unknown channel {channel!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_PERCENTILE)}. "
            "Pass an explicit percentile to override."
        )
    return OPTICAL_FLOW_CHANNEL_PERCENTILE[channel]


def resolve_attachment(channel: str, explicit: float | None = None) -> float:
    """Return the TVL1 attachment (lambda) for a given channel.

    If *explicit* is None, look up the channel in the built-in table
    (EGFP -> 7.5, BF -> 2.5), else use the given value.

    Parameters
    ----------
    channel
        Channel name (e.g. ``"BF"``).
    explicit
        Override value.  When provided, the channel table is ignored.

    Returns
    -------
    :
        Attachment value to pass to :func:`compute_tvl1`.
    """
    if explicit is not None:
        return float(explicit)
    if channel not in OPTICAL_FLOW_CHANNEL_ATTACHMENT:
        raise ValueError(
            f"Unknown channel {channel!r}. "
            f"Supported channels: {sorted(OPTICAL_FLOW_CHANNEL_ATTACHMENT)}. "
            "Pass an explicit attachment to override."
        )
    return OPTICAL_FLOW_CHANNEL_ATTACHMENT[channel]


# ---------------------------------------------------------------------------
# Annotation exclusion
# ---------------------------------------------------------------------------
def default_annotations_to_exclude(
    include_cell_piling: bool = False,
    include_pre_steady_state: bool = False,
) -> list[TimepointAnnotation]:
    """Build the default timepoint-annotation exclusion list.

    Delegates to :func:`get_subset_of_timepoint_annotations` and passes
    the annotations that should be *kept* (ignored from exclusion).
    By default every annotation is excluded; the two boolean flags
    optionally preserve ``CELL_PILING`` and ``NOT_STEADY_STATE``
    timepoints.

    Note: the caller can set ``include_all_conditions=True`` to skip this function
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
    :
        Annotations whose timepoints should be filtered out.
    """
    annotations_to_ignore: list[TimepointAnnotation] = []
    if include_cell_piling:
        annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
    if include_pre_steady_state:
        annotations_to_ignore.append(TimepointAnnotation.NOT_STEADY_STATE)
    return get_subset_of_timepoint_annotations(annotations_to_ignore)
