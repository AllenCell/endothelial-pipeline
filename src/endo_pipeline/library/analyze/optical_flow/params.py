import logging

from endo_pipeline.settings.optical_flow import OPTICAL_FLOW_CHANNEL_ATTACHMENT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel-aware parameter resolution
# ---------------------------------------------------------------------------


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
