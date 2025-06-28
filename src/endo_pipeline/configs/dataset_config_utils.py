"""Methods for working with dataset configs."""

import logging

from src.endo_pipeline.configs import DatasetConfig

logger = logging.getLogger(__name__)


def get_specific_channel_order(
    dataset: DatasetConfig,
) -> tuple[int | None, int, int | None, int | None, int | None]:
    """Get the specific channel order for given dataset."""

    return (
        dataset.channel_488_index,
        dataset.brightfield_channel_index,
        dataset.channel_405_index,
        dataset.channel_561_index,
        dataset.channel_640_index,
    )
