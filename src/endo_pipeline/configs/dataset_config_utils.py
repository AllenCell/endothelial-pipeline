"""Methods for working with dataset configs."""

import logging
from pathlib import Path
from typing import Literal

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


def get_nuclear_prediction_path(
    dataset: DatasetConfig,
    position: int,
    nuc_seg_type: Literal["label_free", "stain"],
) -> Path:
    """Get path to nuclear prediction for given position."""

    if nuc_seg_type == "label_free":
        if dataset.nuclear_label_free_seg_path is None:
            logger.error(
                "Dataset [ %s ] does not have a nuclear label free segmentation path", dataset.name
            )
            raise ValueError("'nuclear_label_free_seg_path' is None")
        else:
            return Path(dataset.nuclear_label_free_seg_path) / f"P{position}"
    elif nuc_seg_type == "stain":
        if dataset.nuclear_stain_seg_path is None:
            logger.error(
                "Dataset [ %s ] does not have nuclear stain segmentation path", dataset.name
            )
            raise ValueError("'nuclear_stain_seg_path' is None")
        else:
            return Path(dataset.nuclear_stain_seg_path) / f"P{position}"
    else:
        logger.error("Nuclear segmentation type [ %s ] is not valid", nuc_seg_type)
        raise ValueError("'nuc_seg_type' must be 'label_free' or 'stain'")
