"""Methods for working with dataset configs."""

import logging
from pathlib import Path
from typing import Literal

from src.endo_pipeline.configs import DatasetConfig

logger = logging.getLogger(__name__)


def get_available_zarr_files(dataset: DatasetConfig) -> list[Path]:
    """Get list of all available Zarr files for given dataset."""

    return [get_zarr_file_for_position(dataset, position) for position in dataset.zarr_positions]


def get_zarr_file_for_position(dataset: DatasetConfig, position: int) -> Path:
    """Get zarr file path for given dataset and position."""

    zarr_path = Path(dataset.zarr_path)
    zarr_file = zarr_path / f"{zarr_path.stem}_P{position}.ome.zarr"

    if position not in dataset.zarr_positions:
        logger.error("Position [ %s ] is not valid for dataset [ %s ]", position, dataset.name)
        raise ValueError(f"Dataset [ {dataset.name} ] only has positions {dataset.zarr_positions}")
    elif not zarr_file.exists():
        # This check intentionally does not raise an exception because we do not
        # want this method to fail if we are just getting the file names and not
        # actually loading the file. The appropriate exceptions for being unable
        # to load the file should/will be handled by loading methods.
        logger.warning("Zarr file [ %s ] does not exist", zarr_file)

    return zarr_file


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
