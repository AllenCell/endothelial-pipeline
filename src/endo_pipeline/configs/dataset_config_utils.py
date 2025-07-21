"""Methods for working with dataset configs."""

import logging
from pathlib import Path
from typing import Literal

from src.endo_pipeline.configs import (
    DatasetCollectionConfig,
    DatasetConfig,
    load_all_dataset_configs,
)

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


def make_sample_type_objective_microscope_collection(
    sample_type: Literal["live", "fixed"],
    objective: Literal["20X", "40X"],
    microscope: Literal["3i", "Nikon"],
) -> DatasetCollectionConfig:
    """
    Create and return collection of datasets that are
    of a specific sample type, objective, and microscope.
    """
    dataset_configs = load_all_dataset_configs()
    dataset_collection_names = []
    for dataset_config in dataset_configs:
        if (  # filter datasets based on sample type, objective, and microscope
            dataset_config.live_or_fixed_sample == sample_type
            and objective in dataset_config.name  # this will become a key soon
            and dataset_config.microscope == microscope
        ):
            dataset_collection_names.append(dataset_config.name)
    dataset_collection = DatasetCollectionConfig(
        name=f"{sample_type}_{objective}_objective_{microscope}_microscope",
        description=f"Collection of {sample_type} datasets with {objective} objective from the {microscope} microscope.",
        datasets=dataset_collection_names,
    )

    return dataset_collection
