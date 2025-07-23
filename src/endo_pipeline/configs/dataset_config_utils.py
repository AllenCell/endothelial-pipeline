"""Methods for working with dataset configs."""

import logging
from pathlib import Path
from typing import Literal

from bioio import BioImage

from src.endo_pipeline.configs import (
    DatasetCollectionConfig,
    DatasetConfig,
    load_all_dataset_configs,
    load_dataset_collection_config,
)

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


def get_available_channels_for_all_positions(dataset: DatasetConfig) -> dict[int, list[str]]:
    """Get available channels for all positions in given dataset."""

    return {
        position: get_available_channels_for_position(dataset, position)
        for position in dataset.zarr_positions
    }


def get_available_channels_for_position(dataset: DatasetConfig, position: int) -> list[str]:
    """Get available channels for a position in given dataset."""

    # TODO: we may want to replace this with channel names directly tracked in
    # dataset configs, to avoid needing to load Zarrs every time we want to
    # access channel names

    zarr_file = get_zarr_file_for_position(dataset, position)
    return BioImage(zarr_file).channel_names


def get_channel_indices_for_all_positions(
    dataset: DatasetConfig, channel_names: list[str]
) -> dict[int, list[int | None]]:
    """Get the index of each of the specified channels in given dataset."""

    return {
        position: get_channel_indices_for_position(dataset, position, channel_names)
        for position in dataset.zarr_positions
    }


def get_channel_indices_for_position(
    dataset: DatasetConfig, position: int, channel_names: list[str]
) -> list[int | None]:
    """Get the index of each of the specified channels in given dataset."""

    available_channels = get_available_channels_for_position(dataset, position)
    return [
        available_channels.index(channel) if channel in available_channels else None
        for channel in channel_names
    ]


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


def get_filtered_dataset_collection_name(
    sample_type: Literal["live", "fixed"],
    objective: Literal["20X", "40X"],
    microscope: Literal["3i", "Nikon"],
) -> str:
    """Get name of dataset collection filtered by sample type, objective, and microscope."""

    return f"{sample_type}_{objective}_objective_{microscope}_microscope"


def make_filtered_dataset_collection(
    sample_type: Literal["live", "fixed"],
    objective: Literal["20X", "40X"],
    microscope: Literal["3i", "Nikon"],
) -> DatasetCollectionConfig:
    """Create dataset collection filtered by sample type, objective, and microscope."""

    dataset_configs = load_all_dataset_configs()
    dataset_collection_names = []

    for dataset_config in dataset_configs:
        if (
            dataset_config.live_or_fixed_sample == sample_type
            and objective in dataset_config.name  # this will become a key soon
            and dataset_config.microscope == microscope
        ):
            dataset_collection_names.append(dataset_config.name)

    dataset_collection = DatasetCollectionConfig(
        name=get_filtered_dataset_collection_name(sample_type, objective, microscope),
        description=(
            f"Collection of {sample_type} datasets with {objective} objective "
            f"from the {microscope} microscope."
        ),
        datasets=dataset_collection_names,
    )

    return dataset_collection


def validate_filtered_dataset_collection(
    sample_type: Literal["live", "fixed"],
    objective: Literal["20X", "40X"],
    microscope: Literal["3i", "Nikon"],
) -> None:
    """Validate dataset collection filtered by sample type, objective, and microscope."""

    collection_name = get_filtered_dataset_collection_name(sample_type, objective, microscope)
    generated_collection = make_filtered_dataset_collection(sample_type, objective, microscope)
    loaded_collection = load_dataset_collection_config(collection_name)

    if sorted(loaded_collection.datasets) != sorted(generated_collection.datasets):
        logger.error(
            "Generated dataset collection [ %s ] does not match loaded dataset collection",
            collection_name,
        )
        logger.info(
            "Generated dataset collection [ %s ] contains datasets [ %s ]",
            collection_name,
            " | ".join(generated_collection.datasets),
        )
        logger.info(
            "Loaded dataset collection [ %s ] contains datasets [ %s ]",
            collection_name,
            " | ".join(generated_collection.datasets),
        )
