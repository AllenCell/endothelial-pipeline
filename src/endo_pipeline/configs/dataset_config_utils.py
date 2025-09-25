"""Methods for working with dataset configs."""

import logging
import re
from pathlib import Path

from endo_pipeline.configs import (
    DatasetCollectionConfig,
    DatasetConfig,
    MicroscopeType,
    ObjectiveType,
    PositionAnnotation,
    SampleType,
    TimepointAnnotation,
    load_all_dataset_configs,
    load_dataset_collection_config,
    load_dataset_config,
)

logger = logging.getLogger(__name__)


def validate_shear_stress_regime(shear_stress: float, shear_stress_regime: tuple) -> bool:
    """Validate that the shear stress regime matches the shear stress value."""
    if shear_stress_regime[0] <= shear_stress <= shear_stress_regime[1]:
        return True
    else:
        return False


def get_regime_for_shear_stress(shear_stress: float) -> tuple[float, float]:
    """Get shear stress regime for given shear stress value."""

    from endo_pipeline.configs import ShearStressRegime

    for regime in ShearStressRegime:
        if regime.value[0] <= shear_stress <= regime.value[1]:
            return regime.value

    logger.error("No shear stress regime found for shear stress [ %f ]", shear_stress)
    raise ValueError(
        f"No shear stress regime found for shear stress [ {shear_stress} ] "
        "please update ShearStressRegime class accordingly."
    )


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


def get_position_string_from_zarr_file_path(zarr_file_path: str | Path) -> str:
    """Extract position as 'P[x]' from the file path, if found."""

    position = re.findall(r"(P[0-9]+)", Path(zarr_file_path).stem)

    if not position:
        logger.error("No position found in path [ %s ]", zarr_file_path)
        raise ValueError(f"Path '{zarr_file_path}' does not contain a valid position")

    return position[0]


def get_position_integer_from_zarr_file_path(zarr_file_path: str | Path) -> int:
    """Extract position as integer from the file path, if found."""

    position_str = get_position_string_from_zarr_file_path(zarr_file_path)

    if not position_str.startswith("P"):
        logger.error("Position string [ %s ] does not start with 'P'", position_str)
        raise ValueError(f"Position string '{position_str}' is not valid")

    return int(position_str.replace("P", ""))  # Convert 'P[x]' to x


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

    from bioio import BioImage

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


def get_frame_before_flow_change(dataset: DatasetConfig) -> int | None:
    """Get frame number immediately before the flow changes."""

    if len(dataset.flow_conditions) == 1:
        logger.warning("Dataset [ %s ] only has one flow condition", dataset.name)
        return None

    if len(dataset.flow_conditions) == 2:
        return dataset.flow_conditions[0].stop

    logger.warning("Dataset [ %s ] must have only one or two flow conditions", dataset.name)
    return None


def get_frame_after_flow_change(dataset: DatasetConfig) -> int | None:
    """Get frame number immediately after the flow changes."""

    if len(dataset.flow_conditions) == 1:
        logger.warning("Dataset [ %s ] only has one flow condition", dataset.name)
        return None

    if len(dataset.flow_conditions) == 2:
        return dataset.flow_conditions[1].start

    logger.warning("Dataset [ %s ] must have only one or two flow conditions", dataset.name)
    return None


def get_flow_at_frame(dataset: DatasetConfig, frame: int) -> float | None:
    """Get the shear stress the dataset was under at the given frame."""

    for condition in dataset.flow_conditions:
        if condition.start <= frame <= condition.stop:
            return condition.shear_stress

    logger.warning(
        "Dataset [ %s ] does not have flow condition for frame [ %d ]", dataset.name, frame
    )
    return None


def get_duration_at_flow(dataset: DatasetConfig, shear_stress: float) -> int:
    """Get the duration the dataset was under the given shear stress."""

    duration = 0

    for condition in dataset.flow_conditions:
        if condition.shear_stress == shear_stress:
            duration = duration + (condition.stop - condition.start)

    return duration


def get_annotated_positions(
    dataset: DatasetConfig, annotations: list[PositionAnnotation] | None = None
) -> list[int]:
    """Get all positions for given annotations."""

    annotated_positions: list[int] = []

    if dataset.position_annotations is None:
        logger.info("Dataset [ %s ] does not have any annotated positions", dataset.name)
        return annotated_positions

    for annotation, positions in dataset.position_annotations.items():
        if annotations is None or annotation in annotations:
            annotated_positions.extend(positions)

    return annotated_positions


def get_annotated_timepoints_for_position(
    dataset: DatasetConfig, position: int, annotations: list[TimepointAnnotation] | None = None
) -> list[int]:
    """Get all timepoints for given annotations at the given position."""

    annotated_timepoints: list[int] = []

    if dataset.timepoint_annotations is None:
        logger.info("Dataset [ %s ] does not have any annotated timepoints", dataset.name)
        return annotated_timepoints

    for annotation, positions in dataset.timepoint_annotations.items():
        if position not in positions:
            logger.info(
                "Dataset [ %s ] does not have any [ %s ] annotations for position [ %d ]",
                dataset.name,
                annotation.value,
                position,
            )
            continue

        if annotations is None or annotation in annotations:
            for timepoint in positions[position]:
                if isinstance(timepoint, int):
                    annotated_timepoints.append(timepoint)
                else:
                    annotated_timepoints.extend(list(range(timepoint[0], timepoint[1] + 1)))

    return sorted(set(annotated_timepoints))


def get_filtered_dataset_collection_name(
    sample_type: SampleType | None = None,
    objective: ObjectiveType | None = None,
    microscope: MicroscopeType | None = None,
) -> str:
    """Get name of dataset collection with various filters applied."""

    name: list[str] = []
    name.append(sample_type if sample_type is not None else "")
    name.append(f"_{objective}_objective" if objective is not None else "")
    name.append(f"_{microscope}_microscope" if microscope is not None else "")
    return "".join(name)


def get_filtered_dataset_collection_description(
    sample_type: SampleType | None = None,
    objective: ObjectiveType | None = None,
    microscope: MicroscopeType | None = None,
) -> str:
    """Get description of dataset collection with various filtered applied."""

    description: list[str] = ["Collection of"]
    description.append(f" {sample_type} datasets" if sample_type is not None else " datasets")
    description.append(f" with {objective} objective" if objective is not None else "")
    description.append(f" from the {microscope} microscope" if microscope is not None else ".")
    return "".join(description)


def make_filtered_dataset_collection(
    sample_type: SampleType | None = None,
    objective: ObjectiveType | None = None,
    microscope: MicroscopeType | None = None,
) -> DatasetCollectionConfig:
    """Create dataset collection filtered by sample type, objective, and microscope."""

    dataset_configs = load_all_dataset_configs()
    dataset_collection_names = []

    for dataset_config in dataset_configs:
        if sample_type is not None and dataset_config.live_or_fixed_sample != sample_type:
            continue

        if objective is not None and dataset_config.objective != objective:
            continue

        if microscope is not None and dataset_config.microscope != microscope:
            continue

        dataset_collection_names.append(dataset_config.name)

    dataset_collection = DatasetCollectionConfig(
        name=get_filtered_dataset_collection_name(sample_type, objective, microscope),
        description=get_filtered_dataset_collection_description(sample_type, objective, microscope),
        datasets=sorted(dataset_collection_names),
    )

    return dataset_collection


def validate_3d_flow_field_dataset_collection() -> None:
    """
    Validate dataset collection used as default for generate_3d_flow_field.

    Validation checks that each dataset in the collection is from an experiment
    with a single flow condition, and that the collection contains each of the
    datasets in the 'pca_reference' collection.
    """

    analysis_datasets = load_dataset_collection_config("3d_flow_field_analysis").datasets
    pca_reference_datasets = load_dataset_collection_config("pca_reference").datasets

    for dataset_name in analysis_datasets:
        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.flow_conditions) != 1:
            logger.error(
                "Dataset [ %s ] in [ 3d_flow_field_analysis ] has multiple flow conditions.",
                dataset_name,
            )

    for pca_dataset_name in pca_reference_datasets:
        if pca_dataset_name not in analysis_datasets:
            logger.error(
                "Dataset [ %s ] used for fitting PCA is not in the collection.",
                dataset_name,
            )


def validate_filtered_dataset_collection(
    sample_type: SampleType | None = None,
    objective: ObjectiveType | None = None,
    microscope: MicroscopeType | None = None,
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
