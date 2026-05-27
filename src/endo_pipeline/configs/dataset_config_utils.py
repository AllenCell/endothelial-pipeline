"""Methods for working with dataset configs."""

import logging
import re
from pathlib import Path

from endo_pipeline.configs import (
    DatasetConfig,
    FlowCondition,
    PositionAnnotation,
    ShearStressRegime,
    TimepointAnnotation,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


def get_regime_for_shear_stress(shear_stress: float) -> ShearStressRegime:
    """Get shear stress regime for given shear stress value."""

    for regime in ShearStressRegime:
        if regime.lower <= shear_stress <= regime.upper:
            return regime

    logger.error("No shear stress regime found for shear stress [ %f ]", shear_stress)
    raise ValueError(f"No shear stress regime found for shear stress [ {shear_stress} ]")


def get_shear_stress_label_for_dataset(
    dataset_config: DatasetConfig, flow_condition: FlowCondition | None = None
) -> str:
    """
    Get shear stress label for given dataset config.

    Label will be in format [ dataset date ] ([ shear stress value(s) ] dyn/cm^2).

    **Multi-flow condition datasets**:

    If the dataset has two flow conditions, then by default, the label will
    include both shear stress values (e.g. "0-12"). If *flow_condition* is
    provided, then only the shear stress value for that flow condition will be
    included in the label (e.g. "0" or "12").

    **Single-flow condition datasets**:

    If the dataset has only one flow condition, then the label will include that
    shear stress value regardless of whether *flow_condition* is provided or not
    (e.g. "12").
    """

    if flow_condition is not None and flow_condition not in dataset_config.flow_conditions:
        logger.error(
            "Provided flow condition [ %s ] is not in dataset [ %s ] flow conditions",
            flow_condition,
            dataset_config.name,
        )
        raise ValueError(
            f"Provided flow condition [ {flow_condition} ] is not "
            f"in dataset [ {dataset_config.name} ] flow conditions"
        )

    if len(dataset_config.flow_conditions) == 1:
        shear_stress_str = f"{dataset_config.flow_conditions[0].shear_stress}"
    elif len(dataset_config.flow_conditions) == 2:
        if flow_condition is None:
            shear_stresses = [
                condition.shear_stress for condition in dataset_config.flow_conditions
            ]
            shear_stress_str = "-".join(str(s) for s in shear_stresses)
        else:
            shear_stress_str = f"{flow_condition.shear_stress}"
    else:
        raise ValueError(
            f"Dataset [ {dataset_config.name} ] must have only one or "
            "two shear stress regimes to get shear stress label"
        )

    shear_stress_label = f"{dataset_config.date} ({shear_stress_str} dyn/cm{Unicode.SQUARED})"
    return shear_stress_label


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


def get_subset_of_timepoint_annotations(
    annotations_to_ignore: list[TimepointAnnotation],
) -> list[TimepointAnnotation]:
    """
    Get a subset of timepoint annotations to use for filtering data points.

    Parameters
    ----------
    annotations_to_ignore
        List of TimepointAnnotation enums to ignore when filtering.

    Returns
    -------
    :
        List of TimepointAnnotation enums to use for filtering.
    """

    annotations_all = list(TimepointAnnotation)

    annotations = [a for a in annotations_all if a not in annotations_to_ignore]

    return annotations


def get_annotated_positions(
    dataset: DatasetConfig, annotations: list[PositionAnnotation] | None = None
) -> list[int]:
    """Get all positions for given annotations."""

    annotated_positions: list[int] = []

    if dataset.position_annotations is None:
        logger.debug("Dataset [ %s ] does not have any annotated positions", dataset.name)
        return annotated_positions

    for annotation, positions in dataset.position_annotations.items():
        if annotations is None or annotation in annotations:
            annotated_positions.extend(positions)

    return annotated_positions


def get_unannotated_positions(
    dataset: DatasetConfig, annotations: list[PositionAnnotation] | None = None
) -> list[int]:
    """
    Get all positions without given annotations.

    If the provided list of annotations is empty, then all positions will be
    returned. If the provided list of annotations is None, then only positions
    without any annotations will be returned.
    """

    all_positions = dataset.zarr_positions
    annotated_positions = get_annotated_positions(dataset, annotations)

    return sorted(set(all_positions) - set(annotated_positions))


def get_annotated_timepoints_for_position(
    dataset: DatasetConfig, position: int, annotations: list[TimepointAnnotation] | None = None
) -> list[int]:
    """
    Get all timepoints with any of given annotations at the given position.

    If the provided list of annotations is empty, then no timepoints will be
    returned. If the provided list of annotations is None, the all timepoints
    with any annotations will be returned.
    """

    annotated_timepoints: list[int] = []

    if dataset.timepoint_annotations is None:
        logger.debug("Dataset [ %s ] does not have any annotated timepoints", dataset.name)
        return annotated_timepoints

    for annotation, positions in dataset.timepoint_annotations.items():
        if position not in positions:
            logger.debug(
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


def get_start_of_steady_state_for_position(dataset: DatasetConfig, position: int) -> int | None:
    """Get the timepoint at which the steady state starts for the given position."""

    if (
        dataset.timepoint_annotations is None
        or TimepointAnnotation.NOT_STEADY_STATE not in dataset.timepoint_annotations
    ):
        logger.warning("Dataset [ %s ] does not have any timepoint annotations", dataset.name)
        return None

    not_steady_state_timepoints = get_annotated_timepoints_for_position(
        dataset, position, annotations=[TimepointAnnotation.NOT_STEADY_STATE]
    )

    if len(not_steady_state_timepoints) == 0:
        logger.warning(
            "Dataset [ %s ] does not have any [ %s ] annotations for position [ %d ]",
            dataset.name,
            TimepointAnnotation.NOT_STEADY_STATE.value,
            position,
        )
        return None
    else:
        # steady state starts at the timepoint immediately after the last
        # annotated "not steady state" timepoint
        start_of_steady_state = max(not_steady_state_timepoints) + 1
        if start_of_steady_state >= dataset.duration:
            logger.warning(
                "Start of steady state [ %d ] is greater than or equal to "
                "dataset duration [ %d ] for dataset [ %s ] position [ %d ].",
                start_of_steady_state,
                dataset.duration,
                dataset.name,
                position,
            )
            return None
        return start_of_steady_state


def get_unannotated_timepoints_for_position(
    dataset: DatasetConfig, position: int, annotations: list[TimepointAnnotation] | None = None
) -> list[int]:
    """
    Get all timepoints without any of the given annotations at given position.

    If the provided list of annotations is empty, then all timepoints will be
    returned. If the provided list of annotations is None, then only timepoints
    without any annotations will be returned.
    """

    all_timepoints = range(dataset.duration)
    annotated_timepoints = get_annotated_timepoints_for_position(dataset, position, annotations)

    return sorted(set(all_timepoints) - set(annotated_timepoints))


def get_all_unannotated_timepoints(
    dataset: DatasetConfig, annotations: list[TimepointAnnotation] | None = None
) -> dict[int, list[int]]:
    """
    Get all timepoints without any of the given annotations for each position in dataset.

    Parameters
    ----------
    dataset: DatasetConfig
    annotations: list[TimepointAnnotation] | None
        Annotations to consider. If annotations is [], then all timepoints will be
        returned. If annotations is None, then only timepoints without any annotations
        will be returned.
    """

    return {
        position: get_unannotated_timepoints_for_position(dataset, position, annotations)
        for position in dataset.zarr_positions
    }
