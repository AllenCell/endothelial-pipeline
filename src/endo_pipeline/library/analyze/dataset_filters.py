from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_positions,
    get_annotated_timepoints_for_position,
)


def get_include_positions(dataset_config: DatasetConfig) -> list[int]:
    """Get list of positions to include based on annotations."""
    exclude_positions = get_annotated_positions(dataset_config)
    only_include_positions = sorted(set(dataset_config.zarr_positions) - set(exclude_positions))
    return only_include_positions


def get_exclude_frames(
    dataset_config: DatasetConfig,
    exclude_cell_piling: bool = False,
    exclude_not_steady_state: bool = False,
) -> dict[int, list[int]]:
    """
    Get dict of frames to exclude per position based on annotations.

    Parameters
    ----------
    dataset_config
        Dataset configuration object.
    exclude_cell_piling
        Whether to exclude timepoints annotated as CELL_PILING.
    exclude_not_steady_state
        Whether to exclude timepoints annotated as NOT_STEADY_STATE.

    Returns
    -------
    :
        Dictionary with position as key and list of timepoints to exclude as values.
    """
    # if exclude_cell_piling is True, then get all annotated timepoints
    # else, get timepoints for all annotations except CELL_PILING
    annotations = None  # default to all annotations
    if not exclude_cell_piling:
        annotations = [ann for ann in TimepointAnnotation if "PILING" not in ann.name]

    # similar with steady state
    if not exclude_not_steady_state:
        if annotations is None:
            annotations = [ann for ann in TimepointAnnotation if "STEADY_STATE" not in ann.name]
        else:
            annotations = [ann for ann in annotations if "STEADY_STATE" not in ann.name]

    # parse dataset annotations to get timepoints to exclude per position
    exclude_frames = {
        pos: get_annotated_timepoints_for_position(dataset_config, pos, annotations=annotations)
        for pos in dataset_config.zarr_positions
    }

    return exclude_frames
