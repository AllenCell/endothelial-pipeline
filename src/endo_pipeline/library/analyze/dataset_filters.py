from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_unannotated_timepoints_for_position,
)


def get_frames_to_include(
    dataset_config: DatasetConfig,
    include_cell_piling: bool = False,
    include_not_steady_state: bool = False,
    annotations: list[TimepointAnnotation] | None = None,
) -> dict[int, list[int]]:
    """
    Get dict of frames to include per position based on annotations.

    Parameters
    ----------
    dataset_config
        Dataset configuration object.
    include_cell_piling
        Whether to include timepoints annotated as CELL_PILING.
    include_not_steady_state
        Whether to include timepoints annotated as NOT_STEADY_STATE.
    annotations
        List of annotations to filter by. If None, uses all annotations.

    Returns
    -------
    :
        Dictionary with position as key and list of timepoints to exclude as values.
    """
    # If no annotations provided, use all
    if annotations is None:
        annotations_to_parse = list(TimepointAnnotation)
    else:
        annotations_to_parse = annotations.copy()

    # if including cell piling timepoints, drop that annotation from the list
    if include_cell_piling:
        annotations_to_parse = [
            ann for ann in annotations_to_parse if ann != TimepointAnnotation.CELL_PILING
        ]

    # if including not steady state timepoints, drop that annotation from the list
    if include_not_steady_state:
        annotations_to_parse = [
            ann for ann in annotations_to_parse if ann != TimepointAnnotation.NOT_STEADY_STATE
        ]

    # loop over positions and get unannotated timepoints
    include_frames: dict[int, list[int]] = {}
    for pos in dataset_config.zarr_positions:
        include_frames[pos] = get_unannotated_timepoints_for_position(
            dataset_config=dataset_config,
            position=pos,
            annotations_to_exclude=annotations_to_parse,
        )
    return include_frames
