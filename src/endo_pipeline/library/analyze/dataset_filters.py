from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_unannotated_positions,
    get_unannotated_timepoints_for_position,
)


def get_include_frames(
    dataset_config: DatasetConfig,
    include_cell_piling: bool = True,
    include_not_steady_state: bool = True,
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

    Returns
    -------
    :
        Dictionary with position as key and list of timepoints to include as values.
    """


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
        annotations = [ann for ann in TimepointAnnotation if ann != TimepointAnnotation.CELL_PILING]

    # similar with steady state
    if not exclude_not_steady_state:
        if annotations is None:
            annotations = [
                ann for ann in TimepointAnnotation if ann != TimepointAnnotation.NOT_STEADY_STATE
            ]
        else:
            annotations = [
                ann for ann in annotations if ann != TimepointAnnotation.NOT_STEADY_STATE
            ]

    # parse dataset annotations to get timepoints to exclude per position
    exclude_frames = {
        pos: get_annotated_timepoints_for_position(dataset_config, pos, annotations=annotations)
        for pos in dataset_config.zarr_positions
    }

    return exclude_frames
