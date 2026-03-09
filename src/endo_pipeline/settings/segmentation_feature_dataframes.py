from enum import StrEnum

from endo_pipeline.settings import ColumnName as ColumnNameDiffAE


class ColumnNameSeg(StrEnum):
    """Dataframe column names used in segmentation-based feature dataframes."""

    # dataset and segmentation information columns
    DATASET = ColumnNameDiffAE.DATASET
    """Column name for dataset name."""

    POSITION = ColumnNameDiffAE.POSITION
    """Column name for position identifier."""

    TIMEPOINT = ColumnNameDiffAE.TIMEPOINT
    """Column name for timepoint (frame number)."""

    TRACK_ID = ColumnNameDiffAE.TRACK_ID

    LABEL = "label"

    # feature columns
    TIME_HRS = "time_hours"
    TIME_MINS = "time_minutes"
    ORIENTATION = "cell_orientation"
    AREA = "area_px2"
    PERIMETER = "perimeter"
    EDGE_FLUOR = "edge_fluorescence"
    NODE_FLUOR = "node_fluorescence"

    CELL_FLUOR_MEAN = "cell_fluorescence_mean"
    EDGE_FLUOR_MEAN = "edge_fluorescence_mean"
    NODE_FLUOR_MEAN = "node_fluorescence_mean"
