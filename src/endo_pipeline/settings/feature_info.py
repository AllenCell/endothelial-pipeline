from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE, ColumnName
from endo_pipeline.settings.segmentation_feature_dataframes import ColumnNameSeg as ColNmSeg

LABEL_MAP = {
    ColNmSeg.ALIGNMENT_DEG: "Alignment Relative to Flow (degrees)",
    ColNmSeg.ALIGNMENT: "Alignment Relative to Flow (rad)",
    ColNmSeg.ORIENTATION: "Cell Orientation (rad)",
    ColNmSeg.AREA_UM_SQ: "Cell Area (µm²)",
    ColNmSeg.ASPECT_RATIO: "Aspect Ratio",
    ColNmSeg.PERIMETER_UM: "Cell Perimeter (µm)",
    ColNmSeg.ECCENTRICITY: "Cell Eccentricity",
    ColNmSeg.CELL_FLUOR_MAX: "Cell Fluorescence Max (a.u.)",
    ColNmSeg.CELL_FLUOR_MEAN: "Cell Fluorescence Mean (a.u.)",
    ColNmSeg.CELL_FLUOR_MEDIAN: "Cell Fluorescence Median (a.u.)",
    ColNmSeg.EDGE_FLUOR_MEAN: "Edge Fluorescence Mean (a.u.)",
    ColNmSeg.NODE_FLUOR_MEAN: "Node Fluorescence Mean (a.u.)",
    ColNmSeg.SOLIDITY: "Cell Solidity",
    ColNmSeg.TIME_HRS: "Time (hours)",
    ColNmSeg.TIME_MINS: "Time (minutes)",
    ColNmSeg.TRACK_ID: "Track ID",
    ColNmSeg.SMOOTHED_AREA_NORMD_DIFF: "Smoothed Area Difference (Normalized)",
    ColNmSeg.TRACK_LENGTH: "Track Duration",
    ColNmSeg.NUM_VALID_TIMEPOINTS_IN_TRACK: "Number of Valid Timepoints",
    ColNmSeg.NUM_NEIGHBORS: "Number of Neighbors",
    # filters
    ColNmSeg.IS_EDGE_SEGMENTATION: "Filter: Touches Edge of Field of View",
    ColNmSeg.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: "Filter: Smoothed Area Change Below Threshold",
    ColNmSeg.IS_GREATER_THAN_MIN_TRACK_DURATION: "Filter: Exceeds Min Track Duration",
    ColNmSeg.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: "Filter: Num Valid Points Exceeds Threshold",
    ColNmSeg.IS_INCLUDED: "Filter: Passed All Filters",
    # Cell-centric DiffAE features and PCs
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
    ColumnName.PC3_FLIPPED.value: "PC Rho",
}

LABEL_MAP_GRID = {
    ColNmSeg.TIME_HRS: "Time (hours)",
    ColNmSeg.TIME_MINS: "Time (minutes)",
    # various PC values
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:MAX_PCS_TO_COMPUTE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
    ColumnName.PC3_FLIPPED.value: "PC Rho",
    # filters
    ColNmSeg.AUTO_BF_SCOPE_ERROR: "Filter: Auto-detected Brightfield Microscope Error",
    ColNmSeg.AUTO_BF_TEMP_ARTIFACT: "Filter: Auto-detected Temporary Artifact",
    ColNmSeg.AUTO_GFP_SCOPE_ERROR: "Filter: Auto-detected GFP Channel Microscope Error",
    ColNmSeg.BF_SCOPE_ERROR: "Filter: Manually Annotated Brightfield Microscope Error",
    ColNmSeg.BF_TEMP_ARTIFACT: "Filter: Manually Annotated Temporary Artifact",
    ColNmSeg.GFP_SCOPE_ERROR: "Filter: Manually Annotated GFP Channel Microscope Error",
    ColNmSeg.CELL_PILING: "Filter: Manually Annotated Significant Cell Piling",
    ColNmSeg.NOT_STEADY_STATE: "Filter: Cells Not At Steady State",
    ColNmSeg.UNFED: "Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
    ColNmSeg.XY_SHIFT: "Filter: Significant Change in XY position of FOV",
    ColNmSeg.Z_SHIFT: "Filter: Significant Change in Z position of FOV",
}
