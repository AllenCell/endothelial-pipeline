from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE

LABEL_MAP = {
    Column.SegData.ALIGNMENT_DEG: "Alignment Relative to Flow (degrees)",
    Column.SegData.ALIGNMENT: "Alignment Relative to Flow (rad)",
    Column.SegData.ORIENTATION: "Cell Orientation (rad)",
    Column.SegData.ORIENTATION_DEG: "Cell Orientation (degrees)",
    Column.SegData.AREA_UM_SQ: "Cell Area (µm²)",
    Column.SegData.ASPECT_RATIO: "Aspect Ratio",
    Column.SegData.PERIMETER_UM: "Cell Perimeter (µm)",
    Column.SegData.ECCENTRICITY: "Cell Eccentricity",
    Column.SegData.CELL_FLUOR_MAX: "Cell Fluorescence Max (a.u.)",
    Column.SegData.CELL_FLUOR_MEAN: "Cell Fluorescence Mean (a.u.)",
    Column.SegData.CELL_FLUOR_MEDIAN: "Cell Fluorescence Median (a.u.)",
    Column.SegData.EDGE_FLUOR_MEAN: "Edge Fluorescence Mean (a.u.)",
    Column.SegData.NODE_FLUOR_MEAN: "Node Fluorescence Mean (a.u.)",
    Column.SegData.SOLIDITY: "Cell Solidity",
    Column.SegData.TIME_HRS: "Time (hours)",
    Column.SegData.TIME_MINS: "Time (minutes)",
    Column.TRACK_ID: "Track ID",
    Column.SegDataFilters.SMOOTHED_AREA_NORMD_DIFF: "Smoothed Area Difference (Normalized)",
    Column.TRACK_LENGTH: "Track Duration",
    Column.SegDataFilters.NUM_VALID_TIMEPOINTS_IN_TRACK: "Number of Valid Timepoints",
    Column.SegData.NUM_NEIGHBORS: "Number of Neighbors",
    Column.SegData.NUM_NUCLEI_IN_CROP: "Number of Nuclei in Crop",
    # dynamics features
    Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: "Cell Migration Angle (deg)",
    Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: "Cell Migration Speed (µm/min)",
    Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: "Nucleus Orientation Relative to Migration (deg)",
    Column.SegData.NUCLEI_POSITION_ANGLE_DEG: "Nucleus Orientation Relative to Flow Angle (deg)",
    Column.SegData.VECTOR_MEAN_FOR_CROP_MAGNITUDE: "Migration Coherence in Crop (Vector Mean Magnitude)",
    # filters
    Column.SegDataFilters.IS_EDGE_SEGMENTATION: "Filter: Touches Edge of Field of View",
    Column.SegDataFilters.IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE: "Filter: Smoothed Area Change Below Threshold",
    Column.SegDataFilters.IS_GREATER_THAN_MIN_TRACK_DURATION: "Filter: Exceeds Min Track Duration",
    Column.SegDataFilters.HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK: "Filter: Num Valid Points Exceeds Threshold",
    Column.SegDataFilters.IS_INCLUDED: "Filter: Passed All Filters",
    Column.SegDataFilters.IS_VALID_BBOX: "Annotation: Crop Box Limits are Within FOV",
    # annotation-based filters
    Column.Annotations.NOT_STEADY_STATE: "Annotation: Cell Population Not At Steady State",
    Column.Annotations.CELL_PILING: "Annotation: Significant Cell Piling in FOV",
    Column.Annotations.UNFED: "Annotation: Unfed Cells (More Than 3 Hours Since Fresh Media Introduced)",
    Column.Annotations.XY_SHIFT: "Annotation: Significant Change in XY position of FOV",
    Column.Annotations.Z_SHIFT: "Annotation: Significant Change in Z position of FOV",
    # Cell-centric DiffAE features and PCs
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:MAX_PCS_TO_COMPUTE]
    },
    Column.DiffAEData.POLAR_ANGLE: "PC Polar Angle",
    Column.DiffAEData.POLAR_RADIUS: "PC Polar Radius",
    Column.DiffAEData.PC3_FLIPPED: "PC Rho",
}

LABEL_MAP_GRID = {
    Column.SegData.TIME_HRS: "Time (hours)",
    Column.SegData.TIME_MINS: "Time (minutes)",
    # various PC values
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:MAX_PCS_TO_COMPUTE]
    },
    Column.DiffAEData.POLAR_ANGLE: "PC Polar Angle",
    Column.DiffAEData.POLAR_RADIUS: "PC Polar Radius",
    Column.DiffAEData.PC3_FLIPPED: "PC Rho",
    # filters
    Column.Annotations.AUTO_BF_SCOPE_ERROR: "Filter: Auto-detected Brightfield Microscope Error",
    Column.Annotations.AUTO_BF_TEMP_ARTIFACT: "Filter: Auto-detected Temporary Artifact",
    Column.Annotations.AUTO_GFP_SCOPE_ERROR: "Filter: Auto-detected GFP Channel Microscope Error",
    Column.Annotations.BF_SCOPE_ERROR: "Filter: Manually Annotated Brightfield Microscope Error",
    Column.Annotations.BF_TEMP_ARTIFACT: "Filter: Manually Annotated Temporary Artifact",
    Column.Annotations.GFP_SCOPE_ERROR: "Filter: Manually Annotated GFP Channel Microscope Error",
    Column.Annotations.CELL_PILING: "Filter: Manually Annotated Significant Cell Piling",
    Column.Annotations.NOT_STEADY_STATE: "Filter: Cells Not At Steady State",
    Column.Annotations.UNFED: "Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
    Column.Annotations.XY_SHIFT: "Filter: Significant Change in XY position of FOV",
    Column.Annotations.Z_SHIFT: "Filter: Significant Change in Z position of FOV",
    "optical_flow_mean_unit_vector_dt1": "Coherent Migration (Optical Flow Mean Unit Vector)",
    "optical_flow_angle_std_dt1": "Coherent Migration (Optical Flow Angle Std Dev)",
    "optical_flow_mean_speed_dt1": "Optical Flow Mean Speed",
    "optical_flow_std_speed_dt1": "Optical Flow Speed Std Dev",
}

RANGE_MAP = {
    "optical_flow_mean_unit_vector_dt1": (0, 1),
    "optical_flow_angle_std_dt1": (0, 4),
    "optical_flow_mean_speed_dt1": (0, 8),
    "optical_flow_std_speed_dt1": (0, 10),
}
