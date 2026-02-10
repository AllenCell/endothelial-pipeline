from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

LABEL_MAP = {
    "alignment_deg_rel_to_flow": "Alignment Relative to Flow (degrees)",
    "alignment_rel_to_flow": "Alignment Relative to Flow (rad)",
    "cell_orientation": "Cell Orientation (rad)",
    "area (um**2)": "Cell Area (µm²)",
    "aspect_ratio": "Aspect Ratio",
    "perimeter (um)": "Cell Perimeter (µm)",
    "cell_eccentricity": "Cell Eccentricity",
    "cell_fluorescence_max (a.u.)": "Cell Fluorescence Max (a.u.)",
    "cell_fluorescence_mean (a.u.)": "Cell Fluorescence Mean (a.u.)",
    "cell_fluorescence_median (a.u.)": "Cell Fluorescence Median (a.u.)",
    "cell_solidity": "Cell Solidity",
    # "tid": "Track ID",
    "time_hours": "Time (hours)",
    "time_minutes": "Time (minutes)",
    "track_id": "Track ID",
    "smoothed_area_normd_diff": "Smoothed Area Difference (Normalized)",
    "track_duration": "Track Duration",
    "num_valid_tp_per_track": "Number of Valid Timepoints",
    "number_of_neighbors": "Number of Neighbors",
    # filters
    "is_edge_segmentation": "Filter: Touches Edge of Field of View",
    "is_less_than_max_smoothed_area_normd_change": "Filter: Smoothed Area Change Below Threshold",
    "is_greater_than_min_track_duration": "Filter: Exceeds Min Track Duration",
    "has_more_than_min_num_valid_points_per_track": "Filter: Num Valid Points Exceeds Threshold",
    "is_included": "Filter: Passed All Filters",
    # Cell-centric DiffAE features and PCs
    "pc_1": "PC 1",
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
}

LABEL_MAP_GRID = {
    "time_hours": "Time (hours)",
    "time_minutes": "Time (minutes)",
    "track_id": "Track ID",
    "duration": "Track Duration",
    # various PC values
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
    # filters
    "auto_bf_scope_error": "Filter: Auto-detected Brightfield Microscope Error",
    "auto_bf_temp_artifact": "Filter: Auto-detected Temporary Artifact",
    "auto_gfp_scope_error": "Filter: Auto-detected GFP Channel Microscope Error",
    "bf_scope_error": "Filter: Manually Annotated Brightfield Microscope Error",
    "bf_temp_artifact": "Filter: Manually Annotated Temporary Artifact",
    "gfp_scope_error": "Filter: Manually Annotated GFP Channel Microscope Error",
    "cell_piling": "Filter: Manually Annotated Significant Cell Piling",
    "not_steady_state": "Filter: Cells Not At Steady State",
    "unfed": "Filter: Unfed (More Than 3 Hours Since Fresh Media Introduced)",
    "xy_shift": "Filter: Significant Change in XY position of FOV",
    "z_shift": "Filter: Significant Change in Z position of FOV",
}
