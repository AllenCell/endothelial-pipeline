from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE, ColumnName

LABEL_MAP = {
    "alignment_deg_rel_to_flow": "Alignment Relative to Flow (degrees)",
    "alignment_rel_to_flow": "Alignment Relative to Flow (rad)",
    "orientation": "Cell Orientation (rad)",
    "orientation_deg": "Cell Orientation (degrees)",
    "area (um**2)": "Cell Area (µm²)",
    "aspect_ratio": "Aspect Ratio",
    "perimeter (um)": "Cell Perimeter (µm)",
    "cell_eccentricity": "Cell Eccentricity",
    "cell_fluorescence_max (a.u.)": "Cell Fluorescence Max (a.u.)",
    "cell_fluorescence_mean (a.u.)": "Cell Fluorescence Mean (a.u.)",
    "cell_fluorescence_median (a.u.)": "Cell Fluorescence Median (a.u.)",
    "cell_solidity": "Cell Solidity",
    "time_hours": "Time (hours)",
    "time_minutes": "Time (minutes)",
    "track_id": "Track ID",
    "smoothed_area_normd_diff": "Smoothed Area Difference (Normalized)",
    "track_duration": "Track Duration",
    "num_valid_tp_per_track": "Number of Valid Timepoints",
    "number_of_neighbors": "Number of Neighbors",
    "num_nuclei_in_crop": "Number of Nuclei in Crop",
    # dynamics features
    "centroid_velocity_angle_deg": "Cell Migration Angle (deg)",
    "centroid_velocity_magnitude": "Cell Migration Speed (µm/min)",
    "cell_nuc_orientation_deg_rel_to_migration": "Nucleus Orientation Relative to Migration (deg)",
    "nuc_pos_rel_cell_angle_deg": "Nucleus Orientation Relative to Flow Angle (deg)",
    # filters
    "is_edge_segmentation": "Filter: Touches Edge of Field of View",
    "is_less_than_max_smoothed_area_normd_change": "Filter: Smoothed Area Change Below Threshold",
    "is_greater_than_min_track_duration": "Filter: Exceeds Min Track Duration",
    "has_more_than_min_num_valid_points_per_track": "Filter: Num Valid Points Exceeds Threshold",
    "is_included": "Filter: Passed All Filters",
    "bbox_is_in_bounds": "Annotation: Crop Box Limits are Within FOV",
    "not_steady_state": "Annotation: Cell Population Not At Steady State",
    "cell_piling": "Annotation: Significant Cell Piling in FOV",
    "unfed": "Annotation: Unfed Cells (More Than 3 Hours Since Fresh Media Introduced)",
    "xy_shift": "Annotation: Significant Change in XY position of FOV",
    "z_shift": "Annotation: Significant Change in Z position of FOV",
    # Cell-centric DiffAE features and PCs
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:MAX_PCS_TO_COMPUTE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
    ColumnName.PC3_FLIPPED.value: "PC Rho",
}

LABEL_MAP_GRID = {
    "time_hours": "Time (hours)",
    "time_minutes": "Time (minutes)",
    # various PC values
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:MAX_PCS_TO_COMPUTE]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
    ColumnName.PC3_FLIPPED.value: "PC Rho",
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
