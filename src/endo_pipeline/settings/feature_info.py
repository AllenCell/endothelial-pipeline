from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

num_pcs_to_analyze = 10
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
    "tid": "Track ID",
    "time_hours": "Time (hours)",
    "time_minutes": "Time (minutes)",
    "track_id": "Track ID",
    "smoothed_area_normd_diff": "Smoothed Area Difference (Normalized)",
    "track_duration": "Track Duration",
    "num_valid_tp_per_track": "Number of Valid Timepoints",
    "number_of_neighbors": "Number of Neighbors",
    "num_nuclei_in_crop": "Number of Nuclei in Crop",
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
    # Cell-centric DiffAE features and PCs
    **{
        f"{pc_col}": f"{pc_col.replace('pc_', 'PC ')}"
        for pc_col in DIFFAE_PC_COLUMN_NAMES[:num_pcs_to_analyze]
    },
    ColumnName.POLAR_ANGLE.value: "PC Polar Angle",
    ColumnName.POLAR_RADIUS.value: "PC Polar Radius",
}
