from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

DATE = "Date"
CELL_LINE = "Cell line"
SHEAR_STRESS = f"Shear stress (dyn/cm{Unicode.SQUARED})"
REPLICATE = "Replicate"
FOV_POSITIONS = "N FOV positions"
TIMEPOINTS = "N Timepoints per FOV position"
POSITION_FILTER = "Position filter (N removed)"
OUTLIER_FILTER = "Single timepoint filter (N removed)"
STEADY_STATE_FILTER = "Steady-state filter (N removed)"
DELAMINATION_FILTER = "Delamination filter (N removed)"
CROWDING_FILTER = "Cell crowding filter (N removed)"
SS_FOV = "Shear stress FOV dataset\n(N timepoints)"
SS_GRID = "Shear stress grid-based\nanalysis dataset (N patches)"
SS_CELL = "Shear stress cell-centered\nanalysis dataset (N patches)"
DA_FOV = "DiffAE FOV dataset\n(N timepoints)"
DA_GRID = "DiffAE grid-based\nanalysis dataset (N patches)"
DA_CELL = "DiffAE cell-centered\nanalysis dataset (N patches)"
PT_FOV = "FOV dataset\n(N timepoints)"
PT_GRID = "Grid-based analysis\ndataset (N patches)"
PT_CELL = "Cell-centered analysis\ndataset (N patches)"
NUC_SAMPLE = "Sample type"
NUC_PREDICTIONS = "N Nuclear segmentations"
SEGS_BEFORE = "N Cell segmentations"
SEGS_AFTER = "N cell segmentations\n(after segmentation QC filtering)"
IN_SHEAR_STRESS = "Included in Shear stress datasets"
IN_DIFFAE = "Included in DiffAE datasets"
"""Consistent column names to use across supplemental tables"""


MAX_CELL_CHARS = 16
"""Maximum number of characters used in supplemental table cell before applying text wrapping"""

TABLE_S1_SHEAR_STRESS = {
    "collection": "shear_stress",
    "title": "Table S1: Shear stress datasets",
    "figure_name": "table_s1_shear_stress_datasets",
    "sort_by": [SHEAR_STRESS, REPLICATE],
    "annotations_to_include": [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
        TimepointAnnotation.GFP_SCOPE_ERROR,
        TimepointAnnotation.NOT_STEADY_STATE,
        TimepointAnnotation.CELL_PILING,
    ],
    "columns": [
        (DATE, "date"),
        (CELL_LINE, "cell_line"),
        (SHEAR_STRESS, "shear_stress"),
        (REPLICATE, "replicate"),
        (FOV_POSITIONS, "fovs"),
        (TIMEPOINTS, "timepoints"),
        (POSITION_FILTER, "pos_removed"),
        (OUTLIER_FILTER, "outlier"),
        (STEADY_STATE_FILTER, "steady_state"),
        (CROWDING_FILTER, "piling"),
        (SS_FOV, "fov_total"),
        (SS_GRID, "grid"),
        (SS_CELL, "cell"),
    ],
}
"""Columns and settings to include in the shear stress supplemental table"""

TABLE_S2_DIFFAE = {
    "collection": "diffae_model_training",
    "title": "Table S2: DiffAE datasets",
    "figure_name": "table_s2_diffae_training_datasets",
    "sort_by": [SHEAR_STRESS, REPLICATE],
    "sort_bottom": ["0", "6 / 21", "24 / 6"],
    "annotations_to_include": [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
        TimepointAnnotation.GFP_SCOPE_ERROR,
        TimepointAnnotation.CELL_PILING,
    ],
    "columns": [
        (DATE, "date"),
        (CELL_LINE, "cell_line"),
        (SHEAR_STRESS, "shear_stress"),
        (REPLICATE, "replicate"),
        (FOV_POSITIONS, "fovs"),
        (TIMEPOINTS, "timepoints"),
        (POSITION_FILTER, "pos_removed"),
        (OUTLIER_FILTER, "outlier"),
        (CROWDING_FILTER, "piling"),
        (DA_FOV, "fov_total"),
        (DA_GRID, "grid"),
        (DA_CELL, "cell"),
    ],
}
"""Columns and settings to include in the DiffAE supplemental table"""


TABLE_S3_PERTURBATION = {
    "collection": "perturbation",
    "title": "Table S3: VE-cadherin Exon3Del perturbation datasets",
    "figure_name": "table_s3_perturbation_datasets",
    "sort_by": [CELL_LINE, REPLICATE],
    "annotations_to_include": [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
        TimepointAnnotation.GFP_SCOPE_ERROR,
        TimepointAnnotation.NOT_STEADY_STATE,
        TimepointAnnotation.CELL_PILING,
    ],
    "columns": [
        (DATE, "date"),
        (CELL_LINE, "cell_line"),
        (SHEAR_STRESS, "shear_stress"),
        (REPLICATE, "replicate"),
        (FOV_POSITIONS, "fovs"),
        (TIMEPOINTS, "timepoints"),
        (POSITION_FILTER, "pos_removed"),
        (OUTLIER_FILTER, "outlier"),
        (STEADY_STATE_FILTER, "adaptation"),
        (DELAMINATION_FILTER, "delamination"),
        (CROWDING_FILTER, "piling"),
        (PT_FOV, "fov_total"),
        (PT_GRID, "grid"),
    ],
}
"""Columns and settings to include in the perturbation supplemental table"""

TABLE_S4_NUCLEAR_LABELFREE = {
    "collection": "nuclear_labelfree_model_training",
    "title": "Table S4: Nuclear label-free model training datasets",
    "figure_name": "table_s4_nuclear_labelfree_training_datasets",
    "sort_by": [DATE],
    "columns": [
        (DATE, "date"),
        (CELL_LINE, "cell_line"),
        (SHEAR_STRESS, "shear_stress"),
        (NUC_SAMPLE, "sample_type"),
        (FOV_POSITIONS, "fovs"),
    ],
}
"""Columns and settings to include in the nuclear label-free supplemental table"""

TABLE_S5_SEGMENTATION = {
    "collection": "live_cdh5_seg_based_feat_datasets",
    "title": "Table S5: Segmentation summary",
    "figure_name": "table_s5_segmentation",
    "sort_by": [SHEAR_STRESS, REPLICATE],
    "sort_bottom": ["0", "6 / 21", "24 / 6"],
    "stats_fn": "get_seg_stats",
    "columns": [
        (DATE, "date"),
        (CELL_LINE, "cell_line"),
        (SHEAR_STRESS, "shear_stress"),
        (REPLICATE, "replicate"),
        (FOV_POSITIONS, "fovs"),
        (TIMEPOINTS, "timepoints"),
        (NUC_PREDICTIONS, "nuc_predictions"),
        (SEGS_BEFORE, "segs_before"),
        (SEGS_AFTER, "segs_after"),
        (IN_SHEAR_STRESS, "in_shear_stress"),
        (IN_DIFFAE, "in_diffae"),
    ],
}
"""Columns and settings to include in the segmentation supplemental table"""
