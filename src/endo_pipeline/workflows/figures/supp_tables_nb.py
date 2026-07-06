"""
**Supplemental Tables**. Tables summarizing per-dataset statistics.

#supp-tables

| Table | Description                                | Type           |
| ----- | ------------------------------------------ | -------------- |
| 1     | Shear stress datasets                      | `dataset`      |
| 2     | DiffAE model training datasets             | `dataset`      |
| 3     | VE-cadherin Exon3Del perturbation datasets | `dataset`      |
| 4     | Nuclear label-free model training datasets | `dataset`      |
| 5     | Segmentation summary                       | `segmentation` |

## Example usage

To run the table workflow:

```bash
uv run endopipe supp-tables
```

## Table types

Table are one of two types:

- `dataset` = dataset-level statistics including number of timepoints, number of positions,
  filtering, and number of patches
- `segmentation` = segmentation-level statistics including number of cell segmentations, number of
  nuclear segmentations, and filtering
"""

# %%
from matplotlib import pyplot as plt

from endo_pipeline.configs import TimepointAnnotation, get_datasets_in_collection
from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.supp_tables import create_supp_table
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.unicode import UnicodeCharacters
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

# %%

DATE = "Date"
CELL_LINE = "Cell line"
SHEAR_STRESS = f"Shear stress (dyn/cm{UnicodeCharacters.SQUARED})"
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

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_tables")
cell_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"])
seg_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME)
shear_stress_datasets = set(get_datasets_in_collection("shear_stress"))
diffae_datasets = set(get_datasets_in_collection("diffae_model_training"))

# %%
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

TABLE_S2_DIFFAE = {
    "collection": "diffae_model_training",
    "title": "Table S2: DiffAE datasets",
    "figure_name": "table_s2_diffae_training_datasets",
    "sort_by": [SHEAR_STRESS, REPLICATE],
    "sort_bottom": ["0", "6, 21", "24, 6"],
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

TABLE_S5_SEGMENTATION = {
    "collection": "live_cdh5_seg_based_feat_datasets",
    "title": "Table S5: Segmentation summary",
    "figure_name": "table_s5_segmentation",
    "sort_by": [SHEAR_STRESS, REPLICATE],
    "sort_bottom": ["0", "6, 21", "24, 6"],
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

# %%
create_supp_table(TABLE_S1_SHEAR_STRESS, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S2_DIFFAE, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S3_PERTURBATION, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S4_NUCLEAR_LABELFREE, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(
    TABLE_S5_SEGMENTATION,
    cell_manifest=seg_manifest,
    save_dir=save_dir,
    shear_stress_datasets=shear_stress_datasets,
    diffae_datasets=diffae_datasets,
)

# %%
