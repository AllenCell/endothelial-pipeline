# %% Import libraries and set preliminary variables
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_datasets_in_collection,
    get_unannotated_positions,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
    CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp-fig-tables")

# -- Table column names --
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
NUC_PREDICTIONS = "Nuclear segmentations"
SEGS_BEFORE = "Cell segmentations"
SEGS_AFTER = "Cell and nuclear segmentations\n(after filtering)"
TRACKS_BEFORE = "Cell trajectories"
TRACKS_AFTER = "Cell trajectories\n(after filtering)"
IN_SHEAR_STRESS = "Included in\nshear stress"
IN_DIFFAE = "Included in\nDiffAE"

MAX_CELL_CHARS = 16

grid_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["grid_based"])
cell_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"])
seg_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME)

_shear_stress_datasets = set(get_datasets_in_collection("shear_stress"))
_diffae_datasets = set(get_datasets_in_collection("diffae_model_training"))


# %% Collect all filter/patch statistics for a single dataset
def _get_dataset_stats(dataset_name: str) -> dict:
    dataset_config = load_dataset_config(dataset_name)
    # only get the annotations for positions that are unannotated
    # (i.e. not removed by the position filter)
    unannotated = get_unannotated_positions(dataset_config)

    # Count annotated timepoints across all unannotated positions
    n_outlier = sum(
        len(
            get_annotated_timepoints_for_position(
                dataset_config,
                pos,
                annotations=[
                    TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
                    TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
                    TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
                    TimepointAnnotation.BF_SCOPE_ERROR,
                    TimepointAnnotation.BF_TEMP_ARTIFACT,
                    TimepointAnnotation.GFP_SCOPE_ERROR,
                ],
            )
        )
        for pos in unannotated
    )
    n_steady = sum(
        len(
            get_annotated_timepoints_for_position(
                dataset_config, pos, annotations=[TimepointAnnotation.NOT_STEADY_STATE]
            )
        )
        for pos in unannotated
    )
    n_piling = sum(
        len(
            get_annotated_timepoints_for_position(
                dataset_config, pos, annotations=[TimepointAnnotation.CELL_PILING]
            )
        )
        for pos in unannotated
    )

    # The NOT_STEADY_STATE annotation can contain multiple ranges per position.
    # The first range represents the adaptation to shear stress period (steady state filter)
    # For perturbation datasets the subsequent range represents delamination (delamination filter)
    n_adaptation_timepoints = 0
    n_delamination_timepoints = 0
    for pos in unannotated:
        not_steady_state_ranges = (dataset_config.timepoint_annotations or {}).get(
            TimepointAnnotation.NOT_STEADY_STATE, {}
        )
        for range_idx, timepoint_range in enumerate(not_steady_state_ranges.get(pos, [])):
            n_timepoints_in_range = (
                (timepoint_range[1] - timepoint_range[0] + 1)
                if isinstance(timepoint_range, (list, tuple))
                else 1
            )
            if range_idx == 0:
                n_adaptation_timepoints += n_timepoints_in_range
            else:
                n_delamination_timepoints += n_timepoints_in_range

    # Patch counts from filtered feature manifests (each row = one patch)
    n_grid = n_cell = None
    try:
        df_grid_location = get_dataframe_location_for_dataset(grid_manifest, dataset_name)
        n_grid = len(load_dataframe(df_grid_location))
    except (FileNotFoundError, KeyError):
        pass
    try:
        df_cell_centered_location = get_dataframe_location_for_dataset(cell_manifest, dataset_name)
        n_cell = len(load_dataframe(df_cell_centered_location))
    except (FileNotFoundError, KeyError):
        pass

    return {
        "date": dataset_config.date,
        "cell_line": "\n".join(dataset_config.cell_lines) if dataset_config.cell_lines else "",
        "shear_stress": " / ".join(
            str(fc.shear_stress_bin) for fc in dataset_config.flow_conditions
        ),
        "replicate": dataset_config.replicate_number,
        "fovs": len(dataset_config.zarr_positions),
        "timepoints": dataset_config.duration,
        "pos_removed": len(dataset_config.zarr_positions) - len(unannotated),
        "outlier": n_outlier,
        "steady_state": n_steady,
        "adaptation": n_adaptation_timepoints,
        "delamination": n_delamination_timepoints,
        "piling": n_piling,
        "fov_total": len(unannotated) * dataset_config.duration - n_outlier - n_steady - n_piling,
        "grid": n_grid,
        "cell": n_cell,
        "sample_type": getattr(dataset_config, "live_or_fixed_sample", ""),
    }


# %% Collect segmentation statistics for a single dataset
def _get_seg_stats(dataset_name: str) -> dict:
    dataset_config = load_dataset_config(dataset_name)

    df_cell_centered_location = get_dataframe_location_for_dataset(cell_manifest, dataset_name)
    df = load_dataframe(df_cell_centered_location)[
        [
            Column.DATASET,
            Column.POSITION,
            Column.TIMEPOINT,
            Column.TRACK_ID,
            Column.SegData.NUM_NUCLEI_AT_TIMEPOINT,
            Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
            Column.SegData.NUM_TRACKS_AFTER_FILTERING,
            Column.SegDataFilters.IS_INCLUDED,
        ]
    ]

    # Nuclear segmentations: one value per (timepoint, position), sum across all
    n_nuc = int(
        df.groupby([Column.TIMEPOINT, Column.POSITION])[Column.SegData.NUM_NUCLEI_AT_TIMEPOINT]
        .first()
        .sum()
    )

    # Cell segmentations before filtering: one value per (timepoint, position)
    n_seg_before = int(
        df.groupby([Column.TIMEPOINT, Column.POSITION])[Column.SegData.NUM_TRACKS_BEFORE_FILTERING]
        .first()
        .sum()
    )

    # Total unique tracks before filtering
    n_tracks_before = df.groupby(Column.POSITION)[Column.TRACK_ID].nunique().sum()

    # Apply annotation-based filtering
    df = filter_dataframe_by_annotations(
        df, dataset_config, timepoint_annotations=ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS
    )

    # Cell segmentations after filtering
    n_seg_after = int(
        df.dropna(subset=[Column.SegData.NUM_TRACKS_AFTER_FILTERING])
        .groupby([Column.TIMEPOINT, Column.POSITION])[Column.SegData.NUM_TRACKS_AFTER_FILTERING]
        .first()
        .sum()
    )

    # Tracks after filtering (only rows where IS_INCLUDED is True)
    included = df[df[Column.SegDataFilters.IS_INCLUDED]]
    n_tracks_after = included.groupby(Column.POSITION)[Column.TRACK_ID].nunique().sum()

    return {
        "date": dataset_config.date,
        "cell_line": "\n".join(dataset_config.cell_lines) if dataset_config.cell_lines else "",
        "shear_stress": " / ".join(
            str(fc.shear_stress_bin) for fc in dataset_config.flow_conditions
        ),
        "replicate": dataset_config.replicate_number,
        "fovs": len(dataset_config.zarr_positions),
        "timepoints": dataset_config.duration,
        "nuc_predictions": n_nuc,
        "segs_before": n_seg_before,
        "segs_after": n_seg_after,
        "tracks_before": n_tracks_before,
        "tracks_after": n_tracks_after,
        "in_shear_stress": "X" if dataset_name in _shear_stress_datasets else "",
        "in_diffae": "X" if dataset_name in _diffae_datasets else "",
    }


# %% Table definitions — each table is a collection + ordered list of (column_name, stats_key) pairs
TABLES = [
    {
        "collection": "shear_stress",
        "title": "Table 1A: Shear stress datasets",
        "figure_name": "table_1a_shear_stress_datasets",
        "sort_by": [SHEAR_STRESS, REPLICATE],
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
    },
    {
        "collection": "diffae_model_training",
        "title": "Table 1B: DiffAE datasets",
        "figure_name": "table_1b_diffae_training_datasets",
        "sort_by": [SHEAR_STRESS, REPLICATE],
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
    },
    {
        "collection": "perturbation",
        "title": "Table 1C: VE-cadherin Exon3Del perturbation datasets",
        "figure_name": "table_1c_perturbation_datasets",
        "sort_by": [CELL_LINE, REPLICATE],
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
    },
    {
        "collection": "nuclear_labelfree_model_training",
        "title": "Table 1D: Nuclear label-free model training datasets",
        "figure_name": "table_1d_nuclear_labelfree_training_datasets",
        "sort_by": [DATE],
        "columns": [
            (DATE, "date"),
            (CELL_LINE, "cell_line"),
            (SHEAR_STRESS, "shear_stress"),
            (REPLICATE, "replicate"),
            (NUC_SAMPLE, "sample_type"),
            (FOV_POSITIONS, "fovs"),
        ],
    },
    {
        "collection": "live_cdh5_seg_based_feat_datasets",
        "title": "Supplemental Table: Segmentation summary",
        "figure_name": "supp_table_segmentation",
        "sort_by": [SHEAR_STRESS, REPLICATE],
        "stats_fn": _get_seg_stats,
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
            (TRACKS_BEFORE, "tracks_before"),
            (TRACKS_AFTER, "tracks_after"),
            (IN_SHEAR_STRESS, "in_shear_stress"),
            (IN_DIFFAE, "in_diffae"),
        ],
    },
]


# %% Render a dataframe as a styled matplotlib table and save as SVG
def _render_table_svg(df: pd.DataFrame, title: str, figure_name: str) -> None:
    fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH, MAX_FIGURE_HEIGHT))
    ax.axis("off")

    col_labels = df.columns.tolist()
    n_cols = len(col_labels)
    n_rows = len(df)

    # Wrap long cell values so they fit in columns
    cell_text = [
        [textwrap.fill(str(v), width=MAX_CELL_CHARS) for v in row] for row in df.values.tolist()
    ]

    # Compute column widths proportional to max character width in each column
    max_chars = [
        max(max(len(ln) for ln in cell_text[r][c].split("\n")) for r in range(n_rows))
        for c in range(n_cols)
    ]
    total = sum(max(c, 3) for c in max_chars)
    col_widths = [max(c, 3) / total for c in max_chars]

    # Create table
    tbl = ax.table(
        cellText=cell_text,
        colLabels=None,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.scale(1.0, 1.4)

    # Style cells
    base_height = tbl[0, 0].get_height()
    for row_idx in range(n_rows):
        max_lines = max(v.count("\n") + 1 for v in cell_text[row_idx])
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            cell.set_text_props(fontsize=FONTSIZE_XSMALL)
            cell.set_edgecolor("gray")
            if max_lines > 1:
                cell.set_height(base_height * max_lines)
            if row_idx % 2 == 0:
                cell.set_facecolor("#f2f2f2")

    # Rotated column labels above each column center
    for col_idx, label in enumerate(col_labels):
        x = sum(col_widths[:col_idx]) + col_widths[col_idx] / 2
        ax.text(
            x,
            1.0,
            label,
            transform=ax.transAxes,
            fontsize=FONTSIZE_XSMALL,
            fontweight="bold",
            rotation=45,
            ha="left",
            va="bottom",
        )

    plt.title(title, fontsize=10, weight="bold", pad=60, loc="left")
    tbl.set_in_layout(True)
    save_plot_to_path(
        figure=fig,
        output_path=save_dir,
        figure_name=figure_name,
        file_format=".svg",
        bbox_inches="tight",
    )


# %% Build and render all tables
for table in TABLES:
    stats_fn = table.get("stats_fn", _get_dataset_stats)
    dataset_names = get_datasets_in_collection(table["collection"])
    stats = [stats_fn(name) for name in tqdm(dataset_names, desc=table["title"])]
    cols = table["columns"]
    df = pd.DataFrame([{col: s[key] for col, key in cols} for s in stats])

    # Numeric sort for shear stress column
    sort_cols = table["sort_by"]
    if sort_cols[0] == SHEAR_STRESS:
        df["_sort"] = df[SHEAR_STRESS].str.split(" / ").str[0].astype(int)
        df = df.sort_values(by=["_sort"] + sort_cols[1:], ignore_index=True).drop(columns="_sort")
    else:
        df = df.sort_values(by=sort_cols, ignore_index=True)

    _render_table_svg(df, table["title"], table["figure_name"])

# %%
