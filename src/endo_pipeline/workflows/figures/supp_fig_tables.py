# %% Import libraries and set preliminary variables
import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_datasets_in_collection,
    get_unannotated_positions,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_table1")

# -- Table column names --
DATE = "Date"
CELL_LINE = "Cell line"
SHEAR_STRESS = f"Shear stress (dyn/cm{Unicode.SQUARED})"
REPLICATE = "Replicate"
FOV_POSITIONS = "FOV positions"
TIMEPOINTS = "Timepoints per\nFOV position"
POSITION_FILTER = "Position filter\n(N removed)"
OUTLIER_FILTER = "Single timepoint\nfilter (N removed)"
STEADY_STATE_FILTER = "Steady-state filter\n(N removed)"
DELAMINATION_FILTER = "Delamination filter\n(N removed)"
CROWDING_FILTER = "Cell crowding filter\n(N removed)"
SS_FOV = "Shear stress FOV dataset\n(N total timepoints)"
SS_GRID = "Shear stress grid-based\nanalysis dataset (N patches)"
SS_CELL = "Shear stress cell-centered\nanalysis dataset (N patches)"
DA_FOV = "DiffAE FOV dataset\n(N total timepoints)"
DA_GRID = "DiffAE grid-based\nanalysis dataset (N patches)"
DA_CELL = "DiffAE cell-centered\nanalysis dataset (N patches)"
PT_FOV = "FOV dataset\n(N total timepoints)"
PT_GRID = "Grid-based analysis\ndataset (N patches)"
PT_CELL = "Cell-centered analysis\ndataset (N patches)"
NUC_SAMPLE = "Sample type"

OUTLIER_ANNOTATIONS = [
    TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
    TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
    TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
    TimepointAnnotation.BF_SCOPE_ERROR,
    TimepointAnnotation.BF_TEMP_ARTIFACT,
    TimepointAnnotation.GFP_SCOPE_ERROR,
]

grid_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["grid_based"])
cell_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"])


# %% Collect all values for a single dataset
def _get_dataset_stats(dataset_name: str) -> dict:
    """Return all filter/patch statistics for a dataset."""
    cfg = load_dataset_config(dataset_name)
    unannotated = get_unannotated_positions(cfg)

    def _count(annotations):
        return sum(
            len(get_annotated_timepoints_for_position(cfg, pos, annotations=annotations))
            for pos in unannotated
        )

    n_outlier = _count(OUTLIER_ANNOTATIONS)
    n_steady = _count([TimepointAnnotation.NOT_STEADY_STATE])
    n_piling = _count([TimepointAnnotation.CELL_PILING])

    # Separate initial not-steady-state from later delamination ranges
    n_adapt = n_delam = 0
    for pos in unannotated:
        tp_ann = cfg.timepoint_annotations or {}
        nss = tp_ann.get(TimepointAnnotation.NOT_STEADY_STATE, {})
        for i, rng in enumerate(nss.get(pos, [])):
            n = (rng[1] - rng[0] + 1) if isinstance(rng, (list, tuple)) else 1
            if i == 0:
                n_adapt += n
            else:
                n_delam += n

    # Patch counts from manifests
    n_grid = n_cell = None
    try:
        df = load_dataframe(get_dataframe_location_for_dataset(grid_manifest, dataset_name))
        n_grid = df[Column.CROP_INDEX].nunique()
    except (FileNotFoundError, KeyError):
        pass
    try:
        df = load_dataframe(get_dataframe_location_for_dataset(cell_manifest, dataset_name))
        n_cell = df[Column.CROP_INDEX].nunique()
    except (FileNotFoundError, KeyError):
        pass

    return {
        "date": cfg.date,
        "cell_line": "\n".join(cfg.cell_lines) if cfg.cell_lines else "",
        "shear_stress": " / ".join(str(fc.shear_stress_bin) for fc in cfg.flow_conditions),
        "replicate": cfg.replicate_number,
        "fovs": len(cfg.zarr_positions),
        "timepoints": cfg.duration,
        "pos_removed": len(cfg.zarr_positions) - len(unannotated),
        "outlier": n_outlier,
        "steady_state": n_steady,
        "adaptation": n_adapt,
        "delamination": n_delam,
        "piling": n_piling,
        "fov_total": len(unannotated) * cfg.duration - n_outlier - n_steady - n_piling,
        "grid": n_grid,
        "cell": n_cell,
        "sample_type": getattr(cfg, "live_or_fixed_sample", ""),
        "microscope": getattr(cfg, "microscope", ""),
        "objective": getattr(cfg, "objective", ""),
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
]


# %% Render a dataframe as a styled matplotlib table and save as SVG
import textwrap

MAX_CELL_CHARS = 16  # wrap cell text after this many characters


def _wrap_text(text: str, width: int = MAX_CELL_CHARS) -> str:
    """Wrap each line of text independently, preserving existing newlines."""
    lines = str(text).split("\n")
    wrapped = [textwrap.fill(line, width=width) for line in lines]
    return "\n".join(wrapped)


def _render_table_svg(df: pd.DataFrame, title: str, figure_name: str) -> None:
    fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH, MAX_FIGURE_HEIGHT))
    ax.axis("off")

    col_labels = df.columns.tolist()
    n_cols = len(col_labels)
    n_rows = len(df)

    # Wrap long cell values so they fit in columns
    cell_text = [[_wrap_text(v) for v in row] for row in df.astype(str).values.tolist()]

    # Compute column widths proportional to max character width in each column
    max_chars_per_col = []
    for col_idx in range(n_cols):
        max_len = max(
            max(len(line) for line in cell_text[row_idx][col_idx].split("\n"))
            for row_idx in range(n_rows)
        )
        max_chars_per_col.append(max(max_len, 3))  # minimum width of 3 chars
    total_chars = sum(max_chars_per_col)
    col_widths = [c / total_chars for c in max_chars_per_col]

    # Create table at top of axes, full width
    tbl = ax.table(
        cellText=cell_text,
        colLabels=None,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.scale(1.0, 1.4)

    # Style cells and adjust row heights for multiline content
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

    # Position column labels above each column center
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
    stats = [_get_dataset_stats(name) for name in get_datasets_in_collection(table["collection"])]
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
