import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_datasets_in_collection,
    get_unannotated_positions,
    get_unannotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.unicode import UnicodeCharacters
from endo_pipeline.settings.workflow_defaults import ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS

MAX_CELL_CHARS = 16
"""Maximum number of characters used in supplemental table cell before applying text wrapping"""

SHEAR_STRESS = f"Shear stress (dyn/cm{UnicodeCharacters.SQUARED})"


def get_dataset_stats(
    dataset_name: str,
    cell_manifest: DataframeManifest,
    annotations_to_include: list[TimepointAnnotation] | None = None,
) -> dict:
    """Compute dataset-level statistics for a supplemental table row.

    Counts annotated timepoints removed by each filter stage (outlier,
    steady-state, delamination, cell piling) and tallies the remaining
    FOV-level, grid-based, and cell-centered patch counts.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to summarize.
    cell_manifest : DataframeManifest
        Manifest mapping dataset names to filtered cell-centered feature
        dataframe locations.
    annotations_to_include : list of TimepointAnnotation or None, optional
        Annotations used to determine which timepoints are excluded when
        computing the final FOV total. If None, all annotation types in the
        dataset config are used.

    Returns
    -------
    dict
        Keys correspond to the stat keys referenced in table column
        definitions (e.g. "date", "fov_total", "grid", "cell").
    """
    dataset_config = load_dataset_config(dataset_name)
    # only get the annotations for positions that are unannotated
    # (i.e. not removed by the position filter)
    unannotated_positions = get_unannotated_positions(dataset_config)

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
        for pos in unannotated_positions
    )
    n_steady = sum(
        len(
            get_annotated_timepoints_for_position(
                dataset_config, pos, annotations=[TimepointAnnotation.NOT_STEADY_STATE]
            )
        )
        for pos in unannotated_positions
    )
    n_piling = sum(
        len(
            get_annotated_timepoints_for_position(
                dataset_config, pos, annotations=[TimepointAnnotation.CELL_PILING]
            )
        )
        for pos in unannotated_positions
    )

    # The NOT_STEADY_STATE annotation can contain multiple ranges per position.
    # The first range represents the adaptation to shear stress period (steady state filter)
    # For perturbation datasets the subsequent range represents delamination (delamination filter)
    n_adaptation_timepoints = 0
    n_delamination_timepoints = 0
    for pos in unannotated_positions:
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
    n_cell = None
    try:
        df_cell_centered_location = get_dataframe_location_for_dataset(cell_manifest, dataset_name)
        n_cell = len(load_dataframe(df_cell_centered_location))
    except (FileNotFoundError, KeyError):
        pass

    # Total unannotated timepoints (accounts for overlapping annotations)
    fov_total = sum(
        len(
            get_unannotated_timepoints_for_position(
                dataset_config, pos, annotations=annotations_to_include
            )
        )
        for pos in unannotated_positions
    )

    n_grid = fov_total * 36  # 6x6 grid of patches per FOV

    return {
        "date": dataset_config.date,
        "cell_line": "\n".join(dataset_config.cell_lines) if dataset_config.cell_lines else "",
        "shear_stress": ", ".join(
            str(fc.shear_stress_bin) for fc in dataset_config.flow_conditions
        ),
        "replicate": dataset_config.replicate_number,
        "fovs": len(dataset_config.zarr_positions),
        "timepoints": dataset_config.duration,
        "pos_removed": len(dataset_config.zarr_positions) - len(unannotated_positions),
        "outlier": n_outlier,
        "steady_state": n_steady,
        "adaptation": n_adaptation_timepoints,
        "delamination": n_delamination_timepoints,
        "piling": n_piling,
        "fov_total": fov_total,
        "grid": n_grid,
        "cell": n_cell,
        "sample_type": getattr(dataset_config, "live_or_fixed_sample", ""),
    }


def get_seg_stats(
    dataset_name: str,
    cell_manifest: DataframeManifest,
    shear_stress_datasets: set,
    diffae_datasets: set,
) -> dict:
    """Compute segmentation-level statistics for a supplemental table row.

    Summarises nuclear segmentation counts, cell segmentation counts before
    and after annotation-based filtering, and flags membership in the shear
    stress and DiffAE dataset collections.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to summarize.
    cell_manifest : DataframeManifest
        Manifest mapping dataset names to cell-centered feature dataframe
        locations.
    shear_stress_datasets : set
        Set of dataset names in the shear stress collection (used to mark
        the "included in shear stress" column).
    diffae_datasets : set
        Set of dataset names in the DiffAE collection (used to mark the
        "included in DiffAE" column).

    Returns
    -------
    dict
        Keys correspond to the stat keys referenced in the segmentation
        table column definitions (e.g. "nuc_predictions", "segs_before").
    """
    dataset_config = load_dataset_config(dataset_name)

    df_cell_centered_location = get_dataframe_location_for_dataset(cell_manifest, dataset_name)
    cols_to_load = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.SegData.NUM_NUCLEI_AT_TIMEPOINT,
        Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
        Column.SegData.NUM_TRACKS_AFTER_FILTERING,
    ]
    df = load_dataframe(df_cell_centered_location, delay=True)[cols_to_load].compute()

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

    return {
        "date": dataset_config.date,
        "cell_line": "\n".join(dataset_config.cell_lines) if dataset_config.cell_lines else "",
        "shear_stress": ", ".join(
            str(fc.shear_stress_bin) for fc in dataset_config.flow_conditions
        ),
        "replicate": dataset_config.replicate_number,
        "fovs": len(dataset_config.zarr_positions),
        "timepoints": dataset_config.duration,
        "nuc_predictions": n_nuc,
        "segs_before": n_seg_before,
        "segs_after": n_seg_after,
        "in_shear_stress": "X" if dataset_name in shear_stress_datasets else "",
        "in_diffae": "X" if dataset_name in diffae_datasets else "",
    }


def create_supp_table(
    table: dict,
    cell_manifest: DataframeManifest,
    save_dir: Path,
    shear_stress_datasets: set | None = None,
    diffae_datasets: set | None = None,
) -> None:
    """Generate a supplemental table figure and save as SVG.

    Collects per-dataset statistics using the stats function specified in
    the table config, assembles a DataFrame, sorts rows, and renders the
    result as a matplotlib table saved to ``save_dir``.

    Parameters
    ----------
    table : dict
        Table configuration from ``settings.supp_tables``. Expected keys:
        ``collection``, ``title``, ``figure_name``, ``sort_by``, ``columns``,
        and optionally ``stats_fn``, ``annotations_to_include``, and
        ``sort_bottom``.
    cell_manifest : DataframeManifest
        Manifest mapping dataset names to feature dataframe locations.
    save_dir : Path
        Directory where the SVG output will be saved.
    shear_stress_datasets : set or None, optional
        Set of dataset names in the shear stress collection. Required only
        for the segmentation table (``get_seg_stats``).
    diffae_datasets : set or None, optional
        Set of dataset names in the DiffAE collection. Required only for
        the segmentation table (``get_seg_stats``).
    """
    annotations_to_include = table.get("annotations_to_include", None)
    dataset_names = get_datasets_in_collection(table["collection"])

    # Collect stats using the appropriate function
    if table.get("stats_fn") == "get_seg_stats":
        stats = [
            get_seg_stats(
                name,
                cell_manifest=cell_manifest,
                shear_stress_datasets=shear_stress_datasets or set(),
                diffae_datasets=diffae_datasets or set(),
            )
            for name in tqdm(dataset_names, desc=table["title"])
        ]
    elif annotations_to_include is not None:
        stats = [
            get_dataset_stats(
                name, cell_manifest=cell_manifest, annotations_to_include=annotations_to_include
            )
            for name in tqdm(dataset_names, desc=table["title"])
        ]
    else:
        stats = [
            get_dataset_stats(name, cell_manifest=cell_manifest)
            for name in tqdm(dataset_names, desc=table["title"])
        ]
    cols = table["columns"]
    df = pd.DataFrame([{col: s[key] for col, key in cols} for s in stats])

    # Numeric sort for shear stress column
    sort_cols = table["sort_by"]
    sort_bottom = table.get("sort_bottom", [])

    # Build _bottom column if needed (push 0 dyn and flow switch datasets to bottom)
    if sort_bottom and SHEAR_STRESS in df.columns:
        df["_bottom"] = df[SHEAR_STRESS].apply(
            lambda x: 1 if x in sort_bottom or ", " in str(x) else 0
        )
    else:
        df["_bottom"] = 0

    if sort_cols[0] == SHEAR_STRESS:
        df["_sort"] = df[SHEAR_STRESS].str.split(", ").str[0].astype(int)
        df = df.sort_values(by=["_bottom", "_sort"] + sort_cols[1:], ignore_index=True).drop(
            columns=["_sort", "_bottom"]
        )
    else:
        df = df.sort_values(by=["_bottom"] + sort_cols, ignore_index=True).drop(columns="_bottom")

    # Render table
    fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH, MAX_FIGURE_HEIGHT))
    ax.axis("off")

    col_labels = df.columns.tolist()
    n_cols = len(col_labels)
    n_rows = len(df)

    cell_text = [
        [textwrap.fill(str(v), width=MAX_CELL_CHARS) for v in row] for row in df.values.tolist()
    ]

    max_chars = [
        max(max(len(ln) for ln in cell_text[r][c].split("\n")) for r in range(n_rows))
        for c in range(n_cols)
    ]
    total = sum(max(c, 3) for c in max_chars)
    col_widths = [max(c, 3) / total for c in max_chars]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=None,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.scale(1.0, 1.4)

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

    plt.title(table["title"], fontsize=10, weight="bold", pad=60, loc="left")
    tbl.set_in_layout(True)
    save_plot_to_path(
        figure=fig,
        output_path=save_dir,
        figure_name=table["figure_name"],
        file_format=".svg",
        bbox_inches="tight",
    )
