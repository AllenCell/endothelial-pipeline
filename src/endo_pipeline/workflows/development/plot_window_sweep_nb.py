"""
Load and plot the start of steady state and cell piling sweep of the fixed points.

#dynamical-systems #fixed-points #grid-based #sweep #plot
"""

# %%
import logging

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.window_sweep import (
    DETECTION_RATE,
    SHEAR_STRESS_BIN,
    WINDOW_EDGE_MINUTES,
)
from endo_pipeline.library.visualize.window_sweep_plots import (
    MINUTES_PER_HOUR,
    WINDOW_EDGE_HOURS,
    plot_num_timepoints,
    plot_sweep,
)
from endo_pipeline.manifests import list_datasets_with_dataframes, load_dataframe_manifest
from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES

plt.style.use("endo_pipeline.figure")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_path = get_output_path(__file__)

MANIFEST_NAMES = {
    "start_cell_piling": "window_sweep_start_cell_piling",
    "start_steady_state": "window_sweep_start_steady_state",
}
BASELINE_FIXED_POINT_MANIFEST_NAME = BOOTSTRAPPING_MANIFEST_NAMES["grid_based"]

# %% load data
frames: dict[str, dict[str, pd.DataFrame]] = {}
for boundary, manifest_name in MANIFEST_NAMES.items():
    manifest = load_dataframe_manifest(manifest_name)
    frames[boundary] = {}
    for dataset_name in list_datasets_with_dataframes(manifest):
        df = load_dataframe(manifest.locations[dataset_name])
        # Keep only higher-confidence fixed points (same threshold the other
        # fixed-point workflows use).
        if DETECTION_RATE not in df.columns:
            logger.warning(
                "'%s' column missing for %s / %s (pre-bootstrap output?); skipping. "
                "Re-run `endopipe sweep-window-fixed-points` for this dataset.",
                DETECTION_RATE,
                dataset_name,
                boundary,
            )
            continue
        df = df[df[DETECTION_RATE] >= BOOTSTRAP_THRESHOLD]
        df[WINDOW_EDGE_HOURS] = df[WINDOW_EDGE_MINUTES] / MINUTES_PER_HOUR
        frames[boundary][dataset_name] = df

# Both boundaries share the same absolute-time edge grid so a single x-axis
# range keeps the steady-state and cell-piling plots on the same scale.
x_max_hours = max(
    float(df[WINDOW_EDGE_HOURS].max())
    for boundary_frames in frames.values()
    for df in boundary_frames.values()
)

# %% load baseline fixed points
# The bootstrapping dataframe stores baseline coordinates as `{col}_baseline` and
# a per-fixed-point detection_rate. Rename the baseline columns to the
# `{col}_fixed_point` names the sweep plots use, drop low-confidence fixed points
# (same threshold as the swept points), and attach a shear_stress_bin column for
# color matching, so they can be overlaid as horizontal reference lines.
baseline_manifest = load_dataframe_manifest(BASELINE_FIXED_POINT_MANIFEST_NAME)
baseline_frames: dict[str, pd.DataFrame] = {}
for dataset_name in list_datasets_with_dataframes(baseline_manifest):
    baseline_df = load_dataframe(baseline_manifest.locations[dataset_name])
    if Column.FIXED_POINT_DETECTION_RATE in baseline_df.columns:
        baseline_df = baseline_df[
            baseline_df[Column.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
        ]
    baseline_df = baseline_df.rename(
        columns={
            ColumnTemplate.BASELINE_FIXED_POINT
            % column_name: (ColumnTemplate.FIXED_POINT % column_name)
            for column_name in DYNAMICS_COLUMN_NAMES
        }
    )
    config = load_dataset_config(dataset_name)
    bin_by_stress = {fc.shear_stress: fc.shear_stress_bin for fc in config.flow_conditions}
    baseline_df[SHEAR_STRESS_BIN] = baseline_df[Column.SHEAR_STRESS].map(bin_by_stress)
    baseline_frames[dataset_name] = baseline_df

# %% plot
for boundary, boundary_frames in frames.items():
    for dataset_name, df in boundary_frames.items():
        fig = plot_sweep(
            df,
            dataset_name,
            boundary,
            x_max_hours,
            baseline_df=baseline_frames.get(dataset_name),
        )
        save_plot_to_path(
            fig,
            output_path,
            f"window_sweep_{boundary}_{dataset_name}",
            tight_layout=False,
            bbox_inches="tight",
        )

        fig = plot_num_timepoints(df, dataset_name, boundary, x_max_hours)
        save_plot_to_path(
            fig,
            output_path,
            f"window_sweep_num_timepoints_{boundary}_{dataset_name}",
            tight_layout=False,
            bbox_inches="tight",
        )
# %%
