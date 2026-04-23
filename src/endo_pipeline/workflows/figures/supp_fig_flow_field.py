# %%
"""Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.flow_field_schematic import (
    make_kernel_convolution_schematic,
    make_real_image_panel,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
)

# %%
plt.style.use("endo_pipeline.figure")


output_path = get_output_path("supp_fig_flow_field")
# %%
image_panel_path = make_real_image_panel(output_path)

# %%
# Use the dataset defined in the flow field construction examples
example_dataset_name = "20250409_20X"
dataset_config = load_dataset_config(example_dataset_name)

# Load the manifest and get the dataframe location for this dataset
feature_manifest_name = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
feature_manifest = load_dataframe_manifest(feature_manifest_name)

dataset_location = get_dataframe_location_for_dataset(feature_manifest, example_dataset_name)

# Load only r and rho feature + metadata columns
column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
columns_to_load = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]

df_raw = load_dataframe(dataset_location, delay=True)
df: pd.DataFrame = df_raw[columns_to_load].compute()

bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]
bins, centers = get_bins(bin_widths, df[column_names].to_numpy())
target_point = (1.0, -0.1)
print(f"Target point for schematic panels: {target_point}")
zoom_factor = 3.5
r_half = zoom_factor * KERNEL_BANDWIDTHS_DYNAMICS[column_names[0]]
rho_half = zoom_factor * KERNEL_BANDWIDTHS_DYNAMICS[column_names[1]]
xlim = (target_point[0] - r_half, target_point[0] + r_half)
ylim = (target_point[1] - rho_half, target_point[1] + rho_half)
# Filter to steady-state timepoints and a single flow condition
dataframe_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
kernel_convolution_panel_path = make_kernel_convolution_schematic(
    output_path,
    dataframe_steady_state,
    column_names,
    target_point,
    bin_edges=bins,
    bin_centers=centers,
    axes_xlim=xlim,
    axes_ylim=ylim,
    fig_kwargs={"figsize": (MAX_FIGURE_WIDTH, MAX_FIGURE_WIDTH), "layout": "constrained"},
)

# %%
