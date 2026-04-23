# %%
"""Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.flow_field_schematic import (
    make_kernel_convolution_schematic,
    make_real_image_panel,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import KERNEL_BANDWIDTHS_DYNAMICS
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_figure import SUPP_FIG_TARGET_POINT, SUPP_FIG_ZOOM_FACTOR

# %%
plt.style.use("endo_pipeline.figure")


output_path = get_output_path("supp_fig_flow_field")
# %%
image_panel_path = make_real_image_panel(output_path)

# %% Use the dataset defined in the flow field construction examples for low
# shear stress
dataset_name = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
target_point = SUPP_FIG_TARGET_POINT
r_half = SUPP_FIG_ZOOM_FACTOR * KERNEL_BANDWIDTHS_DYNAMICS[column_names[0]]
rho_half = SUPP_FIG_ZOOM_FACTOR * KERNEL_BANDWIDTHS_DYNAMICS[column_names[1]]
xlim = (target_point[0] - r_half, target_point[0] + r_half)
ylim = (target_point[1] - rho_half, target_point[1] + rho_half)
# Filter to steady-state timepoints and a single flow condition
kernel_convolution_panel_path = make_kernel_convolution_schematic(
    output_path,
    dataset_name,
    column_names,
    target_point,
    axes_xlim=xlim,
    axes_ylim=ylim,
    fig_kwargs={"figsize": (MAX_FIGURE_WIDTH, MAX_FIGURE_WIDTH), "layout": "constrained"},
)

# %%
