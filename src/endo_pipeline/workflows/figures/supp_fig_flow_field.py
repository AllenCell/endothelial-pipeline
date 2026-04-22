# %%
"""Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.flow_field_schematic import make_real_image_panel

# %%
plt.style.use("endo_pipeline.figure")


output_path = get_output_path("supp_fig_flow_field")
# %%
image_panel_path = make_real_image_panel(output_path)
# %%
