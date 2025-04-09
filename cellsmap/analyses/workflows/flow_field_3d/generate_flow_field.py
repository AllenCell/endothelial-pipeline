# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025

import pandas as pd

from cellsmap.util.set_ouput import get_output_path
from cellsmap.analyses.utils.io import vtk_tools

# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load manifest created at preprocessing step
df = pd.read_csv(workflow_output_folder+"manifest.csv")

# Create flow field dx/dt = f(x)
DDFF = vtk_tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df, identifier="crop_index")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()
for condition in df.description.unique():
    DDFF.compute_landscape(condition=condition)
    DDFF.simulate_particles_in_landscape(condition=condition)
DDFF.simulate_particles_in_landscape(condition=["48hr_High"]*50+["48hr_Low"]*50, filename_prefix="High_to_Low")
DDFF.simulate_particles_in_landscape(condition=["48hr_Low"]*50+["48hr_High"]*50, filename_prefix="Low_to_High")