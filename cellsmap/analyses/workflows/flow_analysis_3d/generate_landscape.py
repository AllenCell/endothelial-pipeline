# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import decomposition as skdecomp
from cellsmap.util.set_ouput import get_output_path
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.utils.io import vtk_tools

# Create output folder if does not exist yet
workflow_fig_folder = "flow_analysis_3d/figs"
workflow_vtk_folder = "flow_analysis_3d/vtks"
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load manifest created at preprocessing step
df = pd.read_csv(Path(fig_savedir).parent/"manifest.csv")

# Create landscape
DDFF = vtk_tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df, identifier="CropId")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()
for condition in df.description.unique():
    DDFF.compute_landscape(condition=condition)
    DDFF.simulate_particles_in_landscape(condition=condition)
DDFF.simulate_particles_in_landscape(condition=["48hr_High"]*50+["48hr_Low"]*50, filename_prefix="High_to_Low")
DDFF.simulate_particles_in_landscape(condition=["48hr_Low"]*50+["48hr_High"]*50, filename_prefix="Low_to_High")

# # Visualize mean trajectory (red cross is initial state)
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
# for tid, traj in enumerate(trajs):
#     rmean = np.array(traj).mean(axis=0)
#     xp = xmin + rmean[0]*grid_spacing
#     yp = ymin + rmean[1]*grid_spacing
#     zp = zmin + rmean[2]*grid_spacing
#     ax1.scatter(xp, yp, color="black", s=10)
#     ax2.scatter(xp, zp, color="black", s=10)
#     if tid == 0:
#         ax1.scatter(xmin+rmean[0]*grid_spacing, ymin+rmean[1]*grid_spacing, color="red", marker="X", s=50, label="start")
#         ax2.scatter(xmin+rmean[0]*grid_spacing, zmin+rmean[2]*grid_spacing, color="red", marker="X", s=50, label="start")
#     for axid, (ax, (vmin, vmax)) in enumerate(zip([ax1, ax2], [(ymin,ymax), (zmin,zmax)])):
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(ymin, ymax)
#         ax.set_xlabel("PC1", fontsize=14)
#         ax.set_ylabel(f"PC{axid+2}", fontsize=14)
#         ax.set_aspect("equal")
# plt.legend()
# plt.tight_layout()
# vb.save_plot(fig, filename=fig_savedir+"mean_trajectories", dpi=72)

# # Invert final coordinate to get 8D representation
# print("8D coordinates of final state:")
# print(reducer.inverse_transform([xp, yp, zp]))

# # Compute and plot speed
# path_list = []
# for tid, traj in enumerate(trajs):
#     rmean = np.array(traj).mean(axis=0)
#     xp = xmin + rmean[0]*grid_spacing
#     yp = ymin + rmean[1]*grid_spacing
#     zp = zmin + rmean[2]*grid_spacing
#     path_list.append([xp, yp, zp])
# path_array = np.array(path_list)

# speed = np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1))
# fig, ax = plt.subplots(1,1, figsize=(3,3))
# ax.plot(speed)
# ax.set_xlabel("Simulation frame")
# ax.set_ylabel("Speed (unit of length \n in state space/frame)")
# vb.save_plot(fig, filename=fig_savedir+"population_speed", dpi=72)

# # Run simulation for changing flow
# dUis, dVis, dQis, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition="48hr High", time_step=time_step)
# dUis2, dVis2, dQis2, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition="48hr Low", time_step=time_step)
# filename_prefix = vtk_savedir+"/high_to_low"
# tools.simulate_particles_in_changing_vector_field(dUis=dUis, dVis=dVis, dQis=dQis, transition=50, dUis2=dUis2, dVis2=dVis2, dQis2=dQis2, grid=grid, n_particles=500, speed=mean_speed, grid_spacing=grid_spacing, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, filename_prefix=filename_prefix)
