# Preprocess the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import decomposition as skdecomp
from cellsmap.util.set_ouput import get_output_path
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.workflows.flow_analysis_3d import tools

# Read manifest.
# TODO: update how we read manifest based on latest version of hte repo
df = pd.read_csv("/allen/aics/assay-dev/users/Erin/endo_features/pca_ref_features.csv", index_col=0)
# Exclude bad no flow dataset
df = df.loc[df["group"] != "20241210_20X_timelapse_SLDY"]

# Fix values in column "description"
# Not needed based on latest version of Erin's code
df["description"] = df["description"].str.replace(" ", "_", regex=False).str.replace("(", "", regex=False).str.replace(")", "", regex=False).str.replace("/", "", regex=False)
print(df.description.unique())

# Create output folder if does not exist yet
workflow_fig_folder = "flow_analysis_3d/figs"
workflow_vtk_folder = "flow_analysis_3d/vtks"
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# What the data looks like?
fig, ax = plt.subplots(1,1, figsize=(5,5))
for (group, dfs) in df.groupby("group"):
    ax.scatter(dfs["1"], dfs["4"], s=0.1, label=group)
plt.legend()
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_overview_feats_1_4", dpi=72)

# Matheus' blubbles classifier. Maybe to be replaced with something else
vals = tools.simple_linear_classifier(X=df["1"].values, Y=df["4"].values)
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(df["1"], df["4"], c=vals, s=0.2)
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_overview_feats_1_4_no_bubbles", dpi=72)

print("Manifest shape before and after outlier (bubbles) removal:")
print(df.shape)
df["outlier"] = vals
df = df.loc[df.outlier==False]
print(df.shape)

# Apply PCA
X = df[[str(u) for u in range(8)]].values
reducer = skdecomp.PCA(n_components=3)
Xt = reducer.fit_transform(X)

# Save PCA model
with open(Path(vtk_savedir).parent/"pca_model.pkl", "wb") as file:
    pickle.dump(reducer, file)

# Create unique ID for each crop
for pc in range(3):
    df[f"PC{pc+1}"] = Xt[:, pc]
df["CropId"] = df["group"] + "_" + df["FOV_ID"].astype(str) + "_" + df["start_x"].astype(str) + "_" + df["start_y"].astype(str)
df = df.sort_values(by=["CropId", "T"])

DDFF = tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df, identifier="CropId")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(df.PC1, df.PC2, cmap="inferno", s=0.01, c=df["T"])
ax2.scatter(df.PC1, df.PC3, cmap="inferno", s=0.01, c=df["T"])
for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"]):
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
    ax.set_ylim(DDFF._bounds.zmin, DDFF._bounds.zmax)
    ax.set_aspect("equal")
plt.tight_layout()
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_pcs_temporal", dpi=72)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(df.PC1, df.PC2, s=0.1, color="black", alpha=0.05)
for group, df_group in df.groupby("group"):
    for track, df_track in df_group.groupby("CropId"):
        ax.plot(df_track.PC1, df_track.PC2, label=group)
        break
ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
ax.set_ylim(DDFF._bounds.ymin, DDFF._bounds.ymax)
ax.set_aspect("equal")
plt.legend()
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_pcs_with_tracks", dpi=72)

# Save final manifest for creating landscapes
df.to_csv(Path(fig_savedir).parent/"manifest.csv")
