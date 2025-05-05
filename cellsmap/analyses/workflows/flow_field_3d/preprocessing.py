# Preprocess the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import matplotlib.pyplot as plt
import pandas as pd

from cellsmap.analyses.utils.io import vtk_tools as tools
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_io
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path

# %%
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# %%
# only keep the reference datasets for this workflow
datasets_to_use = [
    "20241120_20X",
    "20250409_20X",
    "20241217_20X",
]  # 48hr high flow, 48hr no flow, 48hr low flow
df = []
# load the manifest for each dataset, add outlier column, add crop index column
for name in datasets_to_use:
    df_ = manifest_io.get_diffae_manifest(name)
    # add outlier column
    df_ = manifest_pca.get_outliers(df_)
    # add crop index column
    df_ = diffae_preproc.add_crop_index(df_)
    # add dataset name to crop index
    # this is to make the crop index unique across datasets
    df_["crop_index"] = f"{name}_" + df_["crop_index"].astype(
        str
    )  # add dataset name to crop index to make unique
    df_ = diffae_preproc.add_description_column(
        df_, name, simple=True
    )  # add description column (e.g., 48hr_High)
    df.append(df_)

df = pd.concat(
    df, ignore_index=True
)  # concatenate the dataframes into a single dataframe

# %%
# plot the data in latent space (features 1 and 4) before removing outliers
fig, ax = vb.init_plot(figsize=(5, 5))
for ds_name in datasets_to_use:
    # get the data for the dataset based on ds_name being in the crop_index column
    dfs = df[df["crop_index"].str.contains(ds_name)]
    ax.scatter(dfs["feat_1"], dfs["feat_4"], s=0.1, label=ds_name)
plt.legend()
plt.show()
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_overview_feats_1_4", dpi=72)

# plot latent dims 1 and 4 after with outliers labelled
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(df["feat_1"], df["feat_4"], c=df["outlier"], s=0.2)
plt.show()
vb.save_plot(
    fig,
    filename=fig_savedir + "reference_dataset_overview_feats_1_4_no_bubbles",
    dpi=72,
)

# shape of the dataset before removing outliers
shape_init = df.shape

# remove outliers (bubbles) from the dataset
# note: this is for downstream analysis,
# outliers automatically removed for fitting PCA
df = manifest_pca.remove_outliers(df)
shape_post = df.shape
print(f"Removed {shape_init[0]-shape_post[0]} outliers from the dataset")

# %%

# fit PCA to data
scale = False  # whether to scale the data before PCA
pca = manifest_pca.fit_pca(num_pcs=3, scale=scale)

# save out PCA object (need later for analysis and summary of fit dynamical systems model)
manifest_io.save_pca_model(pca, output_savedir)

# Apply PCA
feat_cols = manifest_io.get_feature_cols(df)
X = df[feat_cols].values
Xt = pca.transform(X)

# add PCA components to dataframe
for pc in range(3):
    df[f"PC{pc+1}"] = Xt[:, pc]

# %%
# initialize the DataDrivenFlowField3D object
df = df.sort_values(by=["crop_index", "frame_number"])
DDFF = tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df, identifier="crop_index")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.set_excluded_fraction(0.0)
DDFF.build()

# %%
# plot the PCA components
fig, (ax1, ax2) = vb.init_subplots(figsize=(15, 5))
for i, ds_name in enumerate(datasets_to_use):
    print(f"Plotting {ds_name}")
    # get the data for the dataset based on ds_name being in the crop_index column
    dfs = df[df["crop_index"].str.contains(ds_name)]
    alpha = 0.75
    if ds_name == "20241217_20X":
        alpha = 0.5
    ax1.scatter(dfs.PC1, dfs.PC2, s=0.01, label=ds_name, alpha=alpha)
    ax2.scatter(dfs.PC1, dfs.PC3, s=0.01, label=ds_name, alpha=alpha)
    for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"]):
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
        if ylab == "PC2":
            ax.set_ylim(DDFF._bounds.ymin, DDFF._bounds.ymax)
        else:
            ax.set_ylim(DDFF._bounds.zmin, DDFF._bounds.zmax)
        ax.set_aspect("auto")
plt.tight_layout()
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_scatter", dpi=72)

# %%
fig, (ax1, ax2) = vb.init_subplots(figsize=(15, 5))
ax1.scatter(df.PC1, df.PC2, cmap="inferno", s=0.01, c=df["frame_number"])
ax2.scatter(df.PC1, df.PC3, cmap="inferno", s=0.01, c=df["frame_number"])
for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"]):
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
    if ylab == "PC2":
        ax.set_ylim(DDFF._bounds.ymin, DDFF._bounds.ymax)
    else:
        ax.set_ylim(DDFF._bounds.zmin, DDFF._bounds.zmax)
    ax.set_aspect("auto")
plt.tight_layout()
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_temporal", dpi=72)

# %%
# plot with example single crop tracks
fig, ax = vb.init_plot(figsize=(5, 5))
ax.scatter(df.PC1, df.PC2, s=0.1, color="black", alpha=0.05)
for ds_name in datasets_to_use:
    # get the data for the dataset based on ds_name being in the crop_index column
    dfs = df[df["crop_index"].str.contains(ds_name)]
    for track, df_track in dfs.groupby("crop_index"):
        ax.plot(df_track.PC1, df_track.PC2, label=ds_name)
        break
ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
ax.set_ylim(DDFF._bounds.ymin, DDFF._bounds.ymax)
ax.set_aspect("equal")
plt.legend(loc="lower left", fontsize=8)
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_with_tracks", dpi=72)

# %%
# Save final manifest for creating flow fields
df.to_csv(output_savedir + "manifest.csv")

# %%
