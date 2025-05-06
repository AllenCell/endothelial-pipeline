# %%
import matplotlib.pyplot as plt
import pandas as pd

from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
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
# only keep the reference datasets for this workflow:
# 48hr high flow, 48hr no flow, 48hr low flow, 
# 2 48hr intermediate flows (12 and 15 dyn)
datasets_to_use = [
    "20241120_20X",
    "20241217_20X",
    "20250409_20X",
    "20250319_20X",
    "20250326_20X",
]  
df = []
# load the manifest for each dataset, add crop index column
for name in datasets_to_use:
    df_ = manifest_io.get_diffae_manifest(name)
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

# fit PCA to data (only working with top 3 PCs)
pca = manifest_pca.fit_pca(num_pcs=3)

# save out PCA object
manifest_io.save_pca_model(pca, output_savedir)

# Apply PCA
feat_cols = manifest_io.get_feature_cols(df)
x_proj = pca.transform(df[feat_cols].values)

# add PCA components to dataframe
for pc in range(3):
    df[f"pc{pc+1}"] = x_proj[:, pc]

# %%
# Save final manifest for creating flow fields
df.to_csv(output_savedir + "manifest.csv")
# %%
# get state space bounds from data
# used for plotting in this file, 
# analysis in generate_flow_field.py
bounds = ddff.set_3d_bounds_from_data(df.pc1, df.pc2, df.pc3, excluded_fraction=0.0)

# plot the PCA components
fig, (ax1, ax2) = vb.init_subplots(figsize=(15, 5))
for ds_name in datasets_to_use:
    print(f"Plotting {ds_name}")
    # get the data for the dataset based on 
    # ds_name being in the crop_index column
    dfs = df[df["crop_index"].str.contains(ds_name)]
    alpha = 0.75
    if ds_name == "20241217_20X":
        alpha = 0.5
    ax1.scatter(dfs.pc1, dfs.pc2, s=0.01, label=ds_name, alpha=alpha)
    ax2.scatter(dfs.pc1, dfs.pc3, s=0.01, label=ds_name, alpha=alpha)
    for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"], strict=False):
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_xlim(bounds[0][0], bounds[0][1])
        if ylab == "PC2":
            ax.set_ylim(bounds[1][0], bounds[1][1])
        else:
            ax.set_ylim(bounds[2][0], bounds[2][1])
        ax.set_aspect("auto")
plt.tight_layout()
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_scatter", dpi=72)

# %%
fig, (ax1, ax2) = vb.init_subplots(figsize=(15, 5))
ax1.scatter(df.pc1, df.pc2, cmap="inferno", s=0.01, c=df["frame_number"])
ax2.scatter(df.pc1, df.pc3, cmap="inferno", s=0.01, c=df["frame_number"])
for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"], strict=False):
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlim(bounds[0][0], bounds[0][-1])
    if ylab == "PC2":
        ax.set_ylim(bounds[1][0], bounds[1][-1])
    else:
        ax.set_ylim(bounds[2][0], bounds[2][-1])
    ax.set_aspect("equal")
plt.tight_layout()
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_temporal", dpi=72)

# %%
# plot with example single crop tracks
fig, ax = vb.init_plot(figsize=(5, 5))
ax.scatter(df.pc1, df.pc2, s=0.1, color="black", alpha=0.05)
for ds_name in datasets_to_use:
    # get the data for the dataset based on 
    # ds_name being in the crop_index column
    dfs = df[df["crop_index"].str.contains(ds_name)]
    for _track, df_track in dfs.groupby("crop_index"):
        ax.plot(df_track.pc1, df_track.pc2, label=ds_name)
        break
ax.set_xlabel("PC1", fontsize=14)
ax.set_ylabel("PC2", fontsize=14)
ax.set_xlim(bounds[0][0], bounds[0][-1])
ax.set_ylim(bounds[1][0], bounds[1][-1])
ax.set_aspect("equal")
plt.legend(loc="lower left", fontsize=8)
vb.save_plot(fig, filename=fig_savedir + "reference_dataset_pcs_with_tracks", dpi=72)


# %%
