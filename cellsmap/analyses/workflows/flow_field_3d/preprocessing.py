# Preprocess the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import matplotlib.pyplot as plt

from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils.io import vtk_tools as tools
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_pca, manifest_io
from cellsmap.util import manifest_io

# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# load manifest to DataFrame with metadata
df = manifest_io.load_manifest_to_df()

# only keep the reference datasets for this workflow
datasets_to_use = ["20241120_20X", "20241203_20X", "20241217_20X", "20240319_20X"] # 48hr high flow, 48hr low flow, 48hr no flow, 48hr medium flow
df = df.loc[df["dataset_name"].str.contains("|".join(datasets_to_use))]

# plot the data in latent space (features 1 and 4) before removing outliers
fig, ax = vb.init_plot(figsize=(5,5))
for (ds_name, dfs) in df.groupby("dataset_name"):
    ax.scatter(dfs["1"], dfs["4"], s=0.1, label=ds_name)
plt.legend()
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_overview_feats_1_4", dpi=72)

# plot latent dims 1 and 4 after with outliers labelled
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(df["1"], df["4"], c=df["outlier"], s=0.2)
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_overview_feats_1_4_no_bubbles", dpi=72)

# shape of the dataset before removing outliers
shape_init = df.shape

# remove outliers (bubbles) from the dataset
# note: this is for downstream analysis, 
# outliers automatically removed for fitting PCA
df = manifest_pca.remove_outliers(df)
shape_post = df.shape
print(f"Removed {shape_init[0]-shape_post[0]} outliers from the dataset")



# fit PCA to data
scale = False # whether to scale the data before PCA
pca = manifest_pca.fit_pca(df, num_pcs=3, scale=scale)

# save out PCA object (need later for analysis and summary of fit dynamical systems model)
manifest_io.save_pca_model(pca, output_savedir)

# Apply PCA
X = df[[str(u) for u in range(8)]].values
Xt = pca.transform(X)

# add PCA components to dataframe
for pc in range(3):
    df[f"PC{pc+1}"] = Xt[:, pc]


# replace description of shear stress in dyncm2 to qualitative description (High, Low, No)
descriptions = manifest_io.get_descriptive_metadata(df,simple=True)
df = manifest_io.add_descriptive_metadata(df, descriptions)

# Create unique ID for each crop
# initialize crop index column
df["crop_index"] = '1'
for ds_name in datasets_to_use:
    df_ = df.loc[df["dataset_name"] == ds_name] # get the dataframe restricted to the dataset
    df_ = manifest_io.add_crop_index(df_) # add crop index to the dataframe
    df.loc[df["dataset_name"] == ds_name, "crop_index"] = ds_name + "_" + df_["crop_index"].astype(str) # add the crop index as column to the original dataframe (append dataset name to make unique)
# %%
# initialize the DataDrivenFlowField3D object
df = df.sort_values(by=["crop_index", "T"])
DDFF = tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df, identifier="crop_index")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

# %%
# plot the PCA components
fig, (ax1, ax2) = vb.init_subplots(figsize=(10, 5))
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

# %%
# plot with example single crop tracks
fig, ax = vb.init_plot(figsize=(5, 5))
ax.scatter(df.PC1, df.PC2, s=0.1, color="black", alpha=0.05)
for ds_name, df_dataset in df.groupby("dataset_name"):
    for track, df_track in df_dataset.groupby("crop_index"):
        ax.plot(df_track.PC1, df_track.PC2, label=ds_name)
        break
ax.set_xlim(DDFF._bounds.xmin, DDFF._bounds.xmax)
ax.set_ylim(DDFF._bounds.ymin, DDFF._bounds.ymax)
ax.set_aspect("equal")
plt.legend(loc = "lower left", fontsize=8)
vb.save_plot(fig, filename=fig_savedir+"reference_dataset_pcs_with_tracks", dpi=72)

# %%
# Save final manifest for creating flow fields
df.to_csv(output_savedir+"manifest.csv")
