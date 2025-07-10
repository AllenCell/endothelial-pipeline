# %%
import fire
import pandas as pd
import torch

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import component_heatmaps
from src.endo_pipeline.library.process.get_images import (
    get_crops_in_dataframe,
    global_contrast_crop_list_channel,
)
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.crop_montage import plot_crop_montage
from src.endo_pipeline.library.visualize.diffae_features.manifest_viz import (
    plot_principal_component_histogram,
)

# %%
# def main(
#     dataset_names: str | list[str] | None = None,
#     pc_axis: int = 1,
#     pc_val: float = 0.25,
#     plot_heatmap: bool = False,
#     frame_range: list | None = None,
# ) -> None:
#     """
#     Run the PC heatmap workflow.

#     For each dataset with Diff AE manifest data, this function:
#         - generates histograms of PC features (saves .png out to .results/crop_visualization/figs/)
#         - filters for crops based on the given input value of a given PC dimension
#         and a given range of timepoints and does the following:
#                 - gets the corresponding crops from the original images
#                     (saves .tiff files out to .results/crop_visualization/figs/original_crops/)
#                 - reconstructs the crops by passing the PC space coordinates
#                     through the Diff AE image generation model
#                     (saves .tiff files out to .results/crop_visualization/figs/reconstructed_crops/)
#                 - saves the filtered dataframe of crops to a .csv file
#                     (saves .csv file out to .results/crop_visualization/outputs/)

#     Inputs:
#     - dataset_names: str or list of str
#         The name(s) of the dataset(s) to use for the workflow.
#     - pc_axis: int
#         The principal component to filter by (0-indexed).
#         For example, if you want to filter by a particular value of
#         the 2nd PC, set this to 1.
#     - pc_val: float
#         The value of the PC dimension to filter by.
#         For example, if you want to filter by crops with pc_to_explore
#         component of the PC coordinates ~= 0.5, set this to 0.5.
#         The filtering is done by binning the PCs and
#             getting the bin index that contains pc_val.
#     - plot_heatmap: bool = False
#         Whether to plot the histogram of the first 3 PCs for each dataset.
#     - frame_range: list of int | None = None
#         The range of timepoints to filter by.
#         For example, if you want to filter by crops between frames 225 and 275,
#         set this to [225, 275].
#         If None, no filtering is done by timepoints.

#     Outputs:
#     - None, but saves out images and csv files to the appropriate directories.
#         (See above for details.)
#     """
# %%
dataset_names: str | list[str] | None = None
pc_axis: int = 1
pc_val: float = 0.25
plot_heatmap: bool = False
frame_range = None
# %%
fig_savedir = get_output_path("crop_visualization")

if isinstance(dataset_names, str):
    list_of_datasets = [dataset_names]
elif dataset_names is None:
    list_of_datasets = manifest_io.list_datasets_with_manifest(
        "diffae_manifest_fmsid", verbose=True, timelapse_only=True
    )
    list_of_datasets = [name for name in list_of_datasets if "mito" not in name]
else:
    list_of_datasets = dataset_names

num_bins = 40  # number of bins for histogram, hardcoded right now but somewhat arbitrary

pca = fit_pca()
# %%
bin_limits = component_heatmaps.get_3d_bounds_from_data(
    list_of_datasets, pca, col_names="feat", filter_to_valid=False
)
# %%
# first load and concatenate datasets
df_list = []
for ds_name in list_of_datasets:
    # get manifest data with crop index column added
    df = get_manifest_for_dynamics_workflows(ds_name, pca=pca, filter_to_valid=False)
    df_list.append(df)
df_all_datasets = pd.concat(df_list, ignore_index=True)
# %%
# get heatmap data for the first 3 PCs over time
# and update the dataframe with binning information
hist_array_list, bin_edges, df_with_bins = component_heatmaps.get_histogram_by_component(
    df_all_datasets,
    num_bins,
    bin_limits,
    feat_cols=manifest_io.get_feature_cols(df_all_datasets)[:3],
)

# plot histogram of PCs for each component (optional)
if plot_heatmap:
    for i, ds_name in enumerate(list_of_datasets):
        fig, _ = plot_principal_component_histogram(hist_array_list[i], bin_edges)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir + f"{ds_name}_pc_histogram")
# %%
# get dataframe of crops with bin_{latent_dim} == bin_index(latent_val),
# where bin_index is the index of the bin that contains latent_val
# in the bin edges over the given latent dimension
df_filtered = component_heatmaps.get_df_by_bin_value(df_with_bins, pc_axis, pc_val, bin_edges)

# select timepoints within the given range
# this is just to filter number of crops we get
# might want to remove this step in the future
if frame_range is not None:
    df_filtered = df_filtered[
        (df_filtered["frame_number"] >= frame_range[0])
        & (df_filtered["frame_number"] <= frame_range[1])
    ]
# %% Only get crops for the subset you want to visualize
n_num_crops = 100
# randomly sample n_num_crops rows from the filtered dataframe
df_sample = df_filtered.sample(
    n=n_num_crops, random_state=42, replace=False
)  # replace=False to avoid duplicates

# %%
# get and save out crops corresponding to
# the rows in the filtered dataframe
crop_list, df_sample_sorted = get_crops_in_dataframe(
    df_sample
)  # Define channel indices and corresponding image content names
channels = [
    (0, "bf_slice_g"),
    (1, "bf_max_proj_g"),
    (2, "stddev_bf_g"),
    (3, "cdh5_g"),
]

# Generate contrast-adjusted crop lists and plot montages
for channel_index, image_content in channels:
    crop_list_channel = global_contrast_crop_list_channel(crop_list, channel_index, "percentile")
    plot_crop_montage(
        crop_list_channel,
        df_sample_sorted,
        pc_axis,
        pc_val,
        image_content=image_content,
        channel_index=None,
        save_dir=fig_savedir,
    )

# Get crops with individual contrast adjustment
ind_contrast_crop_list, df_sample_sorted = get_crops_in_dataframe(
    df_sample, contrast_crops_individually=True
)

# Define channel indices and corresponding image content names for individual contrast
ind_channels = [
    (0, "bf_max_proj_ind"),
    (1, "stddev_bf_ind"),
    (2, "cdh5_ind"),
]

# Plot montages for individual contrast-adjusted crops
for channel_index, image_content in ind_channels:
    plot_crop_montage(
        ind_contrast_crop_list,
        df_sample_sorted,
        pc_axis,
        pc_val,
        image_content=image_content,
        channel_index=channel_index,
        save_dir=fig_savedir,
    )

# %%
# get reconstructed ve-cad crops
# corresponding to the rows in the filtered dataframe
# this requires a GPU to run, so
# we check if a GPU is available before running this part
gpu_available = torch.cuda.is_available()
if gpu_available:
    from src.endo_pipeline.library.model.diffae.generate_image import (
        get_reconstructed_crops_in_dataframe,
    )

    reconstructed_crop_list = get_reconstructed_crops_in_dataframe(df_filtered)

    plot_crop_montage(
        reconstructed_crop_list,
        df_sample_sorted,
        pc_axis,
        pc_val,
        image_content="reconstructed_cdh5",
        channel_index=None,
        save_dir=fig_savedir,
    )
else:
    print("GPU not available, skipping reconstruction of crops.")

# %%
# if __name__ == "__main__":
#     fire.Fire(main)
