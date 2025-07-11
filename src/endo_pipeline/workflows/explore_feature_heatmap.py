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

N_BINS = 40  # number of bins for histogram, hardcoded right now but somewhat arbitrary
N_NUM_CROPS = 100
RANDOM_SEED = 42  # seed for reproducibility in sampling random crops

# %%
dataset_names = None
pc_axis = 1
pc_val = 0.25
plot_heatmap = False
frame_range = None
# %%
# def main(
# dataset_names: str | list[str] | None = None,
# pc_axis: int = 1,
# pc_val: float = 0.25,
# plot_heatmap: bool = False,
# frame_range: list[int] | None = None,
# ) -> None:
"""
Run the PC crop visualization workflow.

Args:
    dataset_names (str | list[str] | None): Name(s) of the dataset(s) to include in the analysis.
    pc_axis (int): Index of the principal component to filter by (0-indexed).
    pc_val (float): Value of the PC dimension to visualize crops for (e.g., 0.25).
    plot_heatmap (bool): Whether to generate and save PC histograms (default: False).
    frame_range (list[int] | None): Timepoint range for filtering (e.g., [225, 275]).
        If None, no filtering is applied.

Returns:
    None: The function saves plots and data files to the output directory.
"""
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

pca = fit_pca()

bin_limits = component_heatmaps.get_3d_bounds_from_data(
    list_of_datasets, pca, col_names="feat", filter_to_valid=False
)

# first load and concatenate datasets
df_list = []
for ds_name in list_of_datasets:
    # get manifest data with crop index column added
    df = get_manifest_for_dynamics_workflows(ds_name, pca=pca, filter_to_valid=False)
    df_list.append(df)
df_all_datasets = pd.concat(df_list, ignore_index=True)

# get heatmap data for the first 3 PCs over time
# and update the dataframe with binning information
hist_array_list, bin_edges, df_with_bins = component_heatmaps.get_histogram_by_component(
    df_all_datasets,
    N_BINS,
    bin_limits,
    feat_cols=manifest_io.get_feature_cols(df_all_datasets)[:3],
)

# plot histogram of PCs for each component (optional)
if plot_heatmap:
    for i, ds_name in enumerate(list_of_datasets):
        fig, _ = plot_principal_component_histogram(hist_array_list[i], bin_edges)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir + f"{ds_name}_pc_histogram")

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
# Only get crops for the subset you want to visualize

# randomly sample n_num_crops rows from the filtered dataframe
df_sample = df_filtered.sample(
    n=N_NUM_CROPS, random_state=RANDOM_SEED, replace=False
)  # replace False avoids duplicates
# %%
(
    bf_single_slice,
    bf_max_projection,
    bf_std_deviation,
    gfp_max_projection,
    df_sample_sorted,
) = get_crops_in_dataframe(df_sample)
(
    bf_single_slice_ind,
    bf_max_projection_ind,
    bf_std_deviation_ind,
    gfp_max_projection_ind,
    df_sample_sorted,
) = get_crops_in_dataframe(df_sample, contrast_crops_individually=True)
# %%
# Define individual crop lists for each channel
bf_single_slice_global = global_contrast_crop_list_channel(bf_single_slice, "percentile")
bf_max_projection_global = global_contrast_crop_list_channel(bf_max_projection, "percentile")
bf_std_deviation_global = global_contrast_crop_list_channel(bf_std_deviation, "percentile")
gfp_max_projection_global = global_contrast_crop_list_channel(gfp_max_projection, "percentile")

# Map channels to their respective crop lists
channels = [
    (bf_single_slice_global, "bf_slice_global_contrast"),
    (bf_max_projection_global, "bf_max_proj_global_contrast"),
    (bf_std_deviation_global, "stddev_bf_global_contrast"),
    (gfp_max_projection_global, "cdh5_global_contrast"),
    (bf_single_slice_ind, "bf_slice_individual_contrast"),
    (bf_max_projection_ind, "bf_max_proj_individual_contrast"),
    (bf_std_deviation_ind, "stddev_bf_individual_contrast"),
    (gfp_max_projection_ind, "cdh5_individual_contrast"),
]

# Plot montages for each channel
for crop_list_channel, image_content in channels:
    plot_crop_montage(
        crop_list_channel,
        df_sample_sorted,
        pc_axis,
        pc_val,
        image_content=image_content,
        channel_index=None,
        save_dir=fig_savedir,
    )

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

# %%
