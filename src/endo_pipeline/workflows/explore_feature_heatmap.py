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
    global_contrast_crop_list,
    individual_contrast_crop_list,
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

# load and concatenate datasets
df_list = []
for ds_name in list_of_datasets:
    # get manifest data with crop index column added
    df = get_manifest_for_dynamics_workflows(ds_name, pca=pca, filter_to_valid=False)
    df_list.append(df)
df_all_datasets = pd.concat(df_list, ignore_index=True)


bin_limits = component_heatmaps.get_3d_bounds_from_data(
    list_of_datasets, pca, col_names="feat", filter_to_valid=False
)
# get heatmap data for the first 3 PCs over time
# and update the dataframe with binning information
hist_array_list, bin_edges, df_with_bins = component_heatmaps.get_histogram_by_component(
    df_all_datasets,
    N_BINS,
    bin_limits,
    feat_cols=manifest_io.get_feature_cols(df_all_datasets)[:3],
)

if plot_heatmap:
    for i, ds_name in enumerate(list_of_datasets):
        fig, _ = plot_principal_component_histogram(hist_array_list[i], bin_edges)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir + f"{ds_name}_pc_histogram")


df_filtered = component_heatmaps.get_df_by_bin_value(df_with_bins, pc_axis, pc_val, bin_edges)

if frame_range is not None:
    df_filtered = df_filtered[
        (df_filtered["frame_number"] >= frame_range[0])
        & (df_filtered["frame_number"] <= frame_range[1])
    ]

# randomly sample n_num_crops rows from the filtered dataframe
df_sample = df_filtered.sample(
    n=N_NUM_CROPS, random_state=RANDOM_SEED, replace=False
)  # replace False avoids duplicates

# get crop images
(
    bf_single_slice,
    bf_max_projection,
    bf_std_deviation,
    gfp_max_projection,
    df_sample_sorted,
) = get_crops_in_dataframe(df_sample)

# %%
# Define crop types and their corresponding lists
crop_types = {
    "bf_slice": bf_single_slice,
    "bf_max_proj": bf_max_projection,
    "stddev_bf": bf_std_deviation,
    "cdh5": gfp_max_projection,
}

# Generate global and individual contrast crop lists
contrast_crops = {}
for name, crop_list in crop_types.items():
    contrast_crops[f"{name}_global_contrast"] = global_contrast_crop_list(crop_list, "percentile")
    contrast_crops[f"{name}_ind_contrast"] = individual_contrast_crop_list(crop_list, "percentile")

# Check for GPU availability
gpu_available = torch.cuda.is_available()
# If GPU is available, import the reconstruction function and add reconstructed crops
if gpu_available:
    from src.endo_pipeline.library.model.diffae.generate_image import (
        get_reconstructed_crops_in_dataframe,
    )

    reconstructed_crop_list = get_reconstructed_crops_in_dataframe(df_filtered)
    contrast_crops["reconstructed_cdh5"] = reconstructed_crop_list
else:
    print("GPU not available, skipping reconstruction of crops.")

# Map channels to their respective crop lists and image content
montage_images = [(crop_list, key) for key, crop_list in contrast_crops.items()]

# Plot montages for each channel
for crop_list_channel, image_content in montage_images:
    plot_crop_montage(
        crop_list_channel,
        df_sample_sorted,
        pc_axis,
        pc_val,
        image_content=image_content,
        channel_index=None,
        save_dir=fig_savedir,
    )


# %%
# if __name__ == "__main__":
#     fire.Fire(main)

# %%
