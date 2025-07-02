# %%
import fire

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import component_heatmaps
from src.endo_pipeline.library.model.diffae.generate_image import (
    get_reconstructed_crops_in_dataframe,
)
from src.endo_pipeline.library.process.get_images import get_original_crops_in_dataframe
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.crop_montage import plot_crop_montage
from src.endo_pipeline.library.visualize.diffae_features.manifest_viz import (
    plot_principal_component_histogram,
)


# %%
def main(pc_to_explore: int = 3, pc_val: float = 0.5, frame_range: list = [250, 300]):
    """
    Main function to run the PC heatmap workflow.
    For each dataset with Diff AE manifest data, this function:
        - generates histograms of PC features (saves .png out to .results/crop_visualization/figs/)
        - filters for crops based on the given input value of a given PC dimension
           and a given range of timepoints and does the following:
                - gets the corresponding crops from the original images
                    (saves .tiff files out to .results/crop_visualization/figs/original_crops/)
                - reconstructs the crops by passing the PC space coordinates
                    through the Diff AE image generation model
                    (saves .tiff files out to .results/crop_visualization/figs/reconstructed_crops/)
                - saves the filtered dataframe of crops to a .csv file
                    (saves .csv file out to .results/crop_visualization/outputs/)

    Inputs:
    - pc_to_explore: int
        The principal component to filter by (0-indexed).
        For example, if you want to filter by a particular value of
          the 2nd PC, set this to 1.
    - pc_val: float
        The value of the PC dimension to filter by.
        For example, if you want to filter by crops with pc_to_explore
          component of the PC coordinates ~= 0.5, set this to 0.5.
          The filtering is done by binning the PCs and
            getting the bin index that contains pc_val.
    - frame_range: list of int
        The range of timepoints to filter by.
        For example, if you want to filter by crops between frames 225 and 275,
          set this to [225, 275].

    Outputs:
    - None, but saves out images and csv files to the appropriate directories.
        (See above for details.)

    """

    # get output subdirectory for intermediate workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_name = "crop_visualization"
    workflow_output_folder = f"{workflow_name}/outputs"
    output_savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = f"{workflow_name}/figs"
    fig_savedir = get_output_path(workflow_fig_folder)
    orig_crop_savedir = get_output_path(fig_savedir + "original_crops")
    recon_crop_savedir = get_output_path(fig_savedir + "reconstructed_crops")

    list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

    num_bins = 40  # number of bins for histogram, hardcoded right now but somewhat arbitrary

    pca = fit_pca()
    bin_limits = component_heatmaps.set_8d_bounds_from_data(list_of_datasets, pca)

    for ds_name in list_of_datasets:
        print(f"Processing dataset: {ds_name}")
        # get manifest data with crop index column added
        # but not projected to PCA space (keep original feature space)
        df = get_manifest_for_dynamics_workflows(ds_name, pca=None)
        hist_array, bin_edges, df = component_heatmaps.get_histogram_by_component(
            df, num_bins, bin_limits=bin_limits[:3], feat_cols=manifest_io.get_feature_cols(df)[:3]
        )

        # plot histogram of PCs for each component
        fig, _ = plot_principal_component_histogram(hist_array, bin_edges)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir + f"{ds_name}_pc_histogram")

        if ds_name != "20250319_20X":
            continue
        # get dataframe of crops with bin_{latent_dim} == bin_index(latent_val),
        # where bin_index is the index of the bin that contains latent_val
        # in the bin edges over the given latent dimension
        df_filtered = component_heatmaps.get_df_by_bin_value(df, pc_to_explore, pc_val, bin_edges)

        # select timepoints within the given range
        df_filtered = df_filtered[
            (df_filtered["frame_number"] >= frame_range[0])
            & (df_filtered["frame_number"] <= frame_range[1])
        ]

        num_filtered_points = df_filtered.shape[0]
        print(
            f"Number of crops in bin along PC {pc_to_explore+1} "
            + f"containing value {pc_val} between frames "
            + f"{frame_range[0]} and {frame_range[1]}: {num_filtered_points}"
        )

        # for now, only save out up to 10 (testing workflow)
        if num_filtered_points > 12:
            print(f"Number of crops in bin exceeds 12, limiting to 12 for testing")
            num_filtered_points = 12
        # get the first num_filtered_points coordinates from the dataframe
        df_filtered = df_filtered.iloc[:num_filtered_points]

        # save out dataframe to csv
        df_filtered.to_csv(output_savedir + f"{ds_name}_dataframe.csv")

        # get and save out crops corresponding to
        # the rows in the filtered dataframe
        original_crop_list = get_original_crops_in_dataframe(
            df_filtered,
            orig_crop_savedir,
        )

        fig, _ = plot_crop_montage(original_crop_list)
        viz_base.save_plot(fig, fig_savedir + f"{ds_name}_original_crops_montage")

        # get and save out reconstructed crops
        # corresponding to the rows in the filtered dataframe
        # reconstructed_crop_array = get_reconstructed_crops_in_dataframe(
        #     df_filtered,
        #     recon_crop_savedir,
        #     )


if __name__ == "__main__":
    fire.Fire(main)
