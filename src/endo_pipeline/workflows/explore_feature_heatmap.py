# %%
import fire
import matplotlib.pyplot as plt
import pandas as pd

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
def main(
    dataset_names: str | list[str] | None = None,
    pc_axis: int = 1,
    pc_val: float = 0.25,
    frame_range: list | None = None,
) -> None:
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
    - pc_axis: int
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
    # first return argument is the heatmap array
    # rename _ to hist_array and uncomment lines 114-116
    # if we want to viz the heatmap again
    _, bin_edges, df_with_bins = component_heatmaps.get_histogram_by_component(
        df_all_datasets,
        num_bins,
        bin_limits=bin_limits,
        feat_cols=manifest_io.get_feature_cols(df)[:3],
    )

    # comment out heat map viz
    # plot histogram of PCs for each component
    # fig, _ = plot_principal_component_histogram(hist_array, bin_edges)
    # fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
    # viz_base.save_plot(fig, fig_savedir + f"{ds_name}_pc_histogram")

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

    # optional print statement?
    # might want to remove this
    num_filtered_points = df_filtered.shape[0]
    print(
        f"Number of crops in bin along PC{pc_axis+1} "
        + f"containing value {pc_val} between frames "
        + f"{frame_range[0]} and {frame_range[1]}: {num_filtered_points}"
    )

    # save out dataframe to csv
    # do we need to do this?
    # probably not
    df_filtered.to_csv(
        output_savedir + f"dataframe_PC{pc_axis+1}_val" + "p".join(str(pc_val).split(".")) + ".csv"
    )

    # get and save out crops corresponding to
    # the rows in the filtered dataframe
    original_crop_list = get_original_crops_in_dataframe(df_filtered)

    fig, _ = plot_crop_montage(original_crop_list)
    fig.suptitle(f"PC{pc_axis+1} value: {pc_val}", y=1.0, fontsize=45)
    plt.tight_layout()
    plt.show()
    viz_base.save_plot(
        fig,
        fig_savedir
        + f"{ds_name}_original_bf_crops_montage_"
        + f"PC{pc_axis+1}_val"
        + "p".join(str(pc_val).split(".")),
    )

    # cdh5 contact sheet
    fig, _ = plot_crop_montage(original_crop_list, channel_index=2)
    fig.suptitle(f"PC{pc_axis+1} value: {pc_val}", y=1.0, fontsize=45)
    plt.tight_layout()
    plt.show()
    viz_base.save_plot(
        fig,
        fig_savedir
        + f"{ds_name}_original_cdh5_crops_montage_"
        + f"PC{pc_axis+1}_val"
        + "p".join(str(pc_val).split(".")),
    )

    # get reconstructed ve-cad crops
    # corresponding to the rows in the filtered dataframe
    # add in try / except checking if GPU is available
    reconstructed_crop_list = get_reconstructed_crops_in_dataframe(df_filtered)

    fig, _ = plot_crop_montage(reconstructed_crop_list, channel_index=None)
    fig.suptitle(f"PC{pc_axis+1} value: {pc_val}", y=1.0, fontsize=45)
    plt.tight_layout()
    plt.show()
    viz_base.save_plot(
        fig,
        fig_savedir
        + f"{ds_name}_reconstructed_cdh5_crops_montage_"
        + f"PC{pc_axis+1}_val"
        + "p".join(str(pc_val).split(".")),
    )


if __name__ == "__main__":
    fire.Fire(main)
