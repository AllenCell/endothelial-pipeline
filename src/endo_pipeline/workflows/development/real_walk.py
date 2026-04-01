from endo_pipeline.cli import Datasets
from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED


def main(
    datasets: Datasets | None = None,
    pc_axis_list: list[int] = [0, 1, 2],
    pc_val_list: list[float] = [-1, -0.5, 0, 0.5, 1],
    random_seed: int = RANDOM_SEED,
    plot_heatmap: bool = False,
    n_pcs_to_analyze: int = NUM_PCS_TO_ANALYZE,
    origin_tolerance: float = 0.25,
) -> None:
    """
    Generate a real walk of cropped images within a specified range of PC values. The crops are
    selected such that one principal component axis varies while the others remain near zero.

    #pc-interpretation #diffae-image-generation

    Note if you want to do the top 8:
    set pc_axis_list to [0, 1, 2, 3, 4, 5, 6, 7]
    set n_pcs_to_analyze to 8
    and increase origin_tolerance to 0.3

    Parameters
    ----------
    datasets
        Optional, list of datasets or dataset collections to load images from.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the most recent run.
    include_cell_piling
        True to include timepoints with cell piling to fit the PCA model, False to exclude them.
    pc_axis_list
        The principal component axis to use for filtering the images (0 for PC1, 1 for PC2, etc.)
    pc_val_list
        The value of the principal component axis to filter the images by.
    random_seed
        Random seed for sampling crops from the filtered DataFrame.
    plot_heatmap
        True to plot a heatmap of the principal component values, False to skip plotting.
    n_pcs_to_analyze
        Number of principal components to analyze. Defaults to NUM_PCS_TO_ANALYZE which corresponds
        to the first 3 principal components. If set to a different value, ensure that pc_axis_list
        only contains indices that are less than the n_pcs_to_analyze.
    origin_tolerance
        Tolerance around zero for other principal components when filtering crops in units of PC value.

    Returns
    -------
    :
        Saves the contact sheet of cropped images to the output directory.
    """
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, load_image, save_plot_to_path
    from endo_pipeline.library.analyze.numerics.binning import (
        get_df_by_bin_value,
        get_histogram_by_component,
    )
    from endo_pipeline.library.process.image_processing import max_proj, std_dev
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_label_for_column,
        plot_component_histograms_over_time,
    )
    from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
    from endo_pipeline.library.visualize.real_walk import sample_dataframe
    from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_LATENT_FEATURES,
    )
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
    from endo_pipeline.settings.plot_defaults import CROP_HIST_BIN_WIDTH
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
    )

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided. Otherwise, use the provided list.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # get dataframe manifest corresponding to the model that generated the features
    # get dataframe manifest for crop-based features
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fig_savedir = get_output_path(__file__)

    df_list = []
    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset %s not found in dataframe manifest %s. Skipping this dataset.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue
        dataframe_location = feature_dataframe_manifest.locations[dataset_name]
        df_dataset = load_dataframe(dataframe_location)
        df_list.append(df_dataset)
    df = pd.concat(df_list, ignore_index=True)

    feat_cols = DIFFAE_PC_COLUMN_NAMES[:n_pcs_to_analyze]
    bin_limits = [(df[feat_col].min(), df[feat_col].max()) for feat_col in feat_cols]
    hist_array_lists, bin_edges, df_with_bins = get_histogram_by_component(
        df,
        CROP_HIST_BIN_WIDTH,
        bin_limits,
        feat_cols=feat_cols,
    )

    if plot_heatmap:
        feat_labels = [
            get_label_for_column(col_name) for col_name in DIFFAE_PC_COLUMN_NAMES[:n_pcs_to_analyze]
        ]
        for i, dataset_name in enumerate(datasets):
            fig, _ = plot_component_histograms_over_time(
                hist_array_lists[i], bin_edges, feature_names=feat_labels
            )
            fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name}_pc_histogram")

    samples = []
    for pc_axis in pc_axis_list:
        for pc_val in pc_val_list:

            df_filtered = get_df_by_bin_value(df_with_bins, pc_axis, pc_val, bin_edges)

            logger.info("%d crops for PC %s around value %s", len(df_filtered), pc_axis, pc_val)

            for i in range(NUM_LATENT_FEATURES):
                if i != pc_axis:
                    pc_col = DIFFAE_PC_COLUMN_NAMES[i]
                    df_filtered = df_filtered[
                        (df_filtered[pc_col] >= -origin_tolerance)
                        & (df_filtered[pc_col] <= origin_tolerance)
                    ]
            logger.info("%d crops for other PCs near zero", len(df_filtered))
            if len(df_filtered) == 0:
                logger.warning(
                    "No crops found for PC %s value %s. Try increasing the origin_tolerance.",
                    pc_axis,
                    pc_val,
                )
                continue

            # one example selected at random
            df_sample = sample_dataframe(df_filtered, n_num_crops=1, random_seed=random_seed)
            samples.append((pc_axis, pc_val, df_sample))

    crops_bf_std_deviation = []
    crops_gfp_max_projection = []

    for pc_axis, pc_val, df_sample in samples:
        logger.info(f"Processing sample for PC {pc_axis} value {pc_val}")
        dataset = df_sample[Column.DATASET].iloc[0]
        dataset = cast(str, dataset)  # Ensure dataset is a string
        dataset_config = load_dataset_config(dataset)
        position = df_sample[Column.POSITION].iloc[0]
        position = cast(str, position)  # Ensure position is a string
        position_integer = int(position[-1])  # Extract the position number from the string
        timepoint = df_sample[Column.TIMEPOINT].iloc[0]

        img_loc = get_zarr_location_for_position(dataset_config, position_integer)
        img = load_image(img_loc, timepoints=[timepoint], level=1, squeeze=True)
        # crop
        start_x = df_sample["start_x"].iloc[0]
        start_y = df_sample["start_y"].iloc[0]
        crop_size_x = df_sample["crop_size_x"].iloc[0]
        crop_size_y = df_sample["crop_size_y"].iloc[0]

        crop = img[:, :, start_y : start_y + crop_size_y, start_x : start_x + crop_size_x]

        # Extract channels once, these channel indices are hardcoded
        # because we defined the order of channels in the zarr
        bf_channel = crop[1, :, :, :].squeeze()
        gfp_channel = crop[0, :, :, :].squeeze()

        # Process channels
        std_dev_proj = std_dev(bf_channel, 0)
        log_norm_std = np.log1p(std_dev_proj)
        low, high = np.percentile(log_norm_std, [0.1, 99.9])
        clipped_std = np.clip(log_norm_std, low, high)
        crops_bf_std_deviation.append(clipped_std)

        cdh5_max_proj = max_proj(gfp_channel, 0)
        low, high = np.percentile(cdh5_max_proj, [10, 98])
        clipped_cdh5 = np.clip(cdh5_max_proj, low, high)
        crops_gfp_max_projection.append(clipped_cdh5)

    # Create panels doing the len of pc_val_list every other contrasted crop type
    panels = []
    for i in range(len(pc_axis_list)):
        for j in range(len(pc_val_list)):
            index = i * len(pc_val_list) + j
            panels.append(crops_bf_std_deviation[index])
        for j in range(len(pc_val_list)):
            index = i * len(pc_val_list) + j
            panels.append(crops_gfp_max_projection[index])

    row_titles = []
    for pc_axis in pc_axis_list:
        row_titles.append(f"PC{pc_axis + 1}\nBF Std Dev")
        row_titles.append(f"PC{pc_axis + 1}\n VE-cad MIP")

    fig = make_contact_sheet(
        panels,
        max_rows=len(pc_axis_list) * 2,
        max_cols=len(pc_val_list),
        col_titles=[str(val) for val in pc_val_list],
        row_titles=row_titles,
        fig_kwargs={"layout": "tight", "figsize": (MAX_FIGURE_WIDTH, len(pc_axis_list) * 2)},
        font_size=FONTSIZE_SMALL,
    )
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    scale_bar_um = 20
    add_scalebar(
        fig.axes[0],
        scale_bar_um=scale_bar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        bar_thickness=10,
        padding=10,
    )

    plt.show()
    save_plot_to_path(
        fig,
        fig_savedir,
        f"real_walk_diffae_pc_{'_'.join(map(str, pc_axis_list))}_{scale_bar_um}um_scalebar",
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
