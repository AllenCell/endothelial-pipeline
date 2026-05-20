from endo_pipeline.cli import Datasets
from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.workflow_defaults import (
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    RANDOM_SEED,
)


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

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, load_image, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_binned_value
    from endo_pipeline.library.analyze.numerics.binning import get_histogram_by_component
    from endo_pipeline.library.process.image_processing import max_proj, std_dev
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        plot_component_histograms_over_time,
    )
    from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
    from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
    from endo_pipeline.settings.plot_defaults import CROP_HIST_BIN_WIDTH
    from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided. Otherwise, use the provided list.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # get dataframe manifest corresponding to the model that generated the features
    # get dataframe manifest for crop-based features
    feature_dataframe_manifest_name = GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME
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
    hist_array_lists, bin_edges = get_histogram_by_component(
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
    for feat_col, bin_edge_array in zip(feat_cols, bin_edges, strict=True):
        for pc_val in pc_val_list:
            df_filtered = filter_dataframe_to_binned_value(df, feat_col, pc_val, bin_edge_array)

            logger.info("%d crops for PC %s around value %s", len(df_filtered), feat_col, pc_val)

            for other_feat_col in feat_cols:
                if other_feat_col != feat_col:
                    df_filtered = df_filtered[
                        (df_filtered[other_feat_col] >= -origin_tolerance)
                        & (df_filtered[other_feat_col] <= origin_tolerance)
                    ]
            logger.info("%d crops for other PCs near zero", len(df_filtered))
            if len(df_filtered) == 0:
                logger.warning(
                    "No crops found for PC %s value %s. Try increasing the origin_tolerance.",
                    feat_col,
                    pc_val,
                )
                continue

            # one example selected at random
            num_crop_samples = 1
            df_sample = df_filtered.sample(
                n=num_crop_samples, random_state=random_seed, replace=False
            )
            samples.append((feat_col, pc_val, df_sample))

    crops_bf_std_deviation = []
    crops_gfp_max_projection = []

    for feat_col, pc_val, df_sample in samples:
        logger.info(f"Processing sample for PC {feat_col} value {pc_val}")
        dataset_name = df_sample[Column.DATASET].iloc[0]
        dataset_config = load_dataset_config(dataset_name)
        position = df_sample[Column.POSITION].iloc[0]
        timepoint = df_sample[Column.TIMEPOINT].iloc[0]

        img_loc = get_zarr_location_for_position(dataset_config, position)
        img = load_image(img_loc, timepoints=[timepoint], level=1, squeeze=True)
        # crop
        start_x = df_sample[Column.DiffAEData.START_X].iloc[0]
        start_y = df_sample[Column.DiffAEData.START_Y].iloc[0]
        crop_size_x = df_sample[Column.DiffAEData.CROP_SIZE_X].iloc[0]
        crop_size_y = df_sample[Column.DiffAEData.CROP_SIZE_Y].iloc[0]

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
    for i in range(len(feat_cols)):
        for j in range(len(pc_val_list)):
            index = i * len(pc_val_list) + j
            panels.append(crops_bf_std_deviation[index])
        for j in range(len(pc_val_list)):
            index = i * len(pc_val_list) + j
            panels.append(crops_gfp_max_projection[index])

    row_titles = []
    for feat_col in feat_cols:
        row_titles.append(f"{feat_col}\nBF Std Dev")
        row_titles.append(f"{feat_col}\n VE-cad MIP")

    fig = make_contact_sheet(
        panels,
        max_rows=len(feat_cols) * 2,
        max_cols=len(pc_val_list),
        col_titles=[str(val) for val in pc_val_list],
        row_titles=row_titles,
        fig_kwargs={"layout": "tight", "figsize": (MAX_FIGURE_WIDTH, len(feat_cols) * 2)},
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
