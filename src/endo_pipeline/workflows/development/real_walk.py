from endo_pipeline.cli import Datasets
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    NUM_LATENT_FEATURES,
)

TAGS = ["pc_interpretation", "diffae_image_generation"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    include_cell_piling: bool = False,
    pc_axis_list: list[int] = [0, 1, 2],
    pc_val_list: list[float] = [
        -1.5,
        -1,
        -0.5,
        0,
        0.5,
        1,
        1.5,
    ],
    n_pcs_to_analyze: int = NUM_LATENT_FEATURES,
) -> None:
    """
    Generate a real walk of cropped images within a specified range of PC values. The crops are
    selected such that one principal component axis varies while the others remain near zero.

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
    n_pcs_to_analyze
        Number of principal components to analyze. Defaults to NUM_PCS_TO_ANALYZE which corresponds
        to the first 3 principal components. If set to a different value, ensure that pc_axis_list
        only contains indices that are less than the n_pcs_to_analyze.

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
    from endo_pipeline.io import get_output_path, load_image, save_plot_to_path
    from endo_pipeline.library.process.image_processing import max_proj, std_dev
    from endo_pipeline.library.visualize.crop_montage import load_data_for_montage
    from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES, ColumnName
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
    from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided. Otherwise, use the provided list.
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # get dataframe manifest corresponding to the model that generated the features
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )
    fig_savedir = get_output_path(
        "crop_visualization",
        model_manifest_name,
        run_name_,
        "include_cell_piling" if include_cell_piling else "exclude_cell_piling",
    )

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    df, pca = load_data_for_montage(
        datasets,
        dataframe_manifest,
        include_cell_piling=include_cell_piling,
        num_pcs=n_pcs_to_analyze,
    )

    samples = []
    distance_records = []

    # Compute column means once
    pc_means = df[DIFFAE_PC_COLUMN_NAMES].mean().values  # numpy array
    primary_weight = 10

    for pc_axis in pc_axis_list:
        pc_axis_col = "pc_" + str(pc_axis + 1)

        # Secondary PC columns
        secondary_cols = [col for col in DIFFAE_PC_COLUMN_NAMES if col != pc_axis_col]
        secondary_means = pc_means[[DIFFAE_PC_COLUMN_NAMES.index(col) for col in secondary_cols]]

        # Convert to NumPy arrays for speed
        primary_vals = df[pc_axis_col].values  # shape (n_rows,)
        secondary_vals = df[secondary_cols].values  # shape (n_rows, n_secondary)

        # Precompute squared differences for secondary PCs (rows x n_secondary)
        secondary_diff_sq = (secondary_vals - secondary_means) ** 2
        secondary_distance = secondary_diff_sq.sum(axis=1)  # sum across secondary PCs

        for pc_val in pc_val_list:
            total_distance = (primary_weight * (primary_vals - pc_val) ** 2) + secondary_distance

            closest_idx = np.argmin(total_distance)
            closest_row_df = df.iloc[[closest_idx]].copy()

            samples.append((pc_axis, pc_val, closest_row_df))
            distance_records.append(
                {
                    "pc_axis": pc_axis,
                    "pc_val": pc_val,
                    "closest_index": closest_idx,
                    "distance_score": total_distance[closest_idx],
                }
            )

            logger.info(
                "Selected 1 crop for PC %s close to value %s (distance %.4f)",
                pc_axis,
                pc_val,
                total_distance[closest_idx],
            )

    # Save distance CSV
    distance_df = pd.DataFrame(distance_records)
    distance_df.to_csv(fig_savedir / "pc_closest_distance_scores.csv", index=False)

    crops_bf_std_deviation = []
    crops_gfp_max_projection = []

    for pc_axis, pc_val, df_sample in samples:
        logger.info(f"Processing sample for PC {pc_axis} value {pc_val}")
        dataset = df_sample[ColumnName.DATASET].iloc[0]
        dataset = cast(str, dataset)  # Ensure dataset is a string
        dataset_config = load_dataset_config(dataset)
        position = df_sample[ColumnName.POSITION].iloc[0]
        position = cast(str, position)  # Ensure position is a string
        position_integer = int(position[-1])  # Extract the position number from the string
        timepoint = df_sample[ColumnName.TIMEPOINT].iloc[0]

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
        fig_kwargs={"layout": "tight", "figsize": (MAX_FIGURE_WIDTH, len(pc_axis_list) * 1.5)},
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
    pc_axis_list = [pc + 1 for pc in pc_axis_list]  # add 1 to pc_axis_list for indexing at 1
    save_plot_to_path(
        fig,
        fig_savedir,
        f"real_walk_diffae_pc_{'_'.join(map(str, pc_axis_list))}_{scale_bar_um}um_scalebar",
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
