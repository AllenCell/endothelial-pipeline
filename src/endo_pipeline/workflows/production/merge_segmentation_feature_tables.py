from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Merge CDH5 segmentation, CDH5 tracking, and labelfree nuclei feature tables.

    #cdh5-segmentation #cdh5-tracking #nuclei-prediction

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe merge-segmentation-feature-tables -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe merge-segmentation-feature-tables --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will merge
    dataframes for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to merge.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_cell_piling_and_steady_state_annotation_columns,
        add_filter_columns,
        calculate_derived_data_dynamics_independent,
        merge_measured_segmentation_features_tables,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

    logger = logging.getLogger(__name__)

    out_dir = get_output_path("merged_segmentation_features")

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")

    if DEMO_MODE:
        datasets = datasets[:1]

    # Load all dataframe manifests
    segprops_manifest = load_dataframe_manifest("cdh5_classic_segmentation")
    tracking_manifest = load_dataframe_manifest("cdh5_classic_segmentation_tracking")
    nucprops_manifest = load_dataframe_manifest("nuclei_labelfree_segmentation")

    for dataset_name in datasets:
        logger.info("Starting feature table merge for dataset '%s' ...", dataset_name)

        # Get locations for current dataset
        segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset_name)
        tracking_location = get_dataframe_location_for_dataset(tracking_manifest, dataset_name)
        nucprops_location = get_dataframe_location_for_dataset(nucprops_manifest, dataset_name)

        # Load dataframes for current dataset
        segprops_df = load_dataframe(segprops_location)
        tracking_df = load_dataframe(tracking_location)
        nucprops_df = load_dataframe(nucprops_location)

        # Skip this dataset if any of the above dataframes are empty
        if segprops_df.empty:
            logger.warning("No CDH5 segmentation data found for '%s'. Skipping.", dataset_name)
            continue
        if tracking_df.empty:
            logger.warning("No CDH5 tracking data found for '%s'. Skipping.", dataset_name)
            continue
        if nucprops_df.empty:
            logger.warning("No labelfree nuclei data found for '%s'. Skipping.", dataset_name)
            continue

        # Combine the tracking data with the segmentation properties data
        logger.info("Combining tracking data with segmentation properties data...")
        big_table = merge_measured_segmentation_features_tables(
            segprops_df, tracking_df, nucprops_df
        )

        # Add some columns to the data table that are calculated from existing
        # columns and do not depend on dynamics / require clean tracks
        logger.info("Calculating dynamics-independent metrics from existing measurements...")
        big_table = calculate_derived_data_dynamics_independent(big_table)

        # Filter to remove regions that touch the image borders and keep only
        # tracks that have a minimum number of datapoints after this
        logger.info("Filtering out regions touching image borders and tracks that are too short...")
        big_table = add_filter_columns(
            big_table, out_dir, min_track_duration=24, max_area_change=0.1
        )
        big_table = add_cell_piling_and_steady_state_annotation_columns(big_table)
        big_table = big_table.reset_index(drop=True)

        # Save merged table out to local path
        filename = f"{dataset_name}_live_segmentation_features.parquet"
        big_table.to_parquet(out_dir / filename, index=False)

        logger.info("Finished merging feature tables for dataset '%s'", dataset_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
