from pathlib import Path

from endo_pipeline.cli import Datasets


def create_segmentation_measured_feature_manifest(
    dataset_name: str,
    out_dir: str | Path,
) -> None:
    """Merge nuclei measurement, cdh5 segmentation measurement, and tracking tables into 1 table."""
    import logging

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_cell_piling_and_steady_state_annotation_columns,
        add_filter_columns,
        calculate_derived_data_dynamics_independent,
        merge_measured_segmentation_features_tables,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

    logger = logging.getLogger(__name__)

    # make the output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the tracking data and the segmentation feature data
    tracking_manifest = load_dataframe_manifest("cdh5_classic_segmentation_tracking")
    tracking_location = get_dataframe_location_for_dataset(tracking_manifest, dataset_name)
    tracking_df = load_dataframe(tracking_location)

    segprops_manifest = load_dataframe_manifest("cdh5_classic_segmentation")
    segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset_name)
    segprops_df = load_dataframe(segprops_location)

    nuc_seg_manifest_name = "nuclei_labelfree_segmentation"
    nucprops_manifest = load_dataframe_manifest(nuc_seg_manifest_name)
    nucprops_location = get_dataframe_location_for_dataset(nucprops_manifest, dataset_name)
    nucprops_df = load_dataframe(nucprops_location)

    if tracking_df.empty or segprops_df.empty or nucprops_df.empty:
        logger.info(
            f"No tracking data or segmentation properties data found for {dataset_name}. Skipping."
        )
        return
    else:
        logger.info(f"Working on {dataset_name}...")

    # combine the tracking data with the segmentation
    # properties data
    logger.info("Combining tracking data with segmentation properties data...")
    big_table = merge_measured_segmentation_features_tables(segprops_df, tracking_df, nucprops_df)

    # add some columns to the data table that are
    # calculated from existing columns and do not
    # depend on dynamics / require clean tracks
    logger.info("Calculating dynamics-independent metrics from existing measurements...")

    big_table = calculate_derived_data_dynamics_independent(big_table)

    # filter the segprops data to remove regions that
    # touch the image borders and keep only tracks that
    # have a minimum number of datapoints after this
    logger.info(
        "Filtering out regions that touch the image borders and tracks that are too short..."
    )

    big_table = add_filter_columns(big_table, out_dir, min_track_duration=24, max_area_change=0.1)

    big_table = add_cell_piling_and_steady_state_annotation_columns(big_table)

    big_table = big_table.reset_index(drop=True)

    # NOTE THIS TABLE WILL BE UPLOADED TO FMS
    # save the raw combined data tables
    # (we want to have an accessible version of the raw data)
    filename = f"{dataset_name}_live_segmentation_features.parquet"

    out_dir_raw = out_dir / "segmentation_features_dataframes/"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_path_raw = out_dir_raw / filename
    big_table.to_parquet(out_path_raw, index=False)


def main(
    datasets: Datasets,
) -> None:
    """Run workflow for merging nuclei, cdh5 segmentation, and tracking data into a single table."""

    import logging

    from tqdm import tqdm

    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path

    logger = logging.getLogger(__name__)

    # set the directory where the output will be saved
    out_dir = get_output_path(__file__)

    datasets = use_default_collection(datasets, "live_cdh5_seg_based_feat_datasets")
    logger.info(f"datasets to analyze: {datasets}")

    # create merged tables for each dataset
    for dataset_name in tqdm(
        datasets,
        total=len(datasets),
        desc="Processing datasets",
        unit="datasets",
    ):
        create_segmentation_measured_feature_manifest(dataset_name, out_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
