import logging
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from endo_pipeline.io import configure_logging, get_output_path, load_dataframe
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_filter_columns,
    calculate_derived_data_dynamics_independent,
    merge_measured_segmentation_features_tables,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

logger = logging.getLogger(__name__)


def create_seg_measured_feat_manifest_multiproc_wrapper(args: Sequence) -> None:
    dataset_name, out_dir = args
    create_segmentation_measured_feature_manifest(dataset_name, out_dir)


def create_segmentation_measured_feature_manifest(
    dataset_name: str,
    out_dir: str | Path,
) -> None:

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

    nucprops_manifest = load_dataframe_manifest("nuclei_label_free_segmentation")
    nucprops_location = get_dataframe_location_for_dataset(nucprops_manifest, dataset_name)
    nucprops_df = load_dataframe(nucprops_location)

    if tracking_df.empty or segprops_df.empty or nucprops_df.empty:
        logger.info(
            f"No tracking data or segmentation properties data found for {dataset_name}. Skipping..."
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

    # NOTE THIS TABLE WILL BE UPLOADED TO FMS
    # save the raw combined data tables
    # (we want to have an accessible version of the raw data)
    out_dir_raw = out_dir / "segmentation_features_dataframes/"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_path_raw = out_dir_raw / f"{dataset_name}_live_segmentation_features.parquet"
    big_table.to_parquet(out_path_raw, index=False)


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    verbose: bool = False,
) -> None:

    # set the directory where the output will be saved
    out_dir = get_output_path(__file__)
    configure_logging(out_dir, logger, verbose)

    # create a list of datasets to analyze if not provided
    if datasets is None:
        dataset_name_list = get_datasets_in_collection("pca_reference")
    else:
        dataset_name_list = datasets

    logger.info(f"datasets to analyze: {dataset_name_list}")

    # decide whether or not to use multiprocessing
    # and then create merged tables for each dataset
    if n_proc > 1:
        n_proc = min(n_proc, len(dataset_name_list))
        with Pool(processes=n_proc) as pool:
            args = zip(
                dataset_name_list,
                [out_dir] * len(dataset_name_list),
                strict=False,
            )
            list(
                tqdm(
                    pool.imap(create_seg_measured_feat_manifest_multiproc_wrapper, args),
                    total=len(dataset_name_list),
                    desc="Processing datasets (MP)",
                    unit="datasets",
                )
            )
            pool.close()
            pool.join()
    else:
        for dataset_name in tqdm(
            dataset_name_list,
            total=len(dataset_name_list),
            desc="Processing datasets",
            unit="datasets",
        ):
            create_segmentation_measured_feature_manifest(dataset_name, out_dir)


if __name__ == "__main__":
    ipython_cli_flexecute(main)
