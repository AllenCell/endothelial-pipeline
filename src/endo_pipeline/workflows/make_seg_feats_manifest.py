import logging
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.endo_pipeline.configs.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_measured_segmentation_table,
    ipython_cli_flexecute,
)
from src.endo_pipeline.io import configure_logging, get_output_path, load_dataframe
from src.endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_cell_segmentation_path_column,
    add_filter_columns,
    calculate_derived_data_dynamics_dependent,
    calculate_derived_data_dynamics_independent,
    filter_and_save_track_data_for_landscape_integration,
    merge_measured_segmentation_features_tables,
)
from src.endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

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
    tracking_df = get_measured_segmentation_table(
        dataset_name_list=[dataset_name],
        kind="cdh5_tracking",
    )

    segprops_manifest = load_dataframe_manifest("cdh5_classic_segmentation")
    segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset_name)
    segprops_df = load_dataframe(segprops_location)

    nucprops_df = get_measured_segmentation_table(
        dataset_name_list=[dataset_name],
        kind="nuclei_labelfree",
    )
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

    big_table = add_cell_segmentation_path_column(big_table)
    big_table = calculate_derived_data_dynamics_independent(big_table)

    # add the size of the crop used to get DiffAE features at full res
    crop_size = 256
    big_table["crop_size"] = crop_size

    # filter the segprops data to remove regions that
    # touch the image borders and keep only tracks that
    # have a minimum number of datapoints after this
    logger.info(
        "Filtering out regions that touch the image borders and tracks that are too short..."
    )

    big_table = add_filter_columns(big_table, out_dir, min_track_duration=24, max_area_change=0.1)
    big_table_filtered = big_table[~big_table["filter_global"]]

    # NOTE THIS TABLE WILL BE UPLOADED TO FMS
    # save the raw combined data tables
    # (we want to have an accessible version of the raw data)
    out_dir_raw = out_dir / "segmentation_features_manifests/"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_path_raw = out_dir_raw / f"{dataset_name}_live_segmentation_features.tsv"
    big_table.to_csv(out_path_raw, sep="\t", index=False)

    # add some columns that are calculated from the
    # existing columns include:
    # orientation in degrees, velocities, nematic order,
    # aspect ratio, number of tracks (i.e. approximate
    # number of detected cells)
    logger.info("Calculating dynamics-dependent metrics from existing measurements...")

    big_table_filtered = calculate_derived_data_dynamics_dependent(
        big_table_filtered.copy(deep=True)
    )

    # create a subset of the data that is used for cell track integration
    logger.info("Outputting a subset of the cell tracking data for integration with landscapes...")

    out_dir_for_integration = Path(out_dir) / "single_cell_track_integration/"
    out_dir_for_integration.mkdir(parents=True, exist_ok=True)
    out_path_integration_table = (
        out_dir_for_integration / f"{dataset_name}_single_cell_track_integration.csv"
    )
    filter_and_save_track_data_for_landscape_integration(
        big_table_filtered,
        out_path_integration_table,
        crop_size=crop_size,
        min_num_points_per_track=120,
        return_df=False,
    )


def main(
    dataset_name: str | None = None,
    n_proc: int = 1,
    verbose: bool = False,
) -> None:

    # set the directory where the output will be saved
    out_dir = get_output_path(Path(__file__).stem)
    configure_logging(out_dir, logger, verbose)

    # create a list of datasets to analyze if not provided
    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
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
            desc="Processing datasets (1P)",
            unit="datasets",
        ):
            create_segmentation_measured_feature_manifest(dataset_name, out_dir)


if __name__ == "__main__":
    ipython_cli_flexecute(main)
