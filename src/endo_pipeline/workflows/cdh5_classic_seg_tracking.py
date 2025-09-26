import logging
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.configs.dataset_io import (
    concatenate_and_save_feature_tables,
    extract_T,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
)
from endo_pipeline.io import configure_logging, get_output_path
from endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    sequence_to_scalar,
)
from endo_pipeline.library.process.lib_tracking import run_tracking
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

logger = logging.getLogger(__name__)


def run_workflow(queue: Sequence) -> None:
    (dataset_name, position), queue_df = queue
    T_to_eval = queue_df["T"].tolist()
    position = sequence_to_scalar(queue_df["position"])
    image_validation_frequency = sequence_to_scalar(queue_df["image_validation_frequency"])
    validation_image = sequence_to_scalar(queue_df["validation_image"])
    verbose = sequence_to_scalar(queue_df["verbose"])
    out_dir = sequence_to_scalar(queue_df["output_dir"]) / f"{dataset_name}/P{position}"
    out_filename_prefix = f"{dataset_name}_P{position}"
    use_sldy_data = sequence_to_scalar(queue_df["use_sldy_data"])

    # get the segmentation images
    dataset = load_dataset_config(dataset_name)
    manifest = load_image_manifest("cdh5_classic_seg")
    seg_locations = [
        get_image_location_for_dataset(manifest, dataset_name, position, timepoint)
        for timepoint in range(dataset.duration)
    ]
    seg_filepaths = [location.path for location in seg_locations if location.path is not None]

    segmentation_channel = 0  # the segmentation images only contain a single channel

    # run the tracking workflow
    if seg_filepaths:
        if validation_image:
            # get the raw cadherin channel from either original data or the zarr version
            scene_index = int(sequence_to_scalar(queue_df["scene_index"]))
            if use_sldy_data:
                dataset_config = load_dataset_config(dataset_name)
                raw_channel = dataset_config.original_channel_indices.channel_488
                raw_filepath = Path(get_original_path(dataset_name))
            else:
                raw_channel = 0  # zarr files are created such that the first channel is always Cdh5
                zarr_name = get_zarr_name(dataset_name, position)
                zarr_path = get_zarr_path(dataset_name, zarr_name)[zarr_name]
                raw_filepath = Path(zarr_path)
        else:
            scene_index = None
            raw_filepath = None
            raw_channel = 0

        run_tracking(
            in_dir=seg_filepaths,
            out_dir=out_dir,
            out_filename_prefix=out_filename_prefix,
            tracking_metrics=["region_overlap"],  # this can be changed to ['centroids'] if desired
            sorting_key=extract_T,
            C=segmentation_channel,
            T=T_to_eval,
            extra_in_dir=raw_filepath,
            extra_C=raw_channel,
            extra_scene=scene_index,
            extra_T=T_to_eval,
            Z_projection=np.max,
            track_tolerance=3,
            image_validation_frequency=image_validation_frequency,
            verbose=verbose,
        )

        # add the dataset name and position to the output table
        tracking_table = pd.read_parquet(out_dir / f"{out_filename_prefix}_tracking.parquet")
        tracking_table["dataset_name"] = dataset_name
        tracking_table["position"] = position
        tracking_table.to_parquet(out_dir / f"{out_filename_prefix}_tracking.parquet", index=False)

    else:
        logger.info(
            f"""
            No segmentation images found for {dataset_name}. Skipping tracking analysis.
            If this is unexpected check that the IS_TEST argument is set to False.
            """
        )
        return


def main(
    datasets: Datasets,
    n_proc: int = 1,
    save_output: bool = True,
    use_sldy_data: bool = False,
    is_test: bool = False,
    verbose: bool = False,
) -> None:

    out_dir = get_output_path(__file__)

    configure_logging(out_dir, logger, verbose=verbose)
    logger.info(f"datasets analyzed: {datasets}")

    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
        use_sldy_data=use_sldy_data,
    )

    analysis_queue_df = pd.DataFrame(analysis_queue)
    analysis_queue_per_position = list(analysis_queue_df.groupby(["dataset_name", "position"]))

    if n_proc > 1:
        if __name__ == "__main__":
            n_proc = min(n_proc, len(analysis_queue_per_position))
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(run_workflow, analysis_queue_per_position, chunksize=1),
                        total=len(analysis_queue_per_position),
                        desc="Tracking (MP)",
                    )
                )
                pool.close()
                pool.join()
    else:
        for queue in tqdm(
            analysis_queue_per_position,
            total=len(analysis_queue_per_position),
            desc="Tracking (1P)",
        ):
            run_workflow(queue)

    if save_output:
        for dataset_name in tqdm(
            datasets, desc="Replacing individual tables with combined table..."
        ):
            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="tracking",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("...done analysis.")
    print("\N{MICROSCOPE}")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
