from collections.abc import Sequence

from endo_pipeline.cli import Datasets

TAGS = ["cdh5_segmentation", "tracking"]


def run_workflow(queue: Sequence) -> None:
    """
    Run the tracking workflow using a queue.
    The queue is a tuple of (dataset_name, position) and a dataframe.
    The dataframe contains the parameters for the workflow and is built using build_analysis_queue.
    """
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.configs.dataset_io import extract_T, get_zarr_name, get_zarr_path
    from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
    from endo_pipeline.library.process.lib_tracking import run_tracking
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    logger = logging.getLogger(__name__)

    (dataset_name, position), queue_df = queue
    timepoints_to_eval = queue_df["T"].tolist()
    position = sequence_to_scalar(queue_df["position"])
    image_validation_frequency = sequence_to_scalar(queue_df["image_validation_frequency"])
    validation_image = sequence_to_scalar(queue_df["is_validation_image"])
    verbose = sequence_to_scalar(queue_df["verbose"])
    out_dir = sequence_to_scalar(queue_df["output_dir"]) / f"{dataset_name}/P{position}"
    out_filename_prefix = f"{dataset_name}_P{position}"

    # get the segmentation images
    dataset = load_dataset_config(dataset_name)
    manifest = load_image_manifest("cdh5_classic_seg")
    seg_locations = [
        get_image_location_for_dataset(manifest, dataset, position, timepoint)
        for timepoint in range(dataset.duration)
    ]
    seg_filepaths = [location.path for location in seg_locations if location.path is not None]

    segmentation_channel = 0  # the segmentation images only contain a single channel

    # run the tracking workflow
    if seg_filepaths:
        if validation_image:
            # get the raw cadherin channel from either original data or the zarr version
            raw_channel = 0  # zarr files are created such that the first channel is always Cdh5
            zarr_name = get_zarr_name(dataset_name, position)
            zarr_path = get_zarr_path(dataset_name, zarr_name)[zarr_name]
            raw_filepath = Path(zarr_path)
        else:
            raw_filepath = None
            raw_channel = 0

        run_tracking(
            in_dir=seg_filepaths,
            out_dir=out_dir,
            out_filename_prefix=out_filename_prefix,
            tracking_metrics=["region_overlap"],  # this can be changed to ['centroids'] if desired
            sorting_key=extract_T,
            C=segmentation_channel,
            T=timepoints_to_eval,
            extra_in_dir=raw_filepath,
            extra_C=raw_channel,
            extra_T=timepoints_to_eval,
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
    is_test: bool = False,
    verbose: bool = False,
) -> None:
    """Run the tracking workflow on a dataset, a list of datasets, or a dataset collection."""
    import logging
    from multiprocessing import Pool

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.configs.dataset_io import concatenate_and_save_feature_tables
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue

    logger = logging.getLogger(__name__)

    out_dir = get_output_path(__file__)

    logger.info(f"datasets analyzed: {datasets}")

    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
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

    from endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    ipython_cli_flexecute(main)
