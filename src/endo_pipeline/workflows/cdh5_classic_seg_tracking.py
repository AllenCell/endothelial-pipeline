import logging
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.endo_pipeline.configs import (
    get_labelfree_nuclei_prediction_model_name,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.configs.dataset_io import (
    concatenate_and_save_feature_tables,
    extract_T,
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
    save_git_versioning_info,
)
from src.endo_pipeline.io import (
    build_fms_annotations,
    configure_logging,
    get_output_path,
    upload_file_to_fms,
)
from src.endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    sequence_to_scalar,
)
from src.endo_pipeline.library.process.lib_tracking import run_tracking

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
    seg_dir = get_cdh5_classic_segmentation_path(dataset_name, position=position)
    if seg_dir is not None:
        seg_dir = Path(seg_dir)
    else:
        print(f"No segmentation directory found for {dataset_name}. Skipping tracking analysis.")
        return

    seg_filepaths = sorted(seg_dir.glob("*.ome.tif*"), key=lambda fp: extract_T(fp.name))
    segmentation_channel = 0  # the segmentation images only contain a single channel

    # run the tracking workflow
    if seg_filepaths:
        if validation_image:
            # get the raw cadherin channel from either original data or the zarr version
            scene_index = int(sequence_to_scalar(queue_df["scene_index"]))
            if use_sldy_data:
                raw_channel = get_dataset_info(dataset_name)["channel_488_index"]
                raw_filepath = Path(get_original_path(dataset_name))
            else:
                raw_channel = 0  # zarr files are created such that the first channel is always Cdh5
                zarr_name = get_zarr_name(dataset_name, position)
                zarr_path = get_zarr_path(dataset_name, zarr_name)[zarr_name]
                raw_filepath = Path(zarr_path)
        else:
            scene_index = None
            raw_filepath = None
            raw_channel = None

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
        tracking_table = pd.read_csv(out_dir / f"{out_filename_prefix}_tracking.tsv", sep="\t")
        tracking_table["dataset_name"] = dataset_name
        tracking_table["position"] = position
        tracking_table.to_csv(
            out_dir / f"{out_filename_prefix}_tracking.tsv", sep="\t", index=False
        )

    else:
        print(
            f"No segmentation images found for {dataset_name}. Skipping tracking analysis. If this is unexpected check that the IS_TEST argument is set to False."
        )
        return


def main(
    n_proc: int = 1,
    dataset_name: str | Sequence | None = None,
    save_output: bool = True,
    use_sldy_data: bool = False,
    is_test: bool = False,
    verbose: bool = False,
    upload_to_fms: bool = False,
) -> None:

    out_dir = get_output_path(Path(__file__).stem)

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    configure_logging(out_dir, logger, verbose=verbose)
    logger.info(f"datasets analyzed: {dataset_name_list}")

    analysis_queue = build_analysis_queue(
        dataset_name_list,
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
            dataset_name_list, desc="Replacing individual tables with combined table..."
        ):
            table_path_out = concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="tracking",
                file_extension=".tsv",
                remove_initial_files_and_folders=True,
            )

            if upload_to_fms:
                # upload the combined table to FMS
                dataset_config = load_dataset_config(dataset_name)
                model_name = get_labelfree_nuclei_prediction_model_name()
                model_config = load_model_config(model_name)
                annotations = build_fms_annotations(dataset_config, model=model_config)
                env: Literal["stg", "prod"] = "stg" if is_test else "prod"
                file_id = upload_file_to_fms(
                    file_path=table_path_out,
                    annotations=annotations,
                    file_type="tsv",
                    env=env,
                )
                logger.info(
                    f"Uploaded tracking table to FMS - dataset:{dataset_name}, environment: {env}, file ID: {file_id}"
                )

        # save git versioning info
        save_git_versioning_info(
            out_dir=out_dir, filename_prefix=f"{Path(__file__).stem}", verbose=verbose
        )

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
