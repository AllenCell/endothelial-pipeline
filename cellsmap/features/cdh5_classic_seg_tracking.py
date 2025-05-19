from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from bioio import BioImage
from tqdm import tqdm

from cellsmap.features.lib_tracking import run_tracking
from cellsmap.util import cdh5_preprocessing as preproc
from cellsmap.util import dataset_io
from cellsmap.util.dataset_io import (
    extract_T,
    get_cdh5_classic_segmentation_path,
    ipython_cli_flexecute,
    load_config,
)
from cellsmap.util.general_image_preprocessing import build_analysis_queue
from cellsmap.util.get_sldy_metadata import get_voxel_size
from cellsmap.util.set_output import get_output_path


def run_workflow(queue):
    (dataset_name, position), queue_df = queue
    T_to_eval = queue_df["T"].tolist()
    scene_index, position = (
        int(queue_df["scene_index"].unique()[0]),
        queue_df["position"].unique()[0],
    )
    image_validation_frequency = queue_df["image_validation_frequency"].unique()[0]
    verbose = queue_df["verbose"].unique()[0]
    out_dir = queue_df["output_dir"].unique()[0] / f"{dataset_name}/P{position}"
    out_filename_prefix = f"{dataset_name}_P{position}"
    use_original_data = queue_df["use_original_data"].unique()[0]
    binning_level = queue_df["image_bin_level"].unique()[0]

    # get the segmentation images
    seg_dir = Path(get_cdh5_classic_segmentation_path(dataset_name, position=position))
    seg_filepaths = sorted(
        seg_dir.glob("*.ome.tif*"), key=lambda fp: extract_T(fp.name)
    )
    segmentation_channel = 0  # the segmentation images only contain a single channel

    # run the tracking workflow
    if seg_filepaths:
        # get the raw cadherin channel from either original data or the zarr version
        if use_original_data:
            raw_channel = dataset_io.get_dataset_info(dataset_name)["488_channel_index"]
            raw_filepath = Path(dataset_io.get_original_path(dataset_name))
            img = BioImage(raw_filepath)
            img.set_scene(scene_index)
        else:
            raw_channel = (
                0  # zarr files are created such that the first channel is always Cdh5
            )
            raw_filepath = Path(dataset_io.get_zarr_path(dataset_name))
            img = BioImage(raw_filepath)
            img.set_resolution_level(binning_level)

        # get voxel sizes from the raw image to pass along to saved images from tracking
        img_metadata = {
            "physical_pixel_sizes": get_voxel_size(img.metadata),
        }

        run_tracking(
            in_dir=seg_filepaths,
            out_dir=out_dir,
            out_filename_prefix=out_filename_prefix,
            tracking_metrics=[
                "region_overlap"
            ],  # this can be changed to ['centroids'] if desired
            sorting_key=preproc.extract_T,
            C=segmentation_channel,
            T=T_to_eval,
            #  extra_in_dir=raw_filepath,
            #  extra_C=raw_channel,
            #  extra_scene=scene_index,
            #  extra_T=T_to_eval,
            Z_projection=np.max,
            track_tolerance=3,
            img_metadata=img_metadata,
            image_validation_frequency=image_validation_frequency,
            verbose=verbose,
        )

    else:
        print(
            f"No segmentation images found for {dataset_name}. Skipping tracking analysis. If this is unexpected check that the IS_TEST argument is set to False."
        )
        return


def main(n_proc=1, dataset_name=None, save_output=True, is_test=False, verbose=False):

    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))

    if dataset_name == None:
        dataset_name_list = [
            config_data["name"]
            for config_data in load_config(config_type="data")
            if (
                config_data["microscope"] == "3i"
                and config_data["live_or_fixed_sample"] == "live"
            )
            and "AICS-126" in config_data["cell_lines"]
            and config_data["duration"] > 1
        ]
    else:
        dataset_name_list = [dataset_name]

    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
        use_original_data=True,
    )

    analysis_queue_df = pd.DataFrame(analysis_queue)
    analysis_queue_per_position = list(
        analysis_queue_df.groupby(["dataset_name", "position"])
    )

    if n_proc > 1:
        if __name__ == "__main__":
            print("Starting multiprocessing...")
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(
                            run_workflow, analysis_queue_per_position, chunksize=1
                        ),
                        total=len(analysis_queue_per_position),
                        desc="Positions completed",
                    )
                )
                pool.close()
                pool.join()
            print("Done multiprocessing.")
    else:
        for queue in analysis_queue_per_position:
            print("Running workflow with single process...")
            run_workflow(queue)
            print("Done single-processing.")

    print("\N{MICROSCOPE} Done analysis.")

    for dataset_name in dataset_name_list:
        tracking_table_paths = (out_dir / dataset_name).glob("**/*.tsv")
        if tracking_table_paths:
            pd.concat(
                [pd.read_csv(fp, sep="\t") for fp in tracking_table_paths]
            ).to_csv(
                out_dir / dataset_name / f"{dataset_name}_tracking.tsv",
                index=False,
                sep="\t",
            )


if __name__ == "__main__":
    ipython_cli_flexecute(main, verbose=True)
