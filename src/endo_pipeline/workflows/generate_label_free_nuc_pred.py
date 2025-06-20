from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
from bioio import BioImage
from cellpose import core, models
from src.endo_pipeline.library.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
    get_dim_map,
    save_image_output,
)
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_dataset_info,
    load_config,
)
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.workflows.cdh5_classic_seg_tracking import ipython_cli_flexecute


# Predict nuclei from brightfield images using the retrained CellPose model
def generate_results(args: dict) -> None:

    verbose = args["verbose"]
    dataset_name = args["dataset_name"]
    create_validation = args["validation_image"]
    img_path = args["input_path"]
    out_dir = Path(args["output_dir"]) / dataset_name / f'P{args["position"]}'
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_validation = (
        Path(args["output_dir"]) / "validation" / dataset_name / f'P{args["position"]}'
    )
    out_dir_validation.mkdir(exist_ok=True, parents=True)

    out_path = out_dir / f'{dataset_name}_P{args["position"]}_T{args["T"]}.ome.tiff'
    out_path_validation = (
        out_dir_validation
        / f'{dataset_name}_P{args["position"]}_T{args["T"]}_cellpose_overlay.ome.tiff'
    )

    (
        print(
            f'Working on dataset {args["dataset_name"]}, T = {args["T"]}, scene = {args["scene_index"]}...'
        )
        if verbose
        else None
    )

    if (args["overwrite"] == False) and out_path.exists():
        print(" - output already exists, skipping...") if verbose else None
        return

    else:
        dim_order = get_default_dim_order()
        dim_map = get_dim_map(dim_order)

        img = BioImage(img_path)
        if args["use_original_data"]:
            img.set_scene(args["scene_index"])

        brightfield_index = get_dataset_info(dataset_name)["brightfield_channel_index"]
        img_arr = img.get_image_dask_data(dim_order, T=args["T"], C=brightfield_index)

        # Load the retrained CellPose label-free nuclear prediction model
        model_config = load_config(config_type="model")
        nuclei_model = model_config["nuc_pred_labelfree_retrained_20250419-18_13"]

        gpu = core.use_gpu()

        model_path = Path(nuclei_model["model_path"])
        model_bf_stdproject = models.CellposeModel(
            gpu=gpu, pretrained_model=str(model_path)
        )

        # Calculate the brightfield standard deviation and the brightfield image with the best contrast
        bf_std_dask_arr = img_arr.std(axis=dim_map["Z"], keepdims=True)
        bf_std_arr = bf_std_dask_arr.squeeze().compute()

        # Predict nuclei from brightfield images
        (
            print(
                " - predicting nuclei from brightfield standard deviation projections..."
            )
            if verbose
            else None
        )
        masks_bf_std = model_bf_stdproject.eval(
            bf_std_arr,
            channels=[0, 0],
            min_size=500,
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        )

        # Save a nuclei prediction image
        images_out = [masks_bf_std[0].squeeze()]
        print(" - saving image...") if verbose else None
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["CellPose_prediction"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "dim_order": "YX",
        }
        save_image_output(out_path, images_out, images_out_metadata)

        if create_validation:
            # Find a brightfield plane with enough contrast to see
            # nuclei by eye
            plane_stdevs = [arr.std().compute() for arr in img_arr.squeeze()]
            # don't allow the possible good contrast plane to be less than 0 (i.e. the bottom of the Z-stack)
            possible_good_contrast_brightfield_plane = max(
                0, np.argmin([plane for plane in plane_stdevs]) - 6
            )
            bf_good_contrast_arr = (
                img_arr.squeeze()[[possible_good_contrast_brightfield_plane]]
                .squeeze()
                .compute()
            )

            # Construct and save a multichannel image
            images_out = [bf_good_contrast_arr, bf_std_arr, masks_bf_std[0].squeeze()]
            print(" - saving validation image...") if verbose else None
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": ["BF_Center", "BF_STD", "CellPose_prediction"],
                "channel_colors": [(255, 255, 255), (255, 255, 255), (0, 255, 255)],
                "physical_pixel_sizes": img.physical_pixel_sizes,
                "dim_order": "YX",
            }
            save_image_output(out_path_validation, images_out, images_out_metadata)


def main(
    dataset_name: str | List | None = None,
    n_proc: int = 1,
    save_output: bool = True,
    overwrite: bool = True,
    is_test: bool = False,
    use_original_data: bool = False,
    verbose: bool = False,
) -> None:
    """
    To enter a list of datasets to analyze, use the following format:
    '\"20241016_20X\",\"20241120_20X\"'
    """
    # Set the output directory
    out_dir = Path(get_output_path(Path(__file__).stem))

    # Build a list of datasets to analyze
    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    # Get a list of timepoints and associated arguments to process from the list of datasets to analyze
    # evaluate every 48 timepoints (ie. 4hrs)
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        out_dir=out_dir,
        save_output=save_output,
        overwrite=overwrite,
        is_test=is_test,
        image_validation_frequency=48,
        use_original_data=use_original_data,
        verbose=verbose,
    )

    gpu = core.use_gpu()
    print(f" - using device: {'GPU' if gpu else 'CPU'}") if verbose else None

    if n_proc > 1:
        if __name__ == "__main__":
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(generate_results, analysis_queue, chunksize=5),
                        total=len(analysis_queue),
                        desc="Predicting nuclei (MP)",
                    )
                )
                pool.close()
                pool.join()
    else:
        for dataset_name_and_args in tqdm(
            analysis_queue, desc="Predicting nuclei (1P)"
        ):
            generate_results(dataset_name_and_args)

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
