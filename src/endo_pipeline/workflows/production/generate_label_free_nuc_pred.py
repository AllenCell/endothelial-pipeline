def generate_results(args: dict) -> None:
    from pathlib import Path

    import numpy as np
    from bioio import BioImage
    from cellpose import core, models

    from endo_pipeline.io import load_zarr_as_dask_array
    from endo_pipeline.library.process.general_image_preprocessing import save_image_output
    from endo_pipeline.manifests import get_model_location_for_run, load_model_manifest
    from endo_pipeline.settings import DIMENSION_ORDER

    """Produce label-free nuclear predictions for a given dataset, position, and timepoint."""
    dataset_name = args["dataset_name"]
    position = args["position"]
    tp = args["T"]
    create_validation = args["is_validation_image"]
    img_path = Path(args["input_path"])
    out_dir = Path(args["output_dir"]) / dataset_name / f"P{position}"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_validation = Path(args["output_dir"]) / "validation" / dataset_name / f"P{position}"
    out_dir_validation.mkdir(exist_ok=True, parents=True)

    out_path = out_dir / f"{dataset_name}_P{position}_T{tp}.ome.tiff"
    out_path_validation = (
        out_dir_validation / f"{dataset_name}_P{position}_T{tp}_cellpose_overlay.ome.tiff"
    )

    logger.info(
        f'Working on dataset {args["dataset_name"]}, T = {tp}, scene = {args["scene_index"]}...'
    )

    if (not args["overwrite"]) and out_path.exists():
        logger.info(" - output already exists, skipping...")
        return

    else:
        img_arr = load_zarr_as_dask_array(
            path=img_path,
            channels=["BF"],
            timepoints=tp,
            level=0,
        )
        voxel_size = BioImage(img_path).physical_pixel_sizes

        # Load the retrained CellPose label-free nuclear prediction model
        model_manifest = load_model_manifest("nuc_pred_labelfree")
        run_name = "finetuned_20250419"
        model_location = get_model_location_for_run(model_manifest, run_name)

        gpu = core.use_gpu()
        global device_used_printed_global
        if not device_used_printed_global:
            logger.info(f" - using device: {'GPU' if gpu else 'CPU'}")
            device_used_printed_global = True

        model_path = model_location.path.as_posix()  # type: ignore[union-attr]
        model_bf_stdproject = models.CellposeModel(gpu=gpu, pretrained_model=model_path)

        # Calculate the brightfield standard deviation projection
        bf_std_dask_arr = img_arr.std(axis=DIMENSION_ORDER.index("Z"), keepdims=True)
        bf_std_arr = bf_std_dask_arr.squeeze().compute()

        # Predict nuclei from brightfield images
        logger.info(" - predicting nuclei from brightfield standard deviation projections...")

        masks_bf_std = model_bf_stdproject.eval(
            bf_std_arr,
            channels=[0, 0],
            min_size=500,
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        )

        # Save a nuclei prediction image
        images_out = [masks_bf_std[0].squeeze()]
        logger.info(" - saving image...")

        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["CellPose_prediction"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": voxel_size,
            "dim_order": "YX",
        }
        save_image_output(out_path, images_out, images_out_metadata)

        if create_validation:
            # Find a brightfield plane with enough contrast to see nuclei by eye
            plane_stdevs = [arr.std().compute() for arr in img_arr.squeeze()]
            # don't allow the possible good contrast plane to be less than 0
            # (0 is the bottom of the Z-stack)
            possible_good_contrast_brightfield_plane = max(0, np.argmin(plane_stdevs) - 6)
            bf_good_contrast_arr = (
                img_arr.squeeze()[[possible_good_contrast_brightfield_plane]].squeeze().compute()
            )

            # Construct and save a multichannel image
            images_out = [bf_good_contrast_arr, bf_std_arr, masks_bf_std[0].squeeze()]
            logger.info(" - saving validation image...")

            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": ["BF_Center", "BF_STD", "CellPose_prediction"],
                "channel_colors": [(255, 255, 255), (255, 255, 255), (0, 255, 255)],
                "physical_pixel_sizes": voxel_size,
                "dim_order": "YX",
            }
            save_image_output(out_path_validation, images_out, images_out_metadata)


def main(
    datasets: Datasets,
    n_proc: int = 1,
    save_output: bool = True,
    overwrite: bool = True,
    is_test: bool = False,
    verbose: bool = False,
) -> None:
    """
    Run the label-free nuclear prediction workflow on a dataset, list of datasets, or collection.

    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20241217_20X 20241120_20X
    """

    from multiprocessing import Pool

    from tqdm import tqdm

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue

    out_dir = get_output_path(__file__)

    logger.info(f"datasets to analyze: {datasets}")

    # Get a list of timepoints and associated arguments to process from the list
    # of datasets to analyze and create validation images every 48 timepoints (ie. 4hrs)
    analysis_queue = build_analysis_queue(
        datasets,
        out_dir=out_dir,
        save_output=save_output,
        overwrite=overwrite,
        is_test=is_test,
        image_validation_frequency=48,
    )

    # Predict nuclei from brightfield images using the retrained CellPose model
    if n_proc > 1:
        if __name__ == "__main__":
            logger.info("Starting multiprocessing...")
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
        logger.info("Starting single-core processing...")
        for dataset_name_and_args in tqdm(analysis_queue, desc="Predicting nuclei (1P)"):
            generate_results(dataset_name_and_args)

    logger.info("...done analysis.")
    print("\N{MICROSCOPE}")


if __name__ == "__main__":
    import logging

    from endo_pipeline.cli import Datasets
    from endo_pipeline.workflows.production.cdh5_classic_seg_tracking import ipython_cli_flexecute

    logger = logging.getLogger(__name__)

    device_used_printed_global = False
    ipython_cli_flexecute(main)
