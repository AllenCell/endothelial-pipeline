import logging

import numpy as np
from cellpose import core

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image, load_model
from endo_pipeline.library.process.general_image_preprocessing import (
    ImageProcessingArgs,
    save_image_output,
)
from endo_pipeline.manifests import (
    get_model_location_for_run,
    get_zarr_location_for_position,
    load_model_manifest,
)
from endo_pipeline.settings import DIMENSION_ORDER

device_used_printed_global = False

logger = logging.getLogger(__name__)


def generate_labelfree_nuclei_predictions(args: ImageProcessingArgs) -> None:
    """Produce label-free nuclear predictions for a given dataset, position, and timepoint."""
    dataset_name = args.dataset_name
    position = args.position
    tp = args.timepoint
    create_validation = args.is_validation_image
    out_dir = args.output_dir / dataset_name / f"P{position}"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_validation = args.output_dir / "validation" / dataset_name / f"P{position}"
    out_dir_validation.mkdir(exist_ok=True, parents=True)

    out_path = out_dir / f"{dataset_name}_P{position}_T{tp}.ome.tiff"
    out_path_validation = (
        out_dir_validation / f"{dataset_name}_P{position}_T{tp}_cellpose_overlay.ome.tiff"
    )

    logger.info(f"Working on dataset {args.dataset_name}, T = {tp}, position = {args.position}...")

    if (not args.overwrite) and out_path.exists():
        logger.info(" - output already exists, skipping...")
        return

    else:
        dataset_config = load_dataset_config(dataset_name)
        location = get_zarr_location_for_position(dataset_config, position)
        reader = load_image(location, read=False)
        img_arr = load_image(location, channels=["BF"], timepoints=tp, level=0)
        voxel_size = reader.physical_pixel_sizes

        # Load the retrained CellPose label-free nuclear prediction model
        model_manifest = load_model_manifest("nuc_pred_labelfree")
        run_name = "finetuned_20250419"
        model_location = get_model_location_for_run(model_manifest, run_name)

        gpu = core.use_gpu()
        global device_used_printed_global
        if not device_used_printed_global:
            logger.info(f" - using device: {'GPU' if gpu else 'CPU'}")
            device_used_printed_global = True

        model_bf_stdproject = load_model(model_location)

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
            possible_good_contrast_brightfield_plane = max(0, int(np.argmin(plane_stdevs) - 6))
            bf_good_contrast_arr = np.take(
                img_arr,
                indices=possible_good_contrast_brightfield_plane,
                axis=DIMENSION_ORDER.index("Z"),
            ).squeeze()
            bf_good_contrast_arr = bf_good_contrast_arr.compute()

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
