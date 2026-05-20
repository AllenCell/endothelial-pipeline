import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from cellpose import core, models
from matplotlib.figure import Figure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.process.general_image_preprocessing import (
    ImageProcessingArgs,
    process_task_queue,
    save_image_output,
)
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings import DIMENSION_ORDER

logger = logging.getLogger(__name__)


def get_training_data_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Return the output directories for the training data."""

    return {
        "labels": output_dir / "inputs" / "cellpose_base_nuclei_model_nuclei_segmentations",
        "nuclei": output_dir / "inputs" / "cellpose_base_nuclei_model_nuclei_max",
        "images": output_dir / "inputs" / "cellpose_base_nuclei_model_brightfield_std",
    }


def get_image_data_from_zarr(dataset_name: str, position: int, timepoint: int) -> tuple:
    """
    Load the NucViolet and Brightfield channels from the zarr file for the given
    dataset, position, and timepoint as well as the size of the voxel along the
    Z, Y, and X dimensions.
    """

    dataset_config = load_dataset_config(dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    voxel_size = load_image(zarr_loc, read=False).physical_pixel_sizes

    nuc_chan = dataset_config.zarr_channel_indices.channel_405
    bf_chan = dataset_config.zarr_channel_indices.brightfield

    img = load_image(zarr_loc, timepoints=timepoint, level=0)
    img_nuc_dask_arr = np.take(img, indices=[nuc_chan], axis=DIMENSION_ORDER.index("C"))
    img_dask_arr_bf_std = np.take(img, indices=[bf_chan], axis=DIMENSION_ORDER.index("C"))

    img_dask_arr_nuc = img_nuc_dask_arr.max(axis=DIMENSION_ORDER.index("Z"), keepdims=True)
    img_dask_arr_bf_std = img_dask_arr_bf_std.std(axis=DIMENSION_ORDER.index("Z"), keepdims=True)

    return (img_dask_arr_nuc, img_dask_arr_bf_std), voxel_size


def save_overlay(
    labels: np.ndarray,
    bg_img: np.ndarray,
    out_name: Path | str,
    outlines: bool = True,
    face: bool = True,
) -> None:
    """
    Save an overlay of the labels on top of the background image.
    If outlines is True, the outlines of the labels will be highlighted.
    If face is True, the labels will be filled in with color.
    """
    from skimage.segmentation import find_boundaries

    seg_outlines = find_boundaries(labels)
    if outlines and face:
        labels[seg_outlines] = 0
    elif outlines and not face:
        labels = seg_outlines
    else:
        pass
    overlay = label2rgb(label=labels, image=bg_img, bg_label=0)

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_name, bbox_inches="tight", pad_inches=0, dpi=180)
    plt.close(fig)


def save_labelfree_nuclei_example_image(
    original_bf_img_array: np.ndarray,
    nuclei_prediction_img_arr: np.ndarray,
    model_name: str,
    out_dir=Path,
) -> plt.Figure:
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    image_rescaled = rescale_intensity(
        np.clip(
            original_bf_img_array,
            a_min=np.percentile(original_bf_img_array, 1),
            a_max=np.percentile(original_bf_img_array, 99),
        ),
        out_range=(0, 1),
    )
    ax0.imshow(image_rescaled, cmap="gray")
    ax0.set_title("BF STD")
    ax1.imshow(label2rgb(nuclei_prediction_img_arr))
    ax1.set_title("Nuclei Predictions")
    overlay = label2rgb(label=nuclei_prediction_img_arr, image=image_rescaled, bg_label=0)
    ax2.imshow(overlay)
    [ax.set_axis_off() for ax in (ax0, ax1, ax2)]
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model_name}_test_image.png",
        bbox_inches="tight",
        dpi=180,
    )

    return fig


def generate_training_data(args: ImageProcessingArgs) -> None:
    """
    Generate training data for retraining a Cellpose model to predict nuclei
    from brightfield standard deviation projections.

    The training data consists of:
    - Brightfield standard deviation projections as the images
    - Nuclei segmentations from the Cellpose base nuclei model as the labels
    """

    # unpack the analysis arguments
    dataset_name = args.dataset_name
    position = args.position
    tp = args.timepoint
    save_training_data = args.save_output
    save_validation_images = args.is_validation_image

    # build output paths
    out_dirs = get_training_data_output_dirs(args.output_dir)
    out_dirs["validation"] = args.output_dir / "validation_overlays" / dataset_name

    use_gpu = core.use_gpu()

    # initialize the base Cellpose nuclei model
    logger.info("loading CellPose model...")
    nuc_model = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    # load and process brightfield and NucViolet channels of image data
    logger.info("loading image data...")
    (nuc_max, bf_std), voxel_size = get_image_data_from_zarr(dataset_name, position, tp)

    logger.info("processing image data...")
    nuc_max = nuc_max.compute().squeeze()
    bf_std = bf_std.compute().squeeze()
    normd_nuc = rescale_intensity(nuc_max, out_range=np.uint16)
    normd_nuc_clipped = rescale_intensity(
        np.clip(normd_nuc, 0, np.percentile(normd_nuc, 99)), out_range=(0, 1)
    )

    # create nuclei segmentations from the NucViolet channel
    logger.info("generating segmentations with CellPose model...")
    seg, _, _ = nuc_model.eval(
        normd_nuc_clipped,
        channels=[0, 0],
        min_size=500,
        flow_threshold=0.6,
        cellprob_threshold=-3.0,
    )

    logger.info("saving images...")
    if save_validation_images:
        # create an overlay to quickly check the accuracy of the Cellpose predictions
        # from NucViolet
        out_dirs["validation"].mkdir(exist_ok=True, parents=True)
        out_name_val = out_dirs["validation"] / f"{dataset_name}_P{position}_classic_seg.png"
        save_overlay(seg, normd_nuc_clipped, out_name_val, outlines=True, face=True)

    if save_training_data:
        out_key = f"{dataset_name}_P{position}_T{tp}"

        # save the labels used as ground truths for training
        # the label-free nuclei model
        out_dirs["labels"].mkdir(exist_ok=True, parents=True)
        out_name_label = out_dirs["labels"] / f"{out_key}_nuclei_seg.ome.tiff"
        images_out = [seg]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_nuclei_prediction"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": voxel_size,
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_label,
            images=images_out,
            images_metadata=images_out_metadata,
        )

        out_dirs["nuclei"].mkdir(exist_ok=True, parents=True)
        out_name_nuclei = out_dirs["nuclei"] / f"{out_key}_nuclei_raw.ome.tiff"
        images_out = [nuc_max]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_nuclei_max_projects"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": voxel_size,
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_nuclei,
            images=images_out,
            images_metadata=images_out_metadata,
        )

        out_dirs["images"].mkdir(exist_ok=True, parents=True)
        out_name_images = out_dirs["images"] / f"{out_key}_bf_std.ome.tiff"
        images_out = [bf_std]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_brightfield_standard_deviation_projection"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": voxel_size,
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_images,
            images=images_out,
            images_metadata=images_out_metadata,
        )


def get_training_data_paths(
    out_dir: Path,
    analysis_queue: list[ImageProcessingArgs],
    num_processes: int = 1,
) -> tuple:
    """
    Get paths to the training data for retraining a Cellpose model to predict
    nuclei from brightfield standard deviation projections.

    The method will iterate through the provided analysis queue to determine if
    the required images and labels exists. If not, it will generate the training
    data:

    - Brightfield standard deviation projections as the images
    - Nuclei segmentations from the Cellpose base nuclei model as the labels
    """

    out_dirs = get_training_data_output_dirs(out_dir)

    # check if training data exists and regenerate if not
    analysis_queue_missing = []
    for arg in analysis_queue:
        out_key = f"{arg.dataset_name}_P{arg.position}_T{arg.timepoint}"

        out_name_label = out_dirs["labels"] / f"{out_key}_nuclei_seg.ome.tiff"
        out_name_images = out_dirs["images"] / f"{out_key}_bf_std.ome.tiff"

        if not out_name_label.exists() or not out_name_images.exists():
            analysis_queue_missing.append(arg)

    process_task_queue(
        generate_training_data,
        analysis_queue_missing,
        num_processes=num_processes,
        description="Creating training data images",
        chunksize=1,
    )

    images_paths = list(out_dirs["images"].glob("**/*.ome.tiff"))
    labels_paths = list(out_dirs["labels"].glob("**/*.ome.tiff"))

    if len(labels_paths) != len(images_paths):
        raise ValueError(
            f"Number of images ({len(images_paths)}) must equal "
            f"number of labels ({len(labels_paths)})"
        )

    return (images_paths, labels_paths)


def load_train_and_test_images(
    out_dir: Path, analysis_queue: list[ImageProcessingArgs], num_processes: int
) -> tuple[list, list, list, list]:
    """
    Load training and testing images for retraining Cellpose base nuclei model
    on nuclei labeled with DAPI.
    """

    # get paths to images and labels
    images_paths, labels_paths = get_training_data_paths(
        out_dir=out_dir,
        analysis_queue=analysis_queue,
        num_processes=num_processes,
    )

    # split the images and labels into training and testing sets
    testing_indices = list(range(0, len(images_paths), 5))
    training_indices = [i for i in range(len(images_paths)) if i not in testing_indices]

    # load the brightfield standard deviation projections as
    # the images and the  nuclei segmentations as the labels
    # from the testing and training data
    logger.info("Loading training and testing data...")
    dim_order = "CYX"
    images_training = []
    images_testing = []
    labels_training = []
    labels_testing = []
    for i in training_indices:
        images_training.append(BioImage(images_paths[i]).get_image_data(dim_order))
        labels_training.append(BioImage(labels_paths[i]).get_image_data(dim_order))
    for i in testing_indices:
        images_testing.append(BioImage(images_paths[i]).get_image_data(dim_order))
        labels_testing.append(BioImage(labels_paths[i]).get_image_data(dim_order))

    return images_training, labels_training, images_testing, labels_testing


def save_training_test_loss_plot(
    train_losses: np.ndarray, test_losses: np.ndarray, model_name: str, out_dir: Path
) -> Figure:

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(
        np.where(train_losses)[0],
        train_losses[np.where(train_losses)],
        label="train_loss",
    )
    ax.plot(
        np.where(test_losses)[0],
        test_losses[np.where(test_losses)],
        label="test_loss",
    )
    ax.set_title(f"{model_name} training and test losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model_name}_training_test_losses.png",
        bbox_inches="tight",
        dpi=180,
    )

    return fig
