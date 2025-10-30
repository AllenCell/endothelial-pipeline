import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from cellpose import models
from numpy.typing import ArrayLike
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_image
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings import DIMENSION_ORDER

logger = logging.getLogger(__name__)


def get_scenes_to_use(dataset_name: str | None = None) -> dict:
    """
    Return the scenes to use for training a Cellpose model to predict nuclei from brightfield.
    If no dataset name is provided, it returns scenes for all datasets that were used for training.
    It is used to filter the analysis queue to only include the
    scenes that are needed for the analysis.
    This is needed because a couple of the older datasets have
    scenes at different magnifications or scenes that are corrupted.
    You can use dataset_name to return a single set of scenes.
    """
    scenes_to_use = {
        "20240328_T02_001": [
            "20240328_T02_001-1711659785-  8",
            "20240328_T02_001-1711659785- 24",
            "20240328_T02_001-1711659785- 39",
            "20240328_T02_001-1711659785-990",
        ],
        "20240328_T01_001": [
            "20240328_T01_001-1711663662-276",
            "20240328_T01_001-1711663662-293",
            "20240328_T01_001-1711663662-307",
            "20240328_T01_001-1711663662-322",
            "20240328_T01_001-1711663662-337",
        ],
        "20250415_SlideA_20X": ["20250415_GE00007488_slideA_20X - Position 1 [50]-1745428788-773"],
        "20250415_SlideE_20X": ["20250416_GE00007101_slideE_20X - Position 1 [50]-1745428816-305"],
        "20250415_SlideH_20X": ["20250415_GE00006885_slideH_20X - Position 1 [50]-1745428734-340"],
    }
    if dataset_name is None:
        return scenes_to_use
    if dataset_name in scenes_to_use:
        return {dataset_name: scenes_to_use[dataset_name]}
    else:
        return {}


def get_training_data_output_dirs(
    kind: list[Literal["images", "labels"]] | None = None,
) -> list:
    """Return the output directories for the training data."""

    out_dir = get_output_path(__file__)
    out_dir_labels = out_dir / "training_data/cellpose_base_nuclei_model_nuclei_segmentations/"
    out_dir_images = out_dir / "training_data/cellpose_base_nuclei_model_brightfield_std/"
    out_dirs = {"images": out_dir_images, "labels": out_dir_labels}
    if kind is None:
        return list(out_dirs.values())
    else:
        return [out_dirs[training_data_kind] for training_data_kind in kind]


def get_image_data_from_zarr(dataset_name: str, position: int, timepoint: int) -> tuple:
    """
    Load the NucViolet and Brightfield channels from the zarr file for the given dataset, position,
    and timepoint as well as the size of the voxel along the Z, Y, and X dimensions.
    """

    dataset_config = load_dataset_config(dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    voxel_size = load_image(zarr_loc, read=False).physical_pixel_sizes

    nuc_chan: int = dataset_config.zarr_channel_indices.channel_405  # type: ignore[assignment]
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


def generate_training_data(analysis_args: dict) -> None:
    """Make training data for retraining a Cellpose model to predict nuclei from brightfield
    standard deviation projections.

    The training data consists of:
    - Brightfield standard deviation projections as the images
    - Nuclei segmentations from the Cellpose base nuclei model as the labels
    """

    from endo_pipeline.library.process.general_image_preprocessing import save_image_output

    # unpack the analysis arguments
    use_gpu = analysis_args["gpu"]
    dataset_name = analysis_args["dataset_name"]
    position = analysis_args["position"]
    tp = analysis_args["T"]
    out_dir_val = analysis_args["output_dir"] / f"training_data/validation_overlays/{dataset_name}/"
    out_dir_nuclei = (
        analysis_args["output_dir"] / "training_data/cellpose_base_nuclei_model_nuclei_max/"
    )
    out_dir_images, out_dir_labels = get_training_data_output_dirs(kind=["images", "labels"])
    save_training_data = analysis_args["save_output"]
    save_validation_images = analysis_args["is_validation_image"]

    # initialize the base Cellpose nuclei model
    logger.info("loading CellPose model...")
    nuc_model = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    # load and process brightfield and NucViolet channels of image data
    logger.info("loading image data...")
    img_dask_arrs, voxel_size = get_image_data_from_zarr(dataset_name, position, tp)

    logger.info("processing image data...")
    nuc_max, bf_std = img_dask_arrs
    nuc_max = nuc_max.compute().squeeze()
    bf_std = bf_std.compute().squeeze()
    normd_nuc = rescale_intensity(nuc_max, out_range=np.uint16)
    normd_nuc_clipped = rescale_intensity(
        np.clip(normd_nuc, 0, np.percentile(normd_nuc, 99)), out_range=(0, 1)
    )

    # create nuclei segmentations from the NucViolet channel
    logger.info("generating segmentations with CellPose model...")
    seg, flows, styles = nuc_model.eval(
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
        out_dir_val.mkdir(exist_ok=True, parents=True)
        out_name_val = out_dir_val / f"{dataset_name}_P{position}_classic_seg.png"
        save_overlay(seg, normd_nuc_clipped, out_name_val, outlines=True, face=True)

    if save_training_data:
        # save the labels used as ground truths for training
        # the label-free nuclei model
        out_dir_labels.mkdir(exist_ok=True, parents=True)
        out_name_label = out_dir_labels / f"{dataset_name}_P{position}_T{tp}_nuclei_seg.ome.tiff"
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

        out_dir_nuclei.mkdir(exist_ok=True, parents=True)
        out_name_dapi = out_dir_nuclei / f"{dataset_name}_P{position}_T{tp}_nuclei_raw.ome.tiff"
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
            out_path=out_name_dapi,
            images=images_out,
            images_metadata=images_out_metadata,
        )

        out_dir_images.mkdir(exist_ok=True, parents=True)
        out_name_images = out_dir_images / f"{dataset_name}_P{position}_T{tp}_bf_std.ome.tiff"
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
    return


def get_training_data_paths(
    analysis_queue: list,
    create_training_data: bool = False,
    n_proc: int = 1,
    gpu: bool = False,
) -> tuple:
    """Return the paths to the training data for retraining a Cellpose model to predict nuclei from
    brightfield standard deviation projections.

    If create_training_data is True, it will create the training data, and if it is False, it will
    return the paths to the training data that already exists.
    The training data consists of:
    - Brightfield standard deviation projections as the images
    - Nuclei segmentations from the Cellpose base nuclei model as the labels
    """
    from multiprocessing import Pool

    from tqdm import tqdm

    # add the whether or not to use the GPU to the analysis queue
    for arg in analysis_queue:
        arg.update({"gpu": gpu})

    if create_training_data:
        if __name__ == "__main__":
            if n_proc > 1 and not gpu:
                with Pool(processes=n_proc) as pool:
                    print("Starting multiprocessing...")
                    list(
                        tqdm(
                            pool.imap(generate_training_data, analysis_queue),
                            total=len(analysis_queue),
                            desc="Training data images created",
                        )
                    )
                    pool.close()
                    pool.join()
                    print("Done multiprocessing.")
            else:
                print(f"Starting {'gpu' if gpu else 'single core'} processing...")
                for analysis_args in tqdm(
                    analysis_queue,
                    total=len(analysis_queue),
                    desc="Training data images created",
                ):
                    generate_training_data(analysis_args)
                print("Done single processing.")
    else:
        pass

    # Open the training data images and labels
    # that were created in the previous step
    (images_dir,) = get_training_data_output_dirs(kind=["images"])
    (labels_dir,) = get_training_data_output_dirs(kind=["labels"])
    images_paths = list(images_dir.glob("**/*.ome.tiff"))
    labels_paths = list(labels_dir.glob("**/*.ome.tiff"))

    assert len(images_paths) == len(
        labels_paths
    ), f"Number of images ({len(images_paths)}) must equal number of labels ({len(labels_paths)})"

    return (images_paths, labels_paths)


def load_train_and_test_images(
    analysis_queue, create_training_data: bool, n_proc: int, gpu: bool
) -> list[ArrayLike]:
    # Generate ground truths from nuclei labeled with DAPI
    # using the Cellpose base nuclei model
    images_paths, labels_paths = get_training_data_paths(
        analysis_queue,
        create_training_data=create_training_data,
        n_proc=n_proc,
        gpu=gpu,
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
) -> plt.Figure:

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
