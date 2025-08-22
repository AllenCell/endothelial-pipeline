from collections.abc import Generator
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from cellpose import core, models, train
from cellpose.io import logger_setup
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.configs.dataset_io import get_original_path, ipython_cli_flexecute, load_config
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process import get_sldy_metadata as sldmd
from endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
    get_dim_map,
    save_image_output,
)


def get_scenes_to_use(dataset_name: str | None = None) -> dict:
    """
    This function returns the scenes to use for a given dataset.
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
    if dataset_name == None:
        return scenes_to_use
    if dataset_name in scenes_to_use:
        return {dataset_name: scenes_to_use[dataset_name]}
    else:
        return {}


def get_training_data_output_dirs(
    kind: list[Literal["images", "labels"]] | None = None,
) -> list:
    out_dir = get_output_path(__file__)
    out_dir_labels = out_dir / "training_data/cellpose_base_nuclei_model_nuclei_segmentations/"
    out_dir_images = out_dir / "training_data/cellpose_base_nuclei_model_brightfield_std/"
    out_dirs = {"images": out_dir_images, "labels": out_dir_labels}
    if kind == None:
        return list(out_dirs.values())
    else:
        return [out_dirs[training_data_kind] for training_data_kind in kind]


def get_image_data_from_original(
    dataset_name: str, scene: str | int, T: int, verbose: bool = False
) -> tuple:

    dim_order = get_default_dim_order()
    dim_map = get_dim_map(dim_order)

    img_path = Path(get_original_path(dataset_name))
    img = BioImage(img_path)
    img.set_scene(scene)
    img_metadata = img.metadata
    print(dataset_name, img.current_scene) if verbose else None

    channel_names = sldmd.get_channel_name(img.metadata)
    channel_names = [chan.split("/")[0] for chan in channel_names]
    dataset_config = load_dataset_config(dataset_name)
    nuc_chan = dataset_config.original_channel_indices.channel_405
    bf_chan = dataset_config.original_channel_indices.brightfield

    img_dask_arr_nuc = img.get_image_dask_data(dim_order, C=[nuc_chan], T=T).max(
        axis=dim_map["Z"], keepdims=True
    )
    img_dask_arr_bf_std = img.get_image_dask_data(dim_order, C=[bf_chan], T=T).std(
        axis=dim_map["Z"], keepdims=True
    )
    return (img_dask_arr_nuc, img_dask_arr_bf_std), img_metadata


def get_image_data_from_zarr(dataset_name: str) -> Generator:
    # NOTE THIS FUNCTION IS NOT YET IMPLEMENTED
    for zarr_name in get_zarr_path(dataset_name):
        img_dict_nuc = load_dataset(dataset_name, zarr_name=zarr_name, channels=["DAPI"])
        img_dict_bf = load_dataset(dataset_name, zarr_name=zarr_name, channels=["BF"])
        img_dask_arr_nuc = img_dict_nuc[zarr_name].max(axis=dim_map["Z"], keepdims=True)
        img_dask_arr_bf_std = img_dict_bf[zarr_name].std(axis=dim_map["Z"], keepdims=True)
        yield (zarr_name, img_dask_arr_nuc, img_dask_arr_bf_std)
    raise NotImplementedError(f"Zarrs not yet implemented yet. Skipping {dataset_name}.")


def save_overlay(
    labels: np.ndarray,
    bg_img: np.ndarray,
    out_name: Path | str,
    outlines: bool = True,
    face: bool = True,
) -> None:
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


def generate_training_data(analysis_args: dict) -> None:
    use_gpu = analysis_args["gpu"]
    use_sldy_data = analysis_args["use_sldy_data"]
    dataset_name = analysis_args["dataset_name"]
    scene_name = analysis_args["scene_name"]
    position = analysis_args["position"]
    T = analysis_args["T"]
    out_dir_val = analysis_args["output_dir"] / f"training_data/validation_overlays/{dataset_name}/"
    out_dir_nuclei = (
        analysis_args["output_dir"] / "training_data/cellpose_base_nuclei_model_nuclei_max/"
    )
    out_dir_images, out_dir_labels = get_training_data_output_dirs(kind=["images", "labels"])
    save_training_data = analysis_args["save_output"]
    save_validation_images = analysis_args["validation_image"]
    verbose = analysis_args["verbose"]

    print("loading CellPose model...") if verbose else None
    nuc_model = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    if scene_name in get_scenes_to_use()[dataset_name]:
        (print(f"Working on {dataset_name} P{position} {scene_name}...") if verbose else None)
        pass
    else:
        (
            print(f"{dataset_name} P{position} {scene_name} not in scenes_to_use. Skipping.")
            if verbose
            else None
        )
        return

    print("loading image data...") if verbose else None
    if use_sldy_data:
        img_dask_arrs, image_metadata = get_image_data_from_original(dataset_name, scene_name, T)
        voxel_size = sldmd.get_voxel_size(image_metadata)
    else:
        print(f"Zarrs not yet implemented. Skipping {dataset_name} P{position}.")
        return  # zarrs not yet implemented
        # img_dask_arrs = get_image_data_from_zarr(dataset_name)

    print("processing image data...") if verbose else None
    nuc_max, bf_std = img_dask_arrs
    nuc_max = nuc_max.compute().squeeze()
    bf_std = bf_std.compute().squeeze()
    normd_nuc = rescale_intensity(nuc_max, out_range=np.uint16)
    normd_nuc_clipped = rescale_intensity(
        np.clip(normd_nuc, 0, np.percentile(normd_nuc, 99)), out_range=(0, 1)
    )
    print("generating segmentations with CellPose model...") if verbose else None
    seg, flows, styles = nuc_model.eval(
        normd_nuc_clipped,
        channels=[0, 0],
        min_size=500,
        flow_threshold=0.6,
        cellprob_threshold=-3.0,
    )

    print("saving images...") if verbose else None
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
        out_name_label = out_dir_labels / f"{dataset_name}_P{position}_T{T}_nuclei_seg.ome.tiff"
        images_out = [seg]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_nuclei_prediction"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": PhysicalPixelSizes(**voxel_size),
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_label,
            images=images_out,
            images_metadata=images_out_metadata,
        )

        out_dir_nuclei.mkdir(exist_ok=True, parents=True)
        out_name_dapi = out_dir_nuclei / f"{dataset_name}_P{position}_T{T}_nuclei_raw.ome.tiff"
        images_out = [nuc_max]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_nuclei_max_projects"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": PhysicalPixelSizes(**voxel_size),
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_dapi,
            images=images_out,
            images_metadata=images_out_metadata,
        )

        out_dir_images.mkdir(exist_ok=True, parents=True)
        out_name_images = out_dir_images / f"{dataset_name}_P{position}_T{T}_bf_std.ome.tiff"
        images_out = [bf_std]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cellpose_brightfield_standard_deviation_projection"],
            "channel_colors": [(255, 255, 255)],
            "physical_pixel_sizes": PhysicalPixelSizes(**voxel_size),
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_name_images,
            images=images_out,
            images_metadata=images_out_metadata,
        )
    return


def get_training_data(
    analysis_queue: list,
    create_training_data: bool = False,
    n_proc: int = 1,
    gpu: bool = False,
) -> tuple:

    # add the whether or not to use the GPU to the analysis queue
    for arg in analysis_queue:
        arg.update({"gpu": gpu})

    if create_training_data:
        if __name__ == "__main__":
            if n_proc > 1 and gpu == False:
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
    images_paths = [filepath for filepath in images_dir.glob("**/*.ome.tiff")]
    labels_paths = [filepath for filepath in labels_dir.glob("**/*.ome.tiff")]

    assert len(images_paths) == len(
        labels_paths
    ), f"Number of images ({len(images_paths)}) must equal number of labels ({len(labels_paths)})"

    return (images_paths, labels_paths)


def main(
    n_proc: int = 1,
    create_training_data: bool = False,
    retrain_Gouthams_model: bool = False,
    train_from_base_cellpose_nuclei_model: bool = True,
    use_sldy_data: bool = True,
    verbose: bool = False,
) -> None:

    datasets_to_use = list(get_scenes_to_use().keys())
    out_dir = get_output_path(__file__)

    analysis_queue = build_analysis_queue(
        datasets_to_use,
        use_sldy_data=use_sldy_data,
        save_output=True,
        image_validation_frequency=1,
        overwrite=True,
        out_dir=out_dir,
        verbose=verbose,
    )

    # return whether or not to use a gpu with CellPose
    gpu = core.use_gpu()

    # Generate ground truths from nuclei labeled with DAPI
    # using the Cellpose base nuclei model
    images_paths, labels_paths = get_training_data(
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
    print("Loading training and testing data...") if verbose else None
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

    print("Beginning training...") if verbose else None
    sgd = True
    learning_rate = 0.1
    weight_decay = 1e-4
    n_epochs = 300  # 100#300

    # create a timestamp for when this workflow was run
    timestamp = datetime.today().strftime("%Y%m%d-%H_%M")

    # get the nuclei model path from the config file
    model_config = load_config(config_type="model")
    nuclei_models = model_config.get("nuc_pred_labelfree")
    if nuclei_models is None:
        raise ValueError("No label-free nuclei model found in the configuration file.")
    else:
        assert len(list(nuclei_models)) == 1, f"Expected 1 model path, found {len(nuclei_models)}"
        model_path = nuclei_models.get("model_path")
        if model_path is not None:
            model_path = Path(model_path)
        else:
            raise ValueError("No model path found in the configuration file.")

    # create a directory to save the models
    # and their losses and a test image
    model_dir = model_path.parent / timestamp
    model_dir.mkdir(exist_ok=True, parents=True)

    # initiate the cellpose logger so that we
    # can extract the training and test losses
    logger_setup(cp_path=model_dir, logfile_name=f"{timestamp}_run.log")

    # will populate this dictionary as we go
    run_record: dict[str, Any] = {}

    if retrain_Gouthams_model:
        # retrain Goutham's Cellpose model
        model_dir_Goutham_retrain = model_dir / "Goutham_model_finetuning"
        Goutham_finetuned_model_name = f"bf_std_model_no_preprocess_retrained_{timestamp}"

        model_bf_stdproject = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
        model_path, train_losses, test_losses = train.train_seg(
            model_bf_stdproject.net,
            train_data=images_training,
            train_labels=labels_training,
            test_data=images_testing,
            test_labels=labels_testing,
            channels=[0, 0],
            normalize=True,
            weight_decay=weight_decay,
            SGD=sgd,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            save_path=model_dir_Goutham_retrain,
            model_name=Goutham_finetuned_model_name,
        )

        run_record[Goutham_finetuned_model_name] = {
            "model_path": model_path,
            "train_losses": train_losses,
            "test_losses": test_losses,
        }

    if train_from_base_cellpose_nuclei_model:
        # fine-tune the basic CellPose nuclei model
        model_dir_from_default = model_dir / "CellPose_default_nuclei_model_finetuning"
        model_dir_from_default.mkdir(exist_ok=True)
        labelfree_nuc_pred_from_default_model_name = f"labelfree_nuc_pred_{timestamp}"

        model_nuclei_original = models.CellposeModel(gpu=gpu, model_type="nuclei")

        model_path, train_losses, test_losses = train.train_seg(
            model_nuclei_original.net,
            train_data=images_training,
            train_labels=labels_training,
            test_data=images_testing,
            test_labels=labels_testing,
            channels=[0, 0],
            normalize=True,
            weight_decay=weight_decay,
            SGD=sgd,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            save_path=model_dir_from_default,
            model_name=labelfree_nuc_pred_from_default_model_name,
        )

        run_record[labelfree_nuc_pred_from_default_model_name] = {
            "model_path": model_path,
            "train_losses": train_losses,
            "test_losses": test_losses,
        }

    # generate plots of the training and test losses
    if any(run_record):
        # load the training and test losses from the run.log file
        # train_losses, test_losses, time_list, epoch_list = get_old_cellpose_train_test_losses(model_dir, model_name_list)

        # save the training and test losses to a file
        for model_name in run_record:
            train_losses = run_record[model_name]["train_losses"]
            test_losses = run_record[model_name]["test_losses"]

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
                model_dir / f"{model_name}_training_test_losses.png",
                bbox_inches="tight",
                dpi=180,
            )
            plt.close(fig)

    # generate a test image to see how the model performs
    # on a live example that it has never seen
    model_nuclei_original_finetuned = models.CellposeModel(
        gpu=False, pretrained_model=str(model_path)
    )

    default_dim_order = get_default_dim_order()
    dim_map = get_dim_map(default_dim_order)

    # load the test image
    test_img_path = Path(get_original_path("20241120_20X"))
    dataset_config = load_dataset_config("20241120_20X")
    test_img_bf_chan = dataset_config.original_channel_indices.brightfield

    test_img = BioImage(test_img_path)
    test_img_dask_arr = test_img.get_image_dask_data(
        default_dim_order, T=[0], C=test_img_bf_chan
    ).std(axis=dim_map["Z"], keepdims=True)
    test_img_arr = test_img_dask_arr.compute().squeeze()

    # run the model on the test image, we're going to be pretty
    # generous with the flow and cellprob threshold settings
    # just to see what is picked up
    test_prediction, flows, probs = model_nuclei_original_finetuned.eval(
        test_img_arr,
        channels=[0, 0],
        min_size=500,
        flow_threshold=0,
        cellprob_threshold=-6.0,
    )

    # plot and save the resulting nuclei prediction
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    image_rescaled = rescale_intensity(
        np.clip(
            test_img_arr,
            a_min=np.percentile(test_img_arr, 1),
            a_max=np.percentile(test_img_arr, 99),
        ),
        out_range=(0, 1),
    )
    ax0.imshow(image_rescaled, cmap="gray")
    ax0.set_title("BF STD")
    ax1.imshow(label2rgb(test_prediction))
    ax1.set_title("Nuclei Predictions")
    overlay = label2rgb(label=test_prediction, image=image_rescaled, bg_label=0)
    ax2.imshow(overlay)
    [ax.set_axis_off() for ax in (ax0, ax1, ax2)]
    plt.tight_layout()
    fig.savefig(model_dir / f"{model_name}_test_image.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ipython_cli_flexecute(main)
