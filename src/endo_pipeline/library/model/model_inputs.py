import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.endo_pipeline.configs import (
    DatasetConfig,
    get_available_zarr_files,
    load_dataset_collection_config,
)
from src.endo_pipeline.io import get_output_path, load_dataframe_from_fms
from src.endo_pipeline.library.process.z_stack_selection import get_plane_indices

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files


def get_model_dir() -> Path:
    """Get the path to `src.endo_pipeline.library.model`."""
    return Path(__file__).resolve().parent


def generate_zarr_csv_for_model_eval(
    dataset_config: DatasetConfig,
    save_path: Path,
    resolution_level: int = 1,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
    overwrite: bool = True,
) -> Path:
    """Generate a CSV file with path to Zarr files for the given dataset."""

    # get unique name for the CSV file
    file_name = "dataset"
    if z_stack_offsets is not None:
        file_name = f"{file_name}_z_stack_{z_stack_offsets[0]}_{z_stack_offsets[1]}"
    if slice_by_global_center:
        file_name = f"{file_name}_ctr"

    file_name = f"{file_name}.csv"
    data_path = save_path / file_name

    # if the file already exists and overwrite is False, return the path
    if data_path.exists() and not overwrite:
        return data_path

    # generate csv with paths to zarr files for each position in the dataset
    zarr_path_list = sorted(get_available_zarr_files(dataset_config))

    df = pd.DataFrame({"path": zarr_path_list})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = resolution_level

    # if z_stack_offsets is not None, add a column with z-slice ranges
    # for each position in the dataset (i.e., zarr file)
    if z_stack_offsets is not None:
        # this is a wrapper function to get z-slice ranges
        # from dataset name and position in the dataset using
        # zarr_file_path our way to get the position
        def _get_z_slices(zarr_file_path: Path, dataset_config: DatasetConfig) -> list[int]:
            # get position from zarr path as an integer (e.g., 'P0' -> 0)
            position_as_int = int(zarr_file_path.stem.split("_")[-1].split(".")[0][-1])
            z_slices = get_plane_indices(
                dataset_config,
                position_as_int,
                lower_offset=z_stack_offsets[0],
                upper_offset=z_stack_offsets[1],
                slice_by_global_center=slice_by_global_center,
            )
            return z_slices

        # apply the function to each zarr file path
        df["Z"] = df["path"].apply(lambda x: _get_z_slices(Path(x), dataset_config))
        df["z_start"] = df["Z"].apply(lambda x: x[0])
        df["z_stop"] = df["Z"].apply(lambda x: x[-1])
        # remove the Z column as it is not needed anymore
        df.drop(columns=["Z"], inplace=True)

    # specify the T column as [0,250,500] for testing purposes
    df["frame_start"] = df["path"].apply(lambda x: 0)
    df["frame_stop"] = df["path"].apply(lambda x: -1)
    df["frame_step"] = df["path"].apply(lambda x: 250)

    # turn paths into strings
    df["path"] = df["path"].apply(lambda x: str(x))

    # save csv and return the path
    df.to_csv(data_path, index=False)
    return data_path


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig,
    save_dir: Path,
    downsample_factor: int = 2,
) -> Path:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""
    fms_id = dataset_config.live_merged_seg_features_manifest_fmsid
    if fms_id is None:
        raise ValueError(
            f"Dataset {dataset_config.name} does not have a live segmentation features FMS ID."
        )
    df = load_dataframe_from_fms(fms_id)

    # keep only rows that were not filtered out by filter_global
    df = df[~df["filter_global"]]

    # filter the dataframe to include only the relevant columns
    colums_to_keep = [
        "zarr_path",
        "image_index",
        "track_id",
        "label",
        "centroid_X",
        "centroid_Y",
        "image_size_x",
        "image_size_y",
        "crop_size",
    ]
    df = df[colums_to_keep]

    # convert centroids to bounding boxes and downsample
    # by half to match currently used model resolution
    # this is currently always 2
    df = centroid_to_bbox(df, downsample_factor)

    # filter the dataframe to exclude anything where the size of
    # the bounding box does not match the downsampled crop size
    # (because the model expects identically sized square crops)
    # check if bounding boxes fit in image bounds without being clipped
    bbox_size_is_correct = bbox_in_image_bounds(df, downsample_factor)
    # filter the dataframe in-place to remove clipped bounding boxess
    df = df[bbox_size_is_correct]

    # group df by zarr_path and convert start and end coordinates to list
    grouped_df = (
        df.groupby(["zarr_path", "image_index"])
        .agg(
            {
                "start_y": lambda x: list(x),
                "start_x": lambda x: list(x),
                "end_y": lambda x: list(x),
                "end_x": lambda x: list(x),
                "track_id": lambda x: list(x),
            }
        )
        .reset_index()
    )
    grouped_df["channel"] = ZARR_BF_CHANNEL
    # NOTE "resolution" below determines what resolution the images will
    # be loaded at, and currently the model loads at native resolution
    # and downsamples in the transforms; therefore this value must be 0
    # The "start" and "end" column values determine the crop locations
    # after downsampling, thus they were adjusted by downsample_factor
    grouped_df["resolution"] = 0
    # only run a single timepoint from zarr
    grouped_df["start"] = grouped_df["image_index"]
    grouped_df["stop"] = grouped_df["image_index"]
    grouped_df.rename({"zarr_path": "path", "image_index": "T"}, axis=1, inplace=True)

    save_path = save_dir / "aggregated_crop_manifest.csv"
    grouped_df.to_csv(save_path, index=False)
    return save_path


def centroid_to_bbox(df: pd.DataFrame, downsample_factor: int = 2) -> pd.DataFrame:
    """
    Convert centroids to bounding boxes.

    Note: coordinates are downsampled by half (downsample_factor = 2)
    to match current model resolution.
    """
    df["start_x"] = ((df["centroid_X"] - df["crop_size"] / 2) / downsample_factor).astype(int)
    df["start_y"] = ((df["centroid_Y"] - df["crop_size"] / 2) / downsample_factor).astype(int)
    df["end_x"] = ((df["centroid_X"] + df["crop_size"] / 2) / downsample_factor).astype(int)
    df["end_y"] = ((df["centroid_Y"] + df["crop_size"] / 2) / downsample_factor).astype(int)
    return df


def bbox_in_image_bounds(df: pd.DataFrame, downsample_factor: int = 2) -> pd.Series:
    # adjust the image size according to the desired downsample factor
    df["image_size_x"] = df["image_size_x"] // downsample_factor
    df["image_size_y"] = df["image_size_y"] // downsample_factor

    # limit start and end of x and y bboxes to be within image size limits
    df["start_x"] = df["start_x"].transform(lambda x: max(0, x))
    df["start_y"] = df["start_y"].transform(lambda y: max(0, y))
    df["end_x"] = df[["end_x", "image_size_x"]].min(axis=1)
    df["end_y"] = df[["end_y", "image_size_y"]].min(axis=1)

    # filter the dataframe to exclude anything where the size of
    # the bounding box does not match the downsampled crop size
    # (because the model expects identically sized square crops)
    bbox_size_y = df.end_y - df.start_y
    bbox_size_x = df.end_x - df.start_x
    bbox_size_is_correct = (bbox_size_y == (df["crop_size"] // downsample_factor)) & (
        bbox_size_x == (df["crop_size"] // downsample_factor)
    )  # ask if both x and y bbox dimensions equal downsampled crop size
    return bbox_size_is_correct


def generate_overrides_for_model_training(
    model_name: str,
    crop_size: int,
    train_csv_path: Path,
    val_csv_path: Path,
) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    Parameters
    ----------
    model_name: str
        The name of the model to train.

    crop_size: int
        The number of pixels in each dimension of the
        image crop to use for training.

        That is, the cropped image will be square
        with size (crop_size px, crop_size px).

    train_csv_path: Path | None
        The path to the training dataset CSV file.
        If None, the default path for the output of
        generate_csv_for_training_diffae will be used.

    val_csv_path: Path | None
        The path to the validation dataset CSV file.
        If None, the default path for the output of
        generate_csv_for_training_diffae will be used.
    """
    # create output directories if they do not exist
    train_output_path = get_output_path("models", model_name, "train", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "logs", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "checkpoints", include_timestamp=False)

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.csv_path": train_csv_path.as_posix(),
        "data.predict_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        "data.val_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        # get repo root directory and current working directory
        "paths.root_dir": Path(__file__).resolve().parents[3],
        "paths.work_dir": os.getcwd(),
        # save outputs to user-specified directory
        "paths.output_dir": (train_output_path / "logs").as_posix(),
        "paths.log_dir": "${paths.output_dir}",
        "callbacks.model_checkpoint.dirpath": (train_output_path / "checkpoints").as_posix(),
        # update run name
        "run_name": model_name,
        # set crop size from input via model.image_shape,
        # the rest are populated by interpolation
        "model.image_shape": [1, crop_size, crop_size],
    }
    return overrides


def generate_overrides_for_model_eval(
    user_overrides: dict,
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> dict:
    """
    Generate overrides for the CytoDLModel configuration
    for evaluating model `model_name` on crops of
    images from dataset `dataset_name`.
    """
    overrides = {
        # train and val dataloaders are unnecessary for prediction
        # and might be slow to instantiate (e.g. if they cache data)
        "data.train_dataloaders": None,
        "data.val_dataloaders": None,
        "data.predict_dataloaders.num_workers": 8,
        "data.predict_dataloaders.batch_size": 1,
        "data.predict_dataloaders.dataset.csv_path": data_path,
        "paths.output_dir": save_path,
        # change checkpoint path to the one downloaded from mlflow
        "checkpoint.ckpt_path": ckpt_path,
        "checkpoint.strict": True,
        "callbacks": None,
        "callbacks.prediction_saver": {
            "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
            "save_dir": save_path,
            "meta_keys": [
                "T",
                "start_y",
                "start_x",
                "filename_or_obj",
            ],
            "save_suffix": f"{dataset_name}_{model_name}_features",
        },
    }
    overrides.update(user_overrides)
    return overrides


def generate_overrides_for_track_based_crops(
    user_overrides: dict[str, Any],
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Generate overrides for the CytoDLModel configuration
    to evaluate model `model_name` on crops of
    tracked objects in dataset `dataset_name`.
    """
    overrides = generate_overrides_for_model_eval(
        user_overrides,
        save_path=save_path,
        data_path=data_path,
        ckpt_path=ckpt_path,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # additional overrides specific to track-based crops
    track_specific_overrides = {
        "callbacks.prediction_saver": {
            "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
            "save_dir": save_path,
            "meta_keys": [
                "T",
                "start_y",
                "start_x",
                "end_y",
                "end_x",
                "filename_or_obj",
                "track_id",
            ],
            "save_suffix": f"{dataset_name}_{model_name}_tracked_crop_features",
        },
        # add cropping transform
        "data.predict_dataloaders.dataset.transform.transforms[6]": {
            "_target_": "cyto_dl.image.transforms.coordinate_crop.CropToCoordsd",
            "keys": ["raw_bf"],
            "start_keys": ["start_y", "start_x"],
            "end_keys": ["end_y", "end_x"],
            "meta_keys": ["track_id"],
        },
        # persist coordinate data through MultiDimImageDataset
        "data.predict_dataloaders.dataset.extra_columns": [
            "start_y",
            "start_x",
            "end_y",
            "end_x",
            "track_id",
        ],
        # no spatial inferer needed
        "model.spatial_inferer": None,
    }
    overrides.update(track_specific_overrides)
    return overrides


def generate_overrides_for_finetuning(
    model_name: str,
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    ckpt_path: Path,
) -> dict:
    """
    Generate overrides for finetuning a DiffAE model.

    Parameters
    ----------
    model_name: str
        The name of the model to finetune. This should correspond to a
        directory in `results/models/` and match the model name used during the
        `paired_data_validation` step.
    dataset_pair_type: Literal['live_fixed', '20x_40x']
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.
    train_csv_path: Path
        The path to the training CSV file containing paired data.
    val_csv_path: Path
        The path to the validation CSV file containing paired data.
    ckpt_path: Path
        The path to the DiffAE checkpoint to finetune.
    """
    # create output directories if they do not exist
    save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "checkpoints",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "logs",
        include_timestamp=False,
    )

    overrides = {
        # point to already projected paired dataset
        "data.train_dataloaders.dataset.csv_path": str(train_csv_path),
        "data.val_dataloaders.dataset.csv_path": str(val_csv_path),
        # load diffae checkpoint to finetune
        "checkpoint.ckpt_path": str(ckpt_path),
        "checkpoint.weights_only": True,
        "checkpoint.strict": False,
        # save to user-specified directory
        "model.save_dir": (save_path / "logs").as_posix(),
        "trainer.default_root_dir": save_path,
        "callbacks.model_checkpoint.dirpath": (save_path / "checkpoints").as_posix(),
        "paths.output_dir": (save_path / "logs").as_posix(),
        # do training
        "train": True,
        # # make sure that last ckpt is saved
        # "callbacks.model_checkpoint.monitor": None,
        # updated mlflow logger
        "logger": {
            "mlflow": {
                "_target_": "cyto_dl.loggers.MLFlowLogger",
                "tracking_uri": "https://production.int.allencell.org/mlflow/",
                "experiment_name": "endo_diffae",
                "run_name": "fixed_finetune_separate_encoder",
            }
        },
    }

    return overrides


def get_dataset_names_used_for_training(
    train_csv_path: Path, val_csv_path: Path, dataset_collection_name: str
) -> list[str]:
    """
    Pull list of dataset names used for model training
    from train.csv and val.csv files that are passed
    into the model training script.
    """
    # load train.csv and val.csv files as dataframes
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # get date part of dataset name from zarr path
    # note: this might be something that
    # gets turned into a zarr method in a future PR
    for df in [train_df, val_df]:
        df["dataset_date"] = df["path"].apply(lambda s: Path(s).stem.split("_")[0])

    # get unique dataset dates used in training from dataset_date
    # by combining the unique dates from both train and val datasets
    training_dataset_dates = list(
        set(train_df["dataset_date"].unique().tolist() + val_df["dataset_date"].unique().tolist())
    )

    # get unique dataset names by looping over
    # the provided dataset collection name,
    # which should be a superset of the datasets used for training

    training_dataset_superset = load_dataset_collection_config(dataset_collection_name)
    training_dataset_names = []
    for dataset_name in training_dataset_superset.datasets:
        for date in training_dataset_dates:
            if date in dataset_name:
                training_dataset_names.append(dataset_name)

    return sorted(training_dataset_names)
