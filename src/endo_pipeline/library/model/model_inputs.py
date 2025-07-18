from pathlib import Path
from typing import Any

import pandas as pd

from src.endo_pipeline.configs import DatasetConfig
from src.endo_pipeline.io import load_dataframe_from_fms


def generate_zarr_csv_for_model_eval(
    dataset_config: DatasetConfig, save_path: Path, resolution_level: int = 1
) -> str:
    """Generate a CSV file with path to Zarr files for the given dataset."""
    # generate csv with paths to zarr files
    # this replaces the call to get_zarr_path from dataset_io
    zarr_path_list = list(Path(dataset_config.zarr_path).glob("*.zarr"))
    zarr_path_dict = {}
    for path in zarr_path_list:
        zarr_path_dict[path.name] = str(path)

    df = pd.DataFrame({"path": sorted(zarr_path_dict.values())})
    df["channel"] = dataset_config.brightfield_channel_index
    df["resolution"] = resolution_level
    data_path = Path(save_path / "dataset.csv").as_posix()
    df.to_csv(data_path, index=False)
    return data_path


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig, save_dir: Path
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
    downsample_factor = 2
    df = centroid_to_bbox(df, downsample_factor)

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
    df = df[bbox_size_is_correct]  # filter the dataframe in-place

    # NOTE: take first and last 5 rows for testing purposes
    # df = pd.concat([df.head(), df.tail()])
    df = pd.concat(
        [
            # df[df["image_index" == 0]].head(),
            df[df["image_index" == 570]].head(),
            df[df["image_index" == 576]].head(),
        ]
    )

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
    grouped_df["channel"] = dataset_config.brightfield_channel_index
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
        "data.predict_dataloaders.num_workers": 128,
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
            "save_suffix": f"{dataset_name}_{model_name}_track_based_features",
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
