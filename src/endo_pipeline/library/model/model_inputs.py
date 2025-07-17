from pathlib import Path
from typing import Any

import pandas as pd

from src.endo_pipeline.configs import (
    DatasetConfig,
    get_live_20X_objective_3i_microscope_datasets,
    load_dataset_collection_config,
)
from src.endo_pipeline.io import load_dataframe_from_fms

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files


def generate_zarr_csv_for_model_eval(
    dataset_config: DatasetConfig, save_path: Path, resolution_level: int = 1
) -> Path:
    """Generate a CSV file with path to Zarr files for the given dataset."""
    # generate csv with paths to zarr files
    # this replaces the call to get_zarr_path from dataset_io
    zarr_path_list = list(Path(dataset_config.zarr_path).glob("*.zarr"))
    zarr_path_dict = {}
    for path in zarr_path_list:
        zarr_path_dict[path.name] = str(path)

    df = pd.DataFrame({"path": sorted(zarr_path_dict.values())})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = resolution_level
    data_path = save_path / "dataset.csv"
    df.to_csv(data_path, index=False)
    return data_path


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig, save_dir: Path
) -> Path:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""
    fms_id = dataset_config.tracking_integration_fmsid
    if fms_id is None:
        raise ValueError(
            f"Dataset {dataset_config.name} does not have a tracking integration FMS ID."
        )
    df = load_dataframe_from_fms(fms_id)
    # convert centroids to bounding boxes
    df = centroid_to_bbox(df)

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
    grouped_df["resolution"] = 0
    # only run a single timepoint from zarr
    grouped_df["start"] = grouped_df["image_index"]
    grouped_df["stop"] = grouped_df["image_index"]
    grouped_df.rename({"zarr_path": "path", "image_index": "T"}, axis=1, inplace=True)

    save_path = save_dir / "aggregated_crop_manifest.csv"
    grouped_df.to_csv(save_path, index=False)
    return save_path


def centroid_to_bbox(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert centroids to bounding boxes.

    Note: coordinates are downsampled by half to match current model resolution.
    """
    df["start_x"] = ((df["centroid_x"] - df["crop_size"] / 2) / 2).astype(int)
    df["start_y"] = ((df["centroid_y"] - df["crop_size"] / 2) / 2).astype(int)
    df["end_x"] = ((df["centroid_x"] + df["crop_size"] / 2) / 2).astype(int)
    df["end_y"] = ((df["centroid_y"] + df["crop_size"] / 2) / 2).astype(int)
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


def get_dataset_names_used_for_training(train_csv_path: Path, val_csv_path: Path) -> list[str]:
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
        df["dataset_date"] = df["path"].apply(
            lambda s: s.split(".")[0].split("/")[-1].split("_")[0]
        )

    # get unique dataset dates used in training from dataset_date
    # by combining the unique dates from both train and val datasets
    training_dataset_dates = list(
        set(train_df["dataset_date"].unique().tolist() + val_df["dataset_date"].unique().tolist())
    )

    # get unique dataset names by looping over
    # all live, 20X objective datasets from the
    # 3i microscope collection (these are the possible
    # datasets used for training)
    # where there are datasets with the same date,
    # get the one with 20X in the name (i.e., don't
    # get the 40X pair of the two) except for Nikon datasets
    try:
        training_appropriate_dataset_collection = load_dataset_collection_config(
            "live_20X_objective_3i_microscope"
        )
    except FileNotFoundError:
        # if the collection is not found, run the function
        training_appropriate_dataset_collection = get_live_20X_objective_3i_microscope_datasets()
    training_dataset_names = []
    for dataset_name in training_appropriate_dataset_collection.datasets:
        for date in training_dataset_dates:
            if date in dataset_name:
                training_dataset_names.append(dataset_name)

    return sorted(training_dataset_names)
