import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import CytoDLModelConfig, DatasetConfig, add_model_manifest
from src.endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from src.endo_pipeline.library.model.mlflow_utils import download_mlflow_artifact, download_model
from src.endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    save_dataframe_manifest,
)

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files


def get_cytodl_commit_hash(run_id: str, model_path: Path) -> str:
    """
    Extract commit hash from the requirements file uploaded to mlflow.

    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    model_path: Path
        The path where the downloaded model artifacts are saved.
    """
    try:
        artifact_path = Path("requirements/train-requirements.txt")
        download_mlflow_artifact(run_id, artifact_path, model_path)
    except ValueError:
        artifact_path = Path("requirements/eval-requirements.txt")
        download_mlflow_artifact(run_id, artifact_path, model_path)

    with open(model_path / artifact_path) as f:
        lines = f.readlines()
    for line in lines:
        if "git+" in line and "cyto-dl" in line:
            commit_hash = line.split("git+")[1].split("#egg")[0].split("/")[-1]
            return commit_hash
    raise ValueError("No commit hash found in requirements.txt")


def load_overrides(overrides: str | dict | None) -> dict:
    """
    Load overrides from a string or dictionary.

    If None, return an empty dictionary.
    """
    if isinstance(overrides, str):
        overrides_dict = json.loads(overrides)
    elif overrides is None:
        overrides_dict = {}
    elif isinstance(overrides, dict):
        overrides_dict = overrides
    elif not isinstance(overrides, dict):
        raise ValueError("Overrides must be a dictionary or a string")
    return overrides_dict


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
        "extras.print_config": False,
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


def generate_zarr_csv_for_model_eval(
    dataset_config: DatasetConfig,
    save_path: Path,
    zarr_resolution: int = 1,
    workflow_testing: bool = False,
) -> Path:
    """Generate a CSV file with path to Zarr files for the given dataset."""
    # generate csv with paths to zarr files
    # this replaces the call to get_zarr_path from dataset_io
    # note that this will likely be refactored after
    # the new zarr methods merge
    zarr_path_list = list(Path(dataset_config.zarr_path).glob("*.zarr"))
    zarr_path_dict = {}
    for path in zarr_path_list:
        zarr_path_dict[path.name] = str(path)

    df = pd.DataFrame({"path": sorted(zarr_path_dict.values())})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = zarr_resolution

    if workflow_testing:
        # for workflow testing, only use first position from each dataset
        # and first two timepoints to speed up the dataloading process
        # (if dataset is not timelapse, then only one timepoint is used)
        df = df.head(1)
        df["start"] = 0
        df["stop"] = 1 if dataset_config.is_timelapse else 0

    data_path = save_path / "dataset.csv"
    df.to_csv(data_path, index=False)
    return data_path


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig,
    save_dir: Path,
    downsample_factor: int = 2,
) -> Path:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""

    manifest = load_dataframe_manifest("live_merged_seg_features")
    location = get_dataframe_location_for_dataset(manifest, dataset_config.name)
    df = load_dataframe(location)

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
    df = _centroid_to_bbox(df, downsample_factor)

    # filter the dataframe to exclude anything where the size of
    # the bounding box does not match the downsampled crop size
    # (because the model expects identically sized square crops)
    # check if bounding boxes fit in image bounds without being clipped
    bbox_size_is_correct = _bbox_in_image_bounds(df, downsample_factor)
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


def _centroid_to_bbox(df: pd.DataFrame, downsample_factor: int = 2) -> pd.DataFrame:
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


def _bbox_in_image_bounds(df: pd.DataFrame, downsample_factor: int = 2) -> pd.Series:
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


def update_prediction_from_crops_with_metadata(
    dataset_name: str,
    model_name: str,
    crop_size: list[int],
    mlflow_id: str,
    save_path: Path,
) -> Path:
    """
    Update the prediction file with metadata,
    return the path to the updated prediction file.
    """
    # add model and dataset information to prediction file
    prediction_path = save_path / f"predict_{dataset_name}_{model_name}_features.parquet"
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_name"] = model_name
    pred_df["mlflow_id"] = mlflow_id

    # note: the current model loads images at resolution
    # level 0 and downsamples in the transforms.
    pred_df["resolution_level"] = 1

    pred_df["end_y"] = pred_df["start_y"] + crop_size[0]
    pred_df["end_x"] = pred_df["start_x"] + crop_size[1]
    pred_df["crop_size_y"] = crop_size[0]
    pred_df["crop_size_x"] = crop_size[1]

    pred_df["position"] = pred_df["filename_or_obj"].apply(
        lambda s: Path(s).stem.split("_")[-1].split(".")[0]
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path


def update_prediction_from_tracks_with_metadata(
    dataset_name: str, model_name: str, mlflow_id: str, save_path: Path
) -> Path:
    """Update the prediction file with metadata."""
    # add model and dataset information to prediction file
    prediction_path = (
        save_path / f"predict_{dataset_name}_{model_name}_tracked_crop_features.parquet"
    )
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_name"] = model_name
    pred_df["mlflow_id"] = mlflow_id

    # NOTE: the current model loads images at resolution level 0 and downsamples in the transforms.
    pred_df["resolution_level"] = 1

    crop_size = (
        pred_df["end_y"].iloc[0] - pred_df["start_y"].iloc[0],
        pred_df["end_x"].iloc[0] - pred_df["start_x"].iloc[0],
    )
    pred_df["crop_size_y"] = crop_size[0]
    pred_df["crop_size_x"] = crop_size[1]
    pred_df["position"] = pred_df["filename_or_obj"].apply(
        lambda s: Path(s).stem.split("_")[-1].split(".")[0]
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path


def apply_model_on_grid_of_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    zarr_resolution: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    workflow_testing: bool = False,
) -> CytoDLModelConfig:
    """
    Apply a DiffAE model to a single dataset.

    Parameters
    ----------
    model_config
        Configuration of the model to apply.
    dataset_config
        Configuration of the dataset to apply the model to.
    zarr_resolution
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        Whether to upload the prediction file to FMS. Default is True.
    save_path
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides
        Optional user overrides to apply to the model config.
    workflow_testing
        Flag to indicate if this script is being run for testing purposes (e.g., code review).

        If True, then only one position and minimal timepoints from the dataset is included for
        loading and performing inferrence on the crops.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    overrides = load_overrides(user_overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, "train", include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    # set default output path
    save_path = get_output_path(
        "models", model_config.name, dataset_config.name, include_timestamp=False
    )

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    # create zarr dataset
    data_path = generate_zarr_csv_for_model_eval(
        dataset_config, save_path, zarr_resolution, workflow_testing
    )

    # apply overrides
    overrides = generate_overrides_for_model_eval(
        overrides,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
    model.override_config(overrides)
    model.predict()
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = update_prediction_from_crops_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        crop_size=crop_size,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        # build FMS annotations
        dataset_annotations = build_fms_annotations(
            dataset_config,
            model=model_config,
        )

        # upload prediction file to FMS and get file ID
        file_id = upload_file_to_fms(
            prediction_path,
            annotations=dataset_annotations,
            file_type="parquet",
        )

        # add new manifest to model config
        model_config = add_model_manifest(
            model_config,
            dataset_config.name,
            file_id,
        )

    return model_config


def apply_model_on_tracked_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    save_path: str | Path | None = None,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
) -> None:
    """
    Apply a DiffAE model to a single dataset with
    cell segmentation and tracking.

    Parameters
    ----------
    model_config: CytoDLModelConfig
        Configuration of the model to apply.
    dataset_config: DatasetConfig
        Configuration of the dataset to apply the model to.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str or Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides: str or dict or None
        Optional user overrides to apply to the model config.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    overrides = load_overrides(user_overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    if save_path is None:
        # if no save path is provided, use the default path
        save_path = get_output_path(
            "models", model_config.name, dataset_config.name, include_timestamp=False
        )
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    data_path = preprocess_tracking_manifest_for_model_eval(dataset_config, save_path)

    # apply overrides
    overrides = generate_overrides_for_track_based_crops(
        overrides,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
    model.override_config(overrides)
    model.predict()

    prediction_path = update_prediction_from_tracks_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        mlflow_id=mlflow_id,
        save_path=save_path,
    )

    if upload_to_fms:
        # build FMS annotations
        dataset_annotations = build_fms_annotations(
            dataset_config,
            model=model_config,
        )

        # upload prediction file to FMS and get file ID
        file_id = upload_file_to_fms(
            prediction_path,
            annotations=dataset_annotations,
            file_type="parquet",
        )

        # Store FMS ID in dataframe manifest

        manifest_name = "diffae_tracking_integration"
        workflow_name = "apply_diffae_model_on_tracked_crops"

        try:
            manifest = load_dataframe_manifest(manifest_name)
        except FileNotFoundError:
            manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

        manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
        save_dataframe_manifest(manifest)
