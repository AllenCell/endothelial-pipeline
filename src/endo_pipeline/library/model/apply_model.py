import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from src.endo_pipeline.configs import (
    CytoDLModelConfig,
    DatasetConfig,
    get_available_zarr_files,
    get_position_integer_from_zarr_file_path,
    get_position_string_from_zarr_file_path,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from src.endo_pipeline.library.model.image_loading import build_zarr_image_loading_dataframe
from src.endo_pipeline.library.model.mlflow_utils import download_mlflow_artifact, download_model
from src.endo_pipeline.library.process.z_stack_selection import get_plane_indices
from src.endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    save_dataframe_manifest,
)

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files

logger = logging.getLogger(__name__)


def get_model_dir() -> Path:
    """Get the path to `src.endo_pipeline.library.model`."""
    return Path(__file__).resolve().parent


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
    prediction_filename_suffix: str | None = None,
    num_workers: int = 128,
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
        "data.predict_dataloaders.num_workers": num_workers,
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
            "save_suffix": prediction_filename_suffix or f"{dataset_name}_{model_name}_features",
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
    prediction_filename_suffix: str | None = None,
) -> dict[str, Any]:
    """
    Generate overrides for the CytoDLModel configuration
    to evaluate model `model_name` on crops of
    tracked objects in dataset `dataset_name`.
    """
    if prediction_filename_suffix is None:
        prediction_filename_suffix = f"{dataset_name}_{model_name}_tracked_crop_features"

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
            "save_suffix": prediction_filename_suffix,
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
    prediction_path: Path,
) -> None:
    """
    Update the prediction file with metadata,
    return the path to the updated prediction file.
    """
    # add model and dataset information to prediction file
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
        lambda s: get_position_string_from_zarr_file_path(s)
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)


def update_prediction_from_tracks_with_metadata(
    dataset_name: str, model_name: str, mlflow_id: str, prediction_path: Path
) -> None:
    """Update the prediction file with metadata."""
    # add model and dataset information to prediction file
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
        lambda s: get_position_string_from_zarr_file_path(s)
    )
    pred_df.rename(columns={"filename_or_obj": "zarr_path", "T": "frame_number"}, inplace=True)
    pred_df.to_parquet(prediction_path)


def _get_zarr_dataframe_for_z_offsets(
    dataset_config: DatasetConfig,
    resolution_level: int,
    z_stack_offsets: tuple[int, int],
    slice_by_global_center: bool = True,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
    only_positions: list[int] | None = None,
) -> pd.DataFrame:
    """
    Get a dataframe with zarr loading metadata when z-slice selection is based
    on the center slice for each position in the dataset.
    """
    # if z_stack_offsets is not None, get z-slice ranges
    # for each position in the dataset (i.e., zarr file)
    z_slice_by_position = None
    available_zarr_files = get_available_zarr_files(dataset_config)
    if z_stack_offsets is not None:
        z_slice_by_position = []
        for zarr_file_path in available_zarr_files:
            # get position from zarr path as an integer (e.g., 'P0' -> 0)
            position_as_int = get_position_integer_from_zarr_file_path(zarr_file_path)
            # get z-slice indices for the given position
            z_slices = get_plane_indices(
                dataset_config,
                position_as_int,
                lower_offset=z_stack_offsets[0],
                upper_offset=z_stack_offsets[1],
                slice_by_global_center=slice_by_global_center,
            )
            z_slice_by_position.append(z_slices)

    # generate dataframe with zarr loading metadata
    # for each position in the dataset
    # done this way because z-stack offsets are generally position-specific
    df_per_position = []
    for i in range(len(available_zarr_files)):
        if only_positions is not None and i not in only_positions:
            continue
        else:
            # build dataframe for position
            # and append it to the list
            if only_positions is None:
                only_position = [i]
            else:
                only_position = [only_positions[i]]
            logger.debug("Building zarr dataframe for position [ %s ]", only_position[0])
            df_per_position.append(
                build_zarr_image_loading_dataframe(
                    dataset_config,
                    resolution_level=resolution_level,
                    channel=ZARR_BF_CHANNEL,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    frame_step=frame_step,
                    z_start=z_slice_by_position[i][0] if z_stack_offsets else None,
                    z_stop=z_slice_by_position[i][-1] if z_stack_offsets else None,
                    only_positions=only_position,
                )
            )
    # concatenate dataframes for all positions
    df = pd.concat(df_per_position, ignore_index=True)
    return df


def apply_model_on_grid_of_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
    testing_mode: bool = False,
) -> CytoDLModelConfig:
    """
    Apply a DiffAE model to a single dataset.

    **Workflow testing**

    If ``testing_mode`` is set to True, the model will only be applied to the first
    position of the dataset and only the first two timepoints will be used. The
    staging environment of FMS will be used for uploading the prediction file.

    **Z-stack offsets**

    The ``z_stack_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_stack_offsets`` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    If ``slice_by_global_center`` is set to True, the z-slice range is calculated based on
    the global center plane for the given position. In this case, ``z_stack_offsets`` should
    indicate the number of slices to include below and above the center plane. Else, the
    ``z_stack_offsets`` are used directly as the range bounds.


    Parameters
    ----------
    model_config
        Configuration of the model to apply.
    dataset_config
        Configuration of the dataset to apply the model to.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        Whether to upload the prediction file to FMS. Default is True.
    save_path
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides
        Optional user overrides to apply to the model config.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center: bool
        Get global center plane per position for z-slicing if True, use offsets directly if False.
    testing_mode
        Execute method in workflow testing mode if True, run full model evaluation if False.

    Returns
    -------
    :
        Creates dataframe of features extracted from the crops and saves it to the output path.

        If ``upload_to_fms`` is True, uploads the prediction file to FMS and adds the file ID to the
        model config manifest.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please run on a GPU machine.")
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")

    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, "train")
    path_dict = download_model(mlflow_id, model_path)

    # right now, need to use the tracked version of the config if using the
    # "legacy" model "diffae_04_10" (temporary workaround until we are only using
    # models trained with the new pipeline)
    if model_config.name == "diffae_04_10":
        path_dict["config_path"] = get_model_dir() / "diffae_04_10_eval.yaml"
        logger.info(
            "Loading legacy model config for diffae_04_10 from [ %s ]", path_dict["config_path"]
        )

    # set default output path
    save_path = get_output_path("models", model_config.name, dataset_config.name)

    # use timestamp to get unique file name for FMS upload later
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    logger.debug("Applying model [ %s ] to dataset [ %s ]", model_config.name, dataset_config.name)
    # get unique name for the CSV file
    file_name = "dataset"
    if z_stack_offsets is not None:
        file_name = f"{file_name}_z_stack_{z_stack_offsets[0]}_{z_stack_offsets[1]}"
        if slice_by_global_center:
            file_name = f"{file_name}_ctr"

    file_name = f"{file_name}_{timestamp}.csv"
    dataset_save_path = save_path / file_name

    # default frame start and stop values are None, i.e., load all timepoints
    frame_start = None
    frame_stop = None
    frame_step = None
    only_positions = None  # keep all rows in the dataset CSV

    if testing_mode:
        # for workflow testing, only use first position from each dataset
        # and first two timepoints to speed up the dataloading process
        # (if dataset is not timelapse, then only one timepoint is used)
        frame_start = 0
        frame_stop = 1 if dataset_config.is_timelapse else 0
        only_positions = [0]  # only use the first position
        logger.debug(
            "Workflow testing is enabled, only processing the first few timepoints "
            "of the first position the dataset."
        )

    if z_stack_offsets is not None:
        # load timepoints 0, 250, and 500 for z-stack offsets summary
        frame_start = 0
        frame_stop = -1
        frame_step = 250
        logger.debug(
            "Using z-stack offsets: [ %s ] with slice_by_global_center = [ %s ] ",
            z_stack_offsets,
            slice_by_global_center,
        )
        logger.debug("Z-stack offsets provided, getting features only for frames 0, 250, and 500.")

        # get the dataframe with zarr loading metadata
        df = _get_zarr_dataframe_for_z_offsets(
            dataset_config,
            resolution_level=resolution_level,
            z_stack_offsets=z_stack_offsets,
            slice_by_global_center=slice_by_global_center,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            only_positions=only_positions,
        )
    else:
        # if no z-stack offsets are provided, can get the dataframe
        # directly from the build_zarr_image_loading_dataframe function
        logger.debug("No z-stack offsets provided, loading all z-slices.")
        df = build_zarr_image_loading_dataframe(
            dataset_config,
            resolution_level=resolution_level,
            channel=ZARR_BF_CHANNEL,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            only_positions=only_positions,
        )

    # save the dataframe to a CSV file
    df.to_csv(dataset_save_path, index=False)

    # apply overrides
    prediction_filename_suffix = f"{dataset_config.name}_{model_config.name}_features_{timestamp}"
    # having issues with zarr loading when using z-slices from global center,
    # need to decrease the num_workers
    num_workers = 64 if (z_stack_offsets is not None and slice_by_global_center) else 128
    logger.debug("Using [ %d ] workers for data loading.", num_workers)
    overrides = generate_overrides_for_model_eval(
        load_overrides(user_overrides),
        save_path=str(save_path),
        data_path=str(dataset_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        prediction_filename_suffix=prediction_filename_suffix,
        num_workers=num_workers,
    )
    model.override_config(overrides)
    local_config_save_path = get_output_path("models", "evaluation_configs")
    model.save_config(local_config_save_path / f"{model_config.name}_eval.yaml")
    logger.info(
        "Evaluation config saved to [ %s ]",
        local_config_save_path / f"{model_config.name}_eval.yaml",
    )
    logger.debug("Starting model prediction...")
    model.predict()
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = save_path / f"predict_{prediction_filename_suffix}.parquet"
    update_prediction_from_crops_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        crop_size=crop_size,
        mlflow_id=mlflow_id,
        prediction_path=prediction_path,
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
        manifest_name = model_config.name
        workflow_name = "apply_diffae_grid"

        if z_stack_offsets is not None:
            manifest_name = f"{manifest_name}_z_stack_{z_stack_offsets[0]}_{z_stack_offsets[1]}"
            parameters = {"z_stack_offsets": z_stack_offsets}
        else:
            parameters = {}

        try:
            manifest = load_dataframe_manifest(manifest_name)
        except FileNotFoundError:
            manifest = DataframeManifest(
                name=manifest_name, workflow=workflow_name, parameters=parameters
            )

        manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
        save_dataframe_manifest(manifest)

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

    # use timestamp to get unique file name for FMS upload later
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    prediction_filename_suffix = f"{dataset_config.name}_{model_config.name}_tracked_crop_features"
    prediction_filename_suffix = f"{prediction_filename_suffix}_{timestamp}"
    # apply overrides
    overrides = generate_overrides_for_track_based_crops(
        overrides,
        save_path=str(save_path),
        data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        prediction_filename_suffix=prediction_filename_suffix,
    )
    model.override_config(overrides)
    model.predict()

    prediction_path = save_path / f"predict_{prediction_filename_suffix}.parquet"
    update_prediction_from_tracks_with_metadata(
        dataset_name=dataset_config.name,
        model_name=model_config.name,
        mlflow_id=mlflow_id,
        prediction_path=prediction_path,
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


### BELOW HERE IS TEST CODE, NOT USED IN PRODUCTION ###
def generate_overrides_for_array_inputs(
    user_overrides: dict[str, Any],
    # save_path: str,
    # data_path: str,
    ckpt_path: str,
    # dataset_name: str,
    # model_name: str,
    # prediction_filename_suffix: str | None = None,
    data: np.ndarray | list[np.ndarray],
    transforms: dict | None = None,
    num_workers: int = 1,  # 128,
    batch_size: int = 1,
) -> dict[str, Any]:
    # """
    # Generate overrides for the CytoDLModel configuration
    # to evaluate model `model_name` on crops of
    # tracked objects in dataset `dataset_name`.
    # """
    # if prediction_filename_suffix is None:
    #     prediction_filename_suffix = f"{dataset_name}_{model_name}_tracked_crop_features"

    # overrides = generate_overrides_for_model_eval(
    #     user_overrides,
    #     save_path=save_path,
    #     data_path=data_path,
    #     ckpt_path=ckpt_path,
    #     dataset_name=dataset_name,
    #     model_name=model_name,
    # )

    # additional overrides specific to track-based crops
    # track_specific_overrides = {
    #     "callbacks.prediction_saver": {
    #         "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
    #         "save_dir": save_path,
    #         "meta_keys": [
    #             "T",
    #             "start_y",
    #             "start_x",
    #             "end_y",
    #             "end_x",
    #             "filename_or_obj",
    #             "track_id",
    #         ],
    #         "save_suffix": prediction_filename_suffix,
    #     },
    # # add cropping transform
    # "data.predict_dataloaders.dataset.transform.transforms[6]": {
    #     "_target_": "cyto_dl.image.transforms.coordinate_crop.CropToCoordsd",
    #     "keys": ["raw_bf"],
    #     "start_keys": ["start_y", "start_x"],
    #     "end_keys": ["end_y", "end_x"],
    #     "meta_keys": ["track_id"],
    # },
    # # persist coordinate data through MultiDimImageDataset
    # "data.predict_dataloaders.dataset.extra_columns": [
    #     "start_y",
    #     "start_x",
    #     "end_y",
    #     "end_x",
    #     "track_id",
    # ],
    """
    Generate overrides for the CytoDLModel configuration
    for evaluating model `model_name` on crops of
    images from dataset `dataset_name`.
    """
    overrides = {
        "train": False,  # True
        "test": False,
        # train and val dataloaders are unnecessary for prediction
        # and might be slow to instantiate (e.g. if they cache data)
        # "data.train_dataloaders": None,
        # "data.val_dataloaders": None,
        # ##################################
        # # "data.predict_dataloaders._target_": "monai.data.DataLoader",
        # "data.predict_dataloaders._target_": "cyto_dl.datamodules.array.make_array_dataloader",
        # "data.predict_dataloaders.data": data,
        # # "data.predict_dataloaders.data": None,
        # # "data.predict_dataloaders.dataset._target_": "monai.data.Dataset",
        # "data.predict_dataloaders.source_key": "raw_bf",
        # "data.predict_dataloaders.transforms": transforms,
        # "data.predict_dataloaders.num_workers": num_workers,
        # "data.predict_dataloaders.batch_size": batch_size,
        # # "data.predict_dataloaders.dataset._target_": "cyto_dl.data.datasets.ArrayDataset",
        # # "data.predict_dataloaders.dataset._target_": "cyto_dl.datamodules.array.make_array_dataloader",
        # # "data.predict_dataloaders.dataset.csv_path": data_path,
        # ##################################
        # "paths.output_dir": save_path,
        "data._target_": "cyto_dl.datamodules.array.make_array_dataloader",
        "data.data": data,
        # "data.predict_dataloaders.data": None,
        # "data.predict_dataloaders.dataset._target_": "monai.data.Dataset",
        "data.source_key": "raw_bf",
        "data.transforms": transforms,
        "data.num_workers": num_workers,
        "data.batch_size": batch_size,
        # change checkpoint path to the one downloaded from mlflow
        "checkpoint.ckpt_path": ckpt_path,
        "checkpoint.strict": True,
        "checkpoint.weights_only": None,  # maybe?
        "callbacks": None,
        # "paths": None,
        "paths.root_dir": r"${oc.env:PROJECT_ROOT, './'}",
        "paths.data_dir": r"${paths.root_dir}/data/",
        "paths.log_dir": r"${paths.root_dir}/logs/",
        "paths.output_dir": "./",
        # "callbacks.prediction_saver": {
        #     "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
        #     "save_dir": save_path,
        #     "meta_keys": [
        #         "T",
        #         "start_y",
        #         "start_x",
        #         "filename_or_obj",
        #     ],
        #     "save_suffix": prediction_filename_suffix or f"{dataset_name}_{model_name}_features",
        # },
        "extras.print_config": False,
        # no spatial inferer needed
        "model.spatial_inferer": None,
        "model.diffusion_key": None,  # diffusion image is not needed
        "model.save_dir": r"${paths.output_dir}",
        # "trainer": None,
        "trainer.max_epochs": 1,  # just one epoch for prediction
        "trainer.accelerator": "auto",  # use CPU for prediction
        # "trainer.devices": 1,  # use one device for prediction
        "trainer.devices": "auto",  # use one device for prediction
        "trainer.default_root_dir": r"${paths.output_dir}",
        "extras.enforce_tags": False,
        "persist_cache": True,
    }
    overrides.update(user_overrides)
    return overrides


def apply_model_on_array2(
    # model_config: CytoDLModelConfig,
    # dataset_config: DatasetConfig,
    data: np.ndarray | list[np.ndarray],
    model_name: str = "diffae_04_10",
    # save_path: str | Path | None = None,
    # upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
) -> None:  # np.ndarray:

    from omegaconf import ListConfig, OmegaConf

    # load model config
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))

    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    overrides = load_overrides(user_overrides)
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    # if save_path is None:
    #     # if no save path is provided, use the default path
    #     save_path = get_output_path(
    #         "models", model_config.name, dataset_config.name, include_timestamp=False
    #     )
    # elif isinstance(save_path, str):
    #     save_path = Path(save_path)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    transforms = model.cfg["data"]["predict_dataloaders"]["dataset"]["transform"]

    # data_path = preprocess_tracking_manifest_for_model_eval(dataset_config, save_path)

    # # use timestamp to get unique file name for FMS upload later
    # timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    # prediction_filename_suffix = f"{dataset_config.name}_{model_config.name}_tracked_crop_features"
    # prediction_filename_suffix = f"{prediction_filename_suffix}_{timestamp}"
    # apply overrides
    # overrides = generate_overrides_for_track_based_crops(
    #     overrides,
    #     save_path=str(save_path),
    #     data_path=str(data_path),
    #     ckpt_path=path_dict["checkpoint_path"],
    #     dataset_name=dataset_config.name,
    #     model_name=model_config.name,
    #     prediction_filename_suffix=prediction_filename_suffix,
    # )

    overrides = generate_overrides_for_array_inputs(
        overrides,
        # save_path=str(save_path),
        # data_path=str(save_path),
        ckpt_path=path_dict["checkpoint_path"].as_posix(),
        data=data,
        transforms=transforms,
        # dataset_name=model_config.dataset_name,
        # model_name=model_config.name,
        # prediction_filename_suffix=None,  # not needed for array inputs
    )

    model.override_config(overrides)
    del model.cfg["ckpt_path"]
    del model.cfg["data"]["predict_dataloaders"]["dataset"]
    del model.cfg["data"]["predict_dataloaders"]["transform"]["transforms"][:2]
    local_config_save_path = get_output_path("models", "evaluation_configs")
    model.save_config(local_config_save_path / f"{model_config.name}_eval.yaml")

    model.predict(data=data)

    # return


def apply_model_on_array(
    # dataset_name: str,
    image_array: np.ndarray,
    # model_name: str = "diffae_04_10_eval_on_array",
    # model_name: str = "diffae_04_10_eval",
    model_name: str = "diffae_04_10",
) -> np.ndarray:

    model_config = load_model_config(model_name)

    # download model from mlflow
    mlflow_id = cast(CytoDLModelConfig, load_model_config(model_name)).mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)

    path_dict = download_model(mlflow_id, model_path)

    # create zarr dataset
    dataset_config = load_dataset_config("20241120_20X")
    resolution_level = 1
    save_path = get_output_path(Path(__file__).stem, include_timestamp=False)
    # data_path = generate_zarr_csv_for_model_eval(dataset_config, save_path, resolution_level)

    overrides = load_overrides(None)  # (overrides)
    # apply overrides
    overrides = generate_overrides_for_model_eval(
        overrides,
        save_path=str(save_path),
        data_path=str(save_path),
        # data_path=str(data_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=dataset_config.name,
        model_name=model_config.name,
    )
    # overrides = {
    #     'data.train_dataloaders': None,
    #     'data.val_dataloaders': None,
    # }

    # load model
    model = CytoDLModel()
    # cfg_path = Path(path_dict["config_path"]).parent / "eval_test.yaml"
    cfg_path = Path(path_dict["config_path"])
    # cfg_path = Path(__file__).parent.resolve() / "diffae_04_10_eval_on_array.yaml"
    model.load_config_from_file(cfg_path.as_posix())

    model.override_config(overrides)

    # model.override_config(overrides)

    # make prediction
    # output is a list with the form
    # [(features_image1, metadata_image1), (features_image2, metadata_image2), ...]
    _, _, cytodl_output = model.predict(data=image_array)

    return cytodl_output


def apply_model_on_array_test1() -> np.ndarray:

    ## example:
    from matplotlib import pyplot as plt

    from src.endo_pipeline.library.process.get_images import get_zarr_img_for_dataset

    dataset_name = "20241120_20X"
    model_name = "diffae_04_10"
    img = get_zarr_img_for_dataset(dataset_name, 0, resolution_level=1)
    dim_order = "TCZYX"

    # img_arr = img.get_image_dask_data(dim_order, T=0).max(dim_order.index("Z"), keepdims=True).compute()
    # img_arr_crop_cdh5 = img_arr[0, 0:1, 0, 0:128, 0:128]  # Example crop
    # img_arr_crop_bf = img_arr[0, 1:2, 0, 0:128, 0:128]  # Example crop
    # data = {"test": img_arr_crop_bf, "val": img_arr_crop_bf, "train": img_arr_crop_bf}
    # apply_model_on_array(data)
    # apply_model_on_crop(img_arr_crop_bf)

    img_arr = img.get_image_dask_data(dim_order, T=0)
    img_arr_crop_cdh5 = img_arr.max(dim_order.index("Z"), keepdims=True)
    img_arr_crop_bf = img_arr.std(dim_order.index("Z"), keepdims=True)

    crop_ex = (slice(None), slice(0, 128), slice(0, 128))  # Example crop
    img_arr_crop_cdh5 = img_arr_crop_cdh5[(0, 0, *crop_ex)].compute()
    img_arr_crop_bf = img_arr_crop_bf[(0, 1, *crop_ex)].compute()

    data = img_arr_crop_bf

    # load model config
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))

    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    # if save_path is None:
    #     # if no save path is provided, use the default path
    #     save_path = get_output_path(
    #         "models", model_config.name, dataset_config.name, include_timestamp=False
    #     )
    # elif isinstance(save_path, str):
    #     save_path = Path(save_path)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    transforms = model.cfg["data"]["predict_dataloaders"]["dataset"]["transform"]
    transforms["transforms"] = transforms["transforms"][2:]  # remove first two transforms

    overrides = load_overrides(None)
    overrides = generate_overrides_for_array_inputs(
        overrides,
        ckpt_path=path_dict["checkpoint_path"].as_posix(),
        data=data.tolist(),  # None,
        transforms=transforms,
    )

    model.override_config(overrides)
    del model.cfg["ckpt_path"]
    # del model.cfg["data"]["predict_dataloaders"]["data"]
    del model.cfg["data"]["train_dataloaders"]
    del model.cfg["data"]["val_dataloaders"]
    del model.cfg["data"]["predict_dataloaders"]  # ["dataset"]
    # del model.cfg["data"]["predict_dataloaders"]["transform"]
    # del model.cfg["data"]["predict_dataloaders"]["transforms"]["transforms"][:2]
    local_config_save_path = get_output_path("models", "evaluation_configs")
    model.save_config(local_config_save_path / f"{model_config.name}_eval.yaml")

    data = img_arr_crop_bf
    # data = {"train": None, "val": None, "test": img_arr_crop_bf}  # , "raw_cdh5": img_arr_crop_cdh5}
    # test = model.predict(data=data)  # , run_async=False)
    test = model.predict(data=img_arr_crop_bf)  # , run_async=False)
    # test = model.predict(data={"raw_bf": img_arr_crop_bf})  # , run_async=False)
    # test = model.predict()  # , run_async=False)

    return test


def apply_model_on_array_test2() -> np.ndarray:
    from cyto_dl.datamodules.array import make_array_dataloader
    from matplotlib import pyplot as plt

    from src.endo_pipeline.library.process.get_images import get_zarr_img_for_dataset

    dataset_name = "20241120_20X"
    model_name = "diffae_04_10"
    img = get_zarr_img_for_dataset(dataset_name, 0, resolution_level=1)
    dim_order = "TCZYX"

    # img_arr = img.get_image_dask_data(dim_order, T=0).max(dim_order.index("Z"), keepdims=True).compute()
    # img_arr_crop_cdh5 = img_arr[0, 0:1, 0, 0:128, 0:128]  # Example crop
    # img_arr_crop_bf = img_arr[0, 1:2, 0, 0:128, 0:128]  # Example crop
    # data = {"test": img_arr_crop_bf, "val": img_arr_crop_bf, "train": img_arr_crop_bf}
    # apply_model_on_array(data)
    # apply_model_on_crop(img_arr_crop_bf)

    img_arr = img.get_image_dask_data(dim_order, T=0)
    img_arr_crop_cdh5 = img_arr.max(dim_order.index("Z"), keepdims=True)
    img_arr_crop_bf = img_arr.std(dim_order.index("Z"), keepdims=True)

    crop_ex = (slice(None), slice(0, 128), slice(0, 128))  # Example crop
    img_arr_crop_cdh5 = img_arr_crop_cdh5[(0, 0, *crop_ex)].compute()
    img_arr_crop_bf = img_arr_crop_bf[(0, 1, *crop_ex)].compute()

    data = img_arr_crop_bf

    # load model config
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))

    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please run on a GPU machine.")
    # download model from mlflow
    mlflow_id = model_config.mlflow_run_id
    model_path = get_output_path("models", model_config.name, include_timestamp=False)
    path_dict = download_model(mlflow_id, model_path)

    # TODO
    # TRY PASSING THE make_array_dataloader FUNCTION DIRECTLY
    # TO model.predict LIKE SO:
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])

    transforms = model.cfg["data"]["predict_dataloaders"]["dataset"]["transform"]
    transforms["transforms"] = transforms["transforms"][2:]  # remove first two transforms
    array_dataloader = make_array_dataloader(
        data=img_arr_crop_bf, transforms=transforms, source_key="raw_bf"
    )

    overrides = load_overrides(None)
    overrides = generate_overrides_for_array_inputs(
        overrides,
        ckpt_path=path_dict["checkpoint_path"].as_posix(),
        data=data.tolist(),  # None,
        transforms=transforms,
    )

    model.override_config(overrides)
    del model.cfg["ckpt_path"]
    del model.cfg["data"]
    local_config_save_path = get_output_path("models", "evaluation_configs")
    model.save_config(local_config_save_path / f"{model_config.name}_eval.yaml")
    test = model.predict(data=array_dataloader)

    return test

    # # from monai.transforms import Compose
    # # from cyto_dl.image.transforms.clip import Clipd
    # # from monai.transforms import NormalizeIntensityd
    # # from monai.transforms import ToTensord
    # # # # data = OmegaConf.create(img_arr_crop_bf)
    # # transforms = Compose([Clipd, NormalizeIntensityd, ToTensord])
    # # make_array_dataloader(img_arr_crop_bf, transforms=np.clip, source_key="raw_bf")

    # from omegaconf import ListConfig, OmegaConf

    # # OmegaConf.to_object(img_arr_crop_bf)
    # # from cyto_dl.eval import evaluate
    # # evaluate(model.cfg, data=img_arr_crop_bf)
    # # from lightning import Trainer

    # from monai.data import DataLoader, Dataset

    # Dataset({"raw_bf": img_arr_crop_bf})  # , "raw_cdh5": img_arr_crop_cdh5})

    # return test
