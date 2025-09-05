import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from cyto_dl.api import CytoDLModel

from endo_pipeline.configs import (
    CytoDLModelConfig,
    DatasetConfig,
    get_position_string_from_zarr_file_path,
)
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.library.model.image_loading import (
    build_zarr_image_loading_dataframe,
    get_exclude_frames,
    get_include_positions,
    get_z_slice_bounds_per_position,
)
from endo_pipeline.library.model.mlflow_utils import download_mlflow_artifact, download_model
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    save_dataframe_manifest,
)

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files

logger = logging.getLogger(__name__)


def get_model_dir() -> Path:
    """Get the path to `endo_pipeline.library.model`."""
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
        logger.error("Overrides must be a dictionary or a path to a .json file.")
        raise ValueError("Overrides must be a dictionary or a path to a .json file.")
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
        "data.predict_dataloaders.dataset.dataframe_path": data_path,
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


def add_diffae_model_eval_crop_columns(
    df: pd.DataFrame, diffae_resolution_level: int = 1, crop_size: int = 256
) -> pd.DataFrame:
    """
    Add columns to the dataframe for DiffAE model evaluation crops.

    **Note on image resolution**

    The centroids, image sizes, and crop sizes are for the images loaded at the native resolution
    (i.e. a resolution level of 0). The diffae_resolution_level parameter will be used to
    downsample those values prior to being passed along to the DiffAE model for evaluation.
    The diffae_resolution_level parameter will also be passed along to the model and used to load
    the images at the appropriate resolution level. The "start" and "end" columns returned by
    this function will determine the crop locations at the same resolution as the centroids.

    **Input dataframe**

    The input dataframe requires the following columns:
        - centroid_X: x-coordinate of the centroid
        - centroid_Y: y-coordinate of the centroid
        - image_size_x: width of the image
        - image_size_y: height of the image

    **Output dataframe**

    The output dataframe has the following additional columns:
        - start_x: x-coordinate of the top-left corner of the crop
        - start_y: y-coordinate of the top-left corner of the crop
        - end_x: x-coordinate of the bottom-right corner of the crop
        - end_y: y-coordinate of the bottom-right corner of the crop
        - bbox_is_in_bounds: boolean indicating if the bounding box is within image bounds

    **Crop extraction and downsampling**

    Consider and input dataframe is one that has centroids from an image of size
    2048x2048 at resolution level 0 and ``diffae_resolution_level``=1 and ``crop_size``=256.
    A crop with size 256x256 is taken around the centroid coordinates and returned under the
    ``start_x``, ``start_y``, ``end_x``, and ``end_y`` columns of the output dataframe.

    These start and end columns are later downsampled by ``diffae_resolution_level``=1
    (-> 2**1 = downsample factor of 2), resulting in crops of size 128x128.
    The ``diffae_resolution_level=1``, and downsampled ``start_x``, ``start_y``, ``end_x``,
    and ``end_y`` columns are passed along to the DiffAE model for evaluation.

    The DiffAE model loads images at resolution level 1 (therefore a size of 1024x1024)
    and extracts each crop according to the ``start_x``, ``start_y``, ``end_x``, ``end_y``
    columns (therefore each crop has a size of 128x128).

    Parameters
    ----------
    df
        Dataframe to operate on.
    diffae_resolution_level
        Level of binning to use when loading the images
    crop_size
        Size of the square crop to extract around each centroid

    Returns
    -------
    :
        Dataframe with additional columns for DiffAE model evaluation crops.
    """
    # add the size of the crop used to get DiffAE features at full res
    df["crop_size"] = crop_size

    # convert centroids to bounding box coordinates and add them as columns
    df["start_x"] = (df["centroid_X"] - df["crop_size"] / 2).astype(int)
    df["start_y"] = (df["centroid_Y"] - df["crop_size"] / 2).astype(int)
    df["end_x"] = (df["centroid_X"] + df["crop_size"] / 2).astype(int)
    df["end_y"] = (df["centroid_Y"] + df["crop_size"] / 2).astype(int)

    # add a column indicating if the size of the bounding box does
    # not match the downsampled crop size (because the model expects
    # identically sized square crops)
    # check if bounding boxes fit in image bounds without being clipped
    df["bbox_is_in_bounds"] = bbox_in_image_bounds(df, diffae_resolution_level)

    # Add column for the resolution level to load images at for DiffAE model:
    # Note from Erin 8/21/25: this has updated now that we have resolution level 1
    # zarr files, removed downsample transform from the model config
    df["diffae_resolution_level_to_use"] = diffae_resolution_level

    return df


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig,
    save_dir: Path,
) -> Path:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""

    manifest = load_dataframe_manifest("live_merged_seg_features")
    location = get_dataframe_location_for_dataset(manifest, dataset_config.name)
    df = load_dataframe(location)

    # keep only rows that were not filtered out by filter_global
    df = df[~df["filter_global"]]

    # filter the dataframe in-place to remove clipped bounding boxes
    df = df[df["bbox_is_in_bounds"]]

    # filter the dataframe to include only the relevant columns
    columns_to_keep = [
        "zarr_path",
        "image_index",
        "track_id",
        "label",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "image_size_x",
        "image_size_y",
        "crop_size",
        "diffae_resolution_level_to_use",
    ]
    df = df[columns_to_keep]

    # Adjust the crop coordinates to be consistent with the resolution level
    resolution = sequence_to_scalar(df["diffae_resolution_level_to_use"])
    columns_to_downsample = ["start_x", "start_y", "end_x", "end_y"]
    for col in columns_to_downsample:
        df[col] = df[col] // (2**resolution)
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
    # Add which channel to evaluate to the model and what resolution to load it at
    grouped_df["channel"] = ZARR_BF_CHANNEL
    grouped_df["resolution"] = resolution

    # only run a single timepoint from zarr
    grouped_df["start"] = grouped_df["image_index"]
    grouped_df["stop"] = grouped_df["image_index"]
    grouped_df = grouped_df.rename({"zarr_path": "path", "image_index": "T"}, axis=1)

    # save the dataframe to a CSV file that the DiffAE model will use to load cropped images
    save_path = save_dir / "aggregated_crop_manifest.csv"
    grouped_df.to_csv(save_path, index=False)
    return save_path


def bbox_in_image_bounds(df: pd.DataFrame, resolution_level: int = 1) -> pd.Series:
    """Indicate if bounding boxes fit in image bounds without being clipped."""
    # adjust the image size according to the desired downsample factor
    downsample_factor = 2**resolution_level
    cols_to_downsample = [
        "image_size_x",
        "image_size_y",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "crop_size",
    ]
    df_temp = df[cols_to_downsample].copy(deep=True)
    for col in cols_to_downsample:
        df_temp[col] = df[col] // downsample_factor

    # limit start and end of x and y bboxes to be within image size limits
    df_temp["start_x"] = df_temp["start_x"].transform(lambda x: max(0, x))
    df_temp["start_y"] = df_temp["start_y"].transform(lambda y: max(0, y))
    df_temp["end_x"] = df_temp[["end_x", "image_size_x"]].min(axis=1)
    df_temp["end_y"] = df_temp[["end_y", "image_size_y"]].min(axis=1)

    # filter the dataframe to exclude anything where the size of
    # the bounding box does not match the downsampled crop size
    # (because the model expects identically sized square crops)
    bbox_size_y = df_temp.end_y - df_temp.start_y
    bbox_size_x = df_temp.end_x - df_temp.start_x
    # ask if both x and y bbox dimensions equal downsampled crop size
    bbox_size_is_correct = (bbox_size_y == df_temp["crop_size"]) & (
        bbox_size_x == df_temp["crop_size"]
    )
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


def apply_model_on_grid_of_crops_from_one_dataset(
    model_config: CytoDLModelConfig,
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
    testing_mode: bool = False,
) -> None:
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
    slice_by_global_center
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
    elif torch.cuda.device_count() < 1:
        logger.error(
            "CUDA available, but no GPU devices found. "
            "Please set `CUDA_VISIBLE_DEVICES` to a valid GPU device "
            "or run workflow with GPU setup enabled (-g flag)."
        )
        raise RuntimeError("CUDA available, but no GPU devices found.")

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

    file_name = f"{file_name}_{timestamp}.parquet"
    dataset_save_path = save_path / file_name

    # default frame start and stop values are None, i.e., load all timepoints
    frame_start = None
    frame_stop = None
    frame_step = None

    # parse dataset annotations to get z-slice information,
    # positions to include, and frames to exclude
    z_slice_bounds_per_position = get_z_slice_bounds_per_position(
        dataset_config, z_stack_offsets, slice_by_global_center
    )
    only_include_positions = get_include_positions(dataset_config)
    exclude_frames = get_exclude_frames(dataset_config)

    if testing_mode:
        # for workflow testing, only use first position from each dataset
        # and first two timepoints to speed up the dataloading process
        # (if dataset is not timelapse, then only one timepoint is used)
        frame_start = 0
        frame_stop = 1 if dataset_config.is_timelapse else 0
        only_include_positions = only_include_positions[0:1]
        logger.debug(
            "Workflow testing is enabled, only processing the first few timepoints "
            "of the first valid position the dataset."
        )

    if z_stack_offsets is not None:
        # load timepoints 0, 250, and 500 for z-stack offsets summary
        frame_start = 0
        frame_stop = -1
        frame_step = 250
        logger.debug("Z-stack offsets provided, getting features only for frames 0, 250, and 500.")

    # build dataframe with zarr loading metadata
    df = build_zarr_image_loading_dataframe(
        dataset_config,
        resolution_level=resolution_level,
        channel=dataset_config.zarr_channel_indices.brightfield,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
        z_slice_bounds_per_position=z_slice_bounds_per_position,
        only_include_positions=only_include_positions,
        exclude_frames=exclude_frames,
    )

    # save the dataframe to a parquet file
    df.to_parquet(dataset_save_path, index=False)

    # apply overrides
    prediction_filename_suffix = f"{dataset_config.name}_{model_config.name}_features_{timestamp}"
    # having issues with zarr loading when using z-slices from global center,
    # need to decrease the num_workers
    num_workers = 64 if (z_stack_offsets is not None and slice_by_global_center) else 128
    logger.debug("Using [ %d ] workers for data loading.", num_workers)
    overrides = generate_overrides_for_model_eval(
        load_overrides(user_overrides),
        save_path=save_path.as_posix(),
        data_path=dataset_save_path.as_posix(),
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
        save_path=save_path.as_posix(),
        data_path=data_path.as_posix(),
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
