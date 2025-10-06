import logging
import typing
from pathlib import Path
from typing import Any

import pandas as pd

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel
    from omegaconf import DictConfig, ListConfig

from endo_pipeline.configs import (
    DatasetConfig,
    get_position_integer_from_zarr_file_path,
    get_position_string_from_zarr_file_path,
)
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    load_model,
    upload_file_to_fms,
)
from endo_pipeline.library.model.image_loading import (
    build_zarr_image_loading_dataframe,
    get_exclude_frames,
    get_z_slice_bounds_per_position,
)
from endo_pipeline.library.model.mlflow_utils import download_mlflow_artifact
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    ModelManifest,
    get_dataframe_location_for_dataset,
    get_model_location_for_run,
    load_dataframe_manifest,
    save_dataframe_manifest,
)

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files

logger = logging.getLogger(__name__)


def load_model_for_inference(
    model_manifest: ModelManifest, run_name: str | None, eval_config: "DictConfig | ListConfig"
) -> "CytoDLModel":
    """
    Load a CytoDLModel for inference from a model manifest, run name, and specified eval config.

    Parameters
    ----------
    model_manifest
        Model manifest to load the model from.
    run_name
        Optional, run name of the specific model to load. Loads the most recent run if None.
    eval_config
        Evaluation configuration to override the loaded model's default configuration.
    """
    # get model location for run_name from model manifest
    run_name_ = list(model_manifest.locations.keys())[-1] if run_name is None else run_name
    model_location = get_model_location_for_run(model_manifest, run_name_)

    # load model from location and override with eval config
    model = load_model(model_location)
    model.override_config(eval_config)

    # make sure model manifest name and run name are in model config
    # as 'experiment_name' and 'run_name' respectively
    # ONLY NEED for legacy purposes, this PR (#745) updates train-diffae
    # to store these values in the model config at training time
    if not hasattr(model.cfg, "experiment_name") or model.cfg.experiment_name is None:
        logger.warning(
            "Model config is missing 'experiment_name', setting it to [ %s ] from model manifest.",
            model_manifest.name,
        )
        model.cfg.experiment_name = model_manifest.name
    elif model.cfg.experiment_name != model_manifest.name:
        logger.warning(
            "Model config 'experiment_name' [ %s ] does not match model manifest name [ %s ]. "
            "Overriding with model manifest name.",
            model.cfg.experiment_name,
            model_manifest.name,
        )
        model.cfg.experiment_name = model_manifest.name

    if not hasattr(model.cfg, "run_name") or model.cfg.run_name is None:
        logger.warning(
            "Model config is missing 'run_name', setting it to [ %s ] from model manifest.",
            run_name_,
        )
        model.cfg.run_name = run_name_
    elif model.cfg.run_name != run_name_:
        logger.warning(
            "Model config 'run_name' [ %s ] does not match specified run name [ %s ]. "
            "Overriding with specified run name.",
            model.cfg.run_name,
            run_name_,
        )
        model.cfg.run_name = run_name_

    return model


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


def generate_overrides_for_model_eval(
    save_path: str,
    data_path: str,
    dataset_name: str,
    model_manifest_name: str,
    run_name: str,
    prediction_filename_suffix: str | None = None,
    cache_rate: float = 1.0,
    num_gpus: int | None = None,
) -> dict:
    """
    Generate overrides for the CytoDLModel configuration for evaluating model
    `run_name` from manifest `model_manifest_name` on images from dataset `dataset_name`.
    """
    if prediction_filename_suffix is None:
        save_suffix = f"{dataset_name}_{model_manifest_name}_{run_name}_features"
    else:
        save_suffix = prediction_filename_suffix
    overrides = {
        # train and val dataloaders are unnecessary for prediction
        # and might be slow to instantiate (e.g. if they cache data)
        "data.train_dataloaders": None,
        "data.val_dataloaders": None,
        "data.predict_dataloaders.dataset.dataframe_path": data_path,
        "data.predict_dataloaders.dataset.cache_rate": cache_rate,
        "paths.output_dir": save_path,
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
            "save_suffix": save_suffix,
        },
        "extras.print_config": False,
    }

    if num_gpus is not None:
        overrides["trainer.accelerator"] = "gpu"
        overrides["trainer.devices"] = num_gpus
        if num_gpus == 1:
            overrides["trainer.strategy"] = "auto"
    else:
        overrides["trainer.accelerator"] = "cpu"
        overrides["trainer.devices"] = 1
        overrides["trainer.strategy"] = "auto"

    return overrides


def generate_overrides_for_track_based_crops(
    save_path: str,
    data_path: str,
    dataset_name: str,
    model_manifest_name: str,
    run_name: str,
    prediction_filename_suffix: str | None = None,
    num_gpus: int | None = None,
) -> dict[str, Any]:
    """
    Generate overrides for the CytoDLModel configuration to evaluate model
    `run_name` from manifest `model_manifest_name` on crops of tracked
    objects from dataset `dataset_name`.
    """
    if prediction_filename_suffix is None:
        save_suffix = f"{dataset_name}_{model_manifest_name}_{run_name}_tracked_crop_features"
    else:
        save_suffix = prediction_filename_suffix

    overrides = generate_overrides_for_model_eval(
        save_path=save_path,
        data_path=data_path,
        dataset_name=dataset_name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        num_gpus=num_gpus,
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
            "save_suffix": save_suffix,
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
    z_slice_bounds_per_position: dict[int, dict[str, int]] | None = None,
    only_include_positions: list[int] | None = None,
    exclude_frames: dict[int, list[int]] | None = None,
) -> Path:
    """Preprocess the manifest for a dataset to prepare it for model prediction."""

    manifest = load_dataframe_manifest("live_merged_seg_features")
    location = get_dataframe_location_for_dataset(manifest, dataset_config.name)
    df = load_dataframe(location)

    # keep only rows that were not filtered out
    df = df[df["is_included"]]

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
    # Add which channel to load and what resolution to load it at
    grouped_df["channel"] = ZARR_BF_CHANNEL
    grouped_df["resolution"] = resolution

    # only run a single timepoint from zarr
    grouped_df["frame_start"] = grouped_df["image_index"]
    grouped_df["frame_stop"] = grouped_df["image_index"]
    grouped_df = grouped_df.rename({"zarr_path": "path", "image_index": "T"}, axis=1)

    # add temporary column with position index for filtering
    grouped_df["position_index"] = grouped_df["path"].apply(
        lambda x: get_position_integer_from_zarr_file_path(x)
    )

    # only load images for specified position indices
    if only_include_positions is not None:
        logger.debug(
            "Filtering Zarr files to only include positions: [ %s ]", only_include_positions
        )

        grouped_df = grouped_df[grouped_df["position_index"].isin(only_include_positions)]

    # add column for excluding frames, if specified
    if exclude_frames is not None:
        # if position has no frames to exclude, set to None
        grouped_df["exclude_frames"] = grouped_df["position_index"].apply(
            lambda x: exclude_frames.get(x, None)
        )

    # if start and stop for loading z slices are specified, add to dataframe
    if z_slice_bounds_per_position is not None:
        # get z info dict for each position index
        # unpack the start, stop, and step values from those dictionaries
        grouped_df["z_start"] = grouped_df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_start", 0)
        )
        grouped_df["z_stop"] = grouped_df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_stop", -1)
        )
        grouped_df["z_step"] = grouped_df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_step", 1)
        )

    # remove temporary column with position index
    grouped_df = grouped_df.drop(columns=["position_index"])

    # save the dataframe to a Parquet file that the DiffAE model will use to load cropped images
    save_path = save_dir / "aggregated_crop_manifest.parquet"
    grouped_df.to_parquet(save_path, index=False)
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
    model_manifest_name: str,
    run_name: str,
    crop_size: list[int],
    prediction_path: Path,
) -> None:
    """
    Update the prediction file with metadata,
    return the path to the updated prediction file.
    """
    # add model and dataset information to prediction file
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_manifest_name"] = model_manifest_name
    pred_df["run_name"] = run_name

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
    dataset_name: str, model_manifest_name: str, run_name: str, prediction_path: Path
) -> None:
    """Update the prediction file with metadata."""
    # add model and dataset information to prediction file
    pred_df = pd.read_parquet(prediction_path)
    pred_df["dataset"] = dataset_name
    pred_df["model_manifest_name"] = model_manifest_name
    pred_df["run_name"] = run_name

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


def evaluate_model_on_grid_of_crops_from_one_dataset(
    model: "CytoDLModel",
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
    z_slice_offsets: tuple[int, int] | None = None,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
    only_include_positions: list[int] | None = None,
    num_gpus: int | None = None,
) -> Path:
    """
    Evaluate a DiffAE model to a single dataset.

    **Z-stack offsets**

    The ``z_slice_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_slice_offsets`` is provided, it limits the number of z-slices to load
    by slicing about a global center (annotated in dataset config). If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    Parameters
    ----------
    model
        Trained model to evaluate.
    dataset_config
        Dataset config object for the dataset of interest.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    frame_start
        First frame to include, if None, include from the start.
    frame_stop
        Last frame to include, if None, include to the end.
    frame_step
        Step size for frame inclusion, if None, include every frame.
    only_include_positions
        List of position indices to include, if None, include all positions.
    num_gpus
        Number of GPUs to use for model prediction.

    Returns
    -------
    :
        Creates dataframe of features extracted from the crops and saves it to the output path.

        If ``upload_to_fms`` is True, uploads the prediction file to FMS and adds the file ID to the
        model config manifest.
    """
    model_manifest_name = model.cfg.experiment_name
    run_name = model.cfg.run_name

    # set default output path
    save_path = get_output_path("models", model_manifest_name, run_name, dataset_config.name)

    logger.debug(
        "Evaluating run [ %s ] from model manifest [ %s ] on dataset [ %s ]",
        run_name,
        model_manifest_name,
        dataset_config.name,
    )

    # get unique name for the parquet file
    file_name = "dataset"
    if z_slice_offsets is not None:
        file_name = f"{file_name}_z_stack_{z_slice_offsets[0]}_{z_slice_offsets[1]}"

    file_name_with_extension = f"{file_name}.parquet"
    dataset_save_path = save_path / file_name_with_extension

    # parse dataset annotations to get z-slice information,
    # positions to include, and frames to exclude
    z_slice_bounds_per_position = get_z_slice_bounds_per_position(dataset_config, z_slice_offsets)
    exclude_frames = get_exclude_frames(dataset_config)

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

    # apply workflow-specific overrides
    prediction_filename_suffix = f"{dataset_config.name}_{model_manifest_name}_{run_name}_features"
    overrides = generate_overrides_for_model_eval(
        save_path=save_path.as_posix(),
        data_path=dataset_save_path.as_posix(),
        dataset_name=dataset_config.name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        prediction_filename_suffix=prediction_filename_suffix,
        num_gpus=num_gpus,
    )
    model.override_config(overrides)
    local_config_save_path = get_output_path(
        "models", "evaluation_configs", model_manifest_name, run_name, "grid_crops"
    )
    model.save_config(local_config_save_path / "eval.yaml")
    logger.info(
        "Evaluation config saved to [ %s ]",
        local_config_save_path / "eval.yaml",
    )
    logger.debug("Starting model prediction...")
    model.predict()
    crop_size = model.cfg.model.spatial_inferer.splitter.patch_size

    prediction_path = save_path / f"predict_{prediction_filename_suffix}.parquet"
    update_prediction_from_crops_with_metadata(
        dataset_name=dataset_config.name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        crop_size=crop_size,
        prediction_path=prediction_path,
    )
    logger.info("Model prediction dataframe saved to [ %s ]", prediction_path)

    return prediction_path


def evaluate_model_on_tracked_crops_from_one_dataset(
    model: "CytoDLModel",
    dataset_config: DatasetConfig,
    save_path: str | Path | None = None,
    z_slice_offsets: tuple[int, int] | None = None,
    only_include_positions: list[int] | None = None,
    num_gpus: int | None = None,
) -> Path:
    """
    Evaluate a DiffAE model on a single dataset with cell segmentation and tracking.

    Parameters
    ----------
    model
        Trained model.
    dataset_config
        Dataset config object for the dataset of interest.
    resolution_level
        Resolution level at which to load images (zarr file format) at.
    save_path
        Path to save the prediction file
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    only_include_positions
        List of position indices to include, if None, include all positions.
    num_gpus
        Number of GPUs to use for model prediction.
    """

    model_manifest_name = model.cfg.experiment_name
    run_name = model.cfg.run_name

    if save_path is None:
        # if no save path is provided, use the default path
        save_path = get_output_path(
            "models", model_manifest_name, run_name, dataset_config.name, include_timestamp=False
        )
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    # parse dataset annotations to get z-slice information,
    # positions to include, and frames to exclude
    z_slice_bounds_per_position = get_z_slice_bounds_per_position(dataset_config, z_slice_offsets)
    exclude_frames = get_exclude_frames(dataset_config)

    data_path = preprocess_tracking_manifest_for_model_eval(
        dataset_config,
        save_path,
        z_slice_bounds_per_position=z_slice_bounds_per_position,
        only_include_positions=only_include_positions,
        exclude_frames=exclude_frames,
    )

    # use timestamp to get unique file name for FMS upload later
    prediction_filename_suffix = f"{dataset_config.name}_{model_manifest_name}_{run_name}"
    prediction_filename_suffix = f"{prediction_filename_suffix}_tracked_crop_features"

    # apply overrides
    overrides = generate_overrides_for_track_based_crops(
        save_path=save_path.as_posix(),
        data_path=data_path.as_posix(),
        dataset_name=dataset_config.name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        prediction_filename_suffix=prediction_filename_suffix,
        num_gpus=num_gpus,
    )
    model.override_config(overrides)
    local_config_save_path = get_output_path(
        "models", "evaluation_configs", model_manifest_name, run_name, "tracked_crops"
    )
    model.save_config(local_config_save_path / "eval.yaml")
    logger.info(
        "Evaluation config saved to [ %s ]",
        local_config_save_path / "eval.yaml",
    )
    model.predict()

    prediction_path = save_path / f"predict_{prediction_filename_suffix}.parquet"
    update_prediction_from_tracks_with_metadata(
        dataset_name=dataset_config.name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        prediction_path=prediction_path,
    )

    logger.info("Model prediction dataframe saved to [ %s ]", prediction_path)
    return prediction_path


def upload_prediction_dataframe_to_fms(
    prediction_path: Path,
    dataset_config: DatasetConfig,
    model_manifest: ModelManifest,
    run_name: str,
    dataframe_manifest_name: str,
    workflow_name: str,
    workflow_parameters: dict[str, Any] | None = None,
) -> None:
    """Upload the prediction dataframe to FMS and update the dataframe manifest."""
    # build FMS annotations
    dataset_annotations = build_fms_annotations(
        dataset_config,
        model_manifest=model_manifest,
        run_name=run_name,
    )

    # upload prediction file to FMS and get file ID
    file_id = upload_file_to_fms(
        prediction_path,
        annotations=dataset_annotations,
        file_type="parquet",
    )

    try:
        # Temporarily set dataframe manifest I/O logger to CRITICAL
        # to suppress error logging if manifest does not exist.
        # We want it to be a warning here instead (see except block below).
        dataframe_manifest_io_logger = logging.getLogger(
            "endo_pipeline.manifests.dataframe_manifest_io"
        )
        original_level = dataframe_manifest_io_logger.level
        dataframe_manifest_io_logger.setLevel(logging.CRITICAL)

        # Attempt to load existing dataframe manifest
        manifest = load_dataframe_manifest(dataframe_manifest_name)
    except FileNotFoundError:
        logger.warning(
            "Dataframe manifest [ %s ] not found, creating a new one.",
            dataframe_manifest_name,
        )
        parameters = {} if workflow_parameters is None else workflow_parameters
        manifest = DataframeManifest(
            name=dataframe_manifest_name, workflow=workflow_name, parameters=parameters
        )
    finally:
        # restore original logging level of dataframe manifest I/O logger
        dataframe_manifest_io_logger.setLevel(original_level)

    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)
    logger.info(
        "Updated dataframe manifest [ %s ] with location for dataset [ %s ]",
        dataframe_manifest_name,
        dataset_config.name,
    )
