"""Methods for loading models and preprocessing data for model evaluation (inference)."""

import logging
import typing
from pathlib import Path

import pandas as pd

if typing.TYPE_CHECKING:
    from cyto_dl.api import CytoDLModel
    from omegaconf import DictConfig, ListConfig

from endo_pipeline.configs import (
    DatasetConfig,
    get_position_integer_from_zarr_file_path,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe, load_model
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import (
    ModelManifest,
    get_dataframe_location_for_dataset,
    get_model_location_for_run,
    get_most_recent_run_name,
    get_zarr_location_for_position,
    load_dataframe_manifest,
)
from endo_pipeline.settings import (
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    DIFFAE_ZARR_RESOLUTION_LEVEL,
    NATIVE_ZARR_RESOLUTION_CROP_SIZE,
    ZARR_BRIGHTFIELD_CHANNEL,
)
from endo_pipeline.settings import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import CytoDLLoadDataKeys, CytoDLSaveDataKeys

logger = logging.getLogger(__name__)


def load_model_for_inference(
    model_manifest: ModelManifest, run_name: str | None, eval_config: "DictConfig | ListConfig"
) -> "CytoDLModel":
    """Load a CytoDLModel for inference from a model manifest, run name, and specified eval config.

    Parameters
    ----------
    model_manifest
        Model manifest to load the model from.
    run_name
        Optional, run name of the specific model to load. Loads the most recent run if None.
    eval_config
        Evaluation configuration to override the loaded model's default configuration.

    Returns
    -------
    :
        CytoDLModel loaded from the specified model manifest and run name, with the configuration
        overridden by the specified eval config.

    """
    # get model location for run_name from model manifest
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
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


def add_diffae_model_eval_crop_columns(
    df: pd.DataFrame,
    diffae_resolution_level: int = DIFFAE_ZARR_RESOLUTION_LEVEL,
    crop_size: int = NATIVE_ZARR_RESOLUTION_CROP_SIZE,
) -> pd.DataFrame:
    """Add columns to the dataframe for DiffAE model evaluation crops.

    **Note on image resolution**

    The centroids, image sizes, and crop sizes are for the images loaded at the
    native resolution (i.e. a resolution level of 0). The
    diffae_resolution_level parameter will be used to downsample those values
    prior to being passed along to the DiffAE model for evaluation. The
    diffae_resolution_level parameter will also be passed along to the model and
    used to load the images at the appropriate resolution level. The "start" and
    "end" columns returned by this function will determine the crop locations at
    the same resolution as the centroids.

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
    - bbox_is_in_bounds: boolean indicating if the bounding box is within
        image bounds

    **Crop extraction and downsampling**

    Consider and input dataframe is one that has centroids from an image of size
    2048x2048 at resolution level 0 and ``diffae_resolution_level``=1 and
    ``crop_size``=256. A crop with size 256x256 is taken around the centroid
    coordinates and returned under the ``start_x``, ``start_y``, ``end_x``, and
    ``end_y`` columns of the output dataframe.

    These start and end columns are later downsampled by
    ``diffae_resolution_level``=1 (-> 2**1 = downsample factor of 2), resulting
    in crops of size 128x128. The ``diffae_resolution_level=1``, and downsampled
    ``start_x``, ``start_y``, ``end_x``, and ``end_y`` columns are passed along
    to the DiffAE model for evaluation.

    The DiffAE model loads images at resolution level 1 (therefore a size of
    1024x1024) and extracts each crop according to the ``start_x``, ``start_y``,
    ``end_x``, ``end_y`` columns (therefore each crop has a size of 128x128).

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
    df[Column.SegData.CROP_SIZE] = crop_size

    # convert centroids to bounding box coordinates and add them as columns
    df[Column.SegData.START_X_RES_0] = (
        df[Column.SegData.CENTROID_X] - df[Column.SegData.CROP_SIZE] / 2
    ).astype(int)
    df[Column.SegData.START_Y_RES_0] = (
        df[Column.SegData.CENTROID_Y] - df[Column.SegData.CROP_SIZE] / 2
    ).astype(int)
    df[Column.SegData.END_X_RES_0] = (
        df[Column.SegData.CENTROID_X] + df[Column.SegData.CROP_SIZE] / 2
    ).astype(int)
    df[Column.SegData.END_Y_RES_0] = (
        df[Column.SegData.CENTROID_Y] + df[Column.SegData.CROP_SIZE] / 2
    ).astype(int)

    # add a column indicating if the size of the bounding box does
    # not match the downsampled crop size (because the model expects
    # identically sized square crops)
    # check if bounding boxes fit in image bounds without being clipped
    df[Column.SegDataFilters.IS_VALID_BBOX] = bbox_in_image_bounds(df, diffae_resolution_level)

    # Add column for the resolution level to load images at for DiffAE model:
    # Note from Erin 8/21/25: this has updated now that we have resolution level 1
    # zarr files, removed downsample transform from the model config
    df[Column.SegData.RESOLUTION_FOR_DIFFAE] = diffae_resolution_level

    return df


def preprocess_tracking_manifest_for_model_eval(
    dataset_config: DatasetConfig,
    z_slice_bounds_per_position: dict[int, dict[CytoDLLoadDataKeys, int]] | None = None,
    only_include_positions: list[int] | None = None,
    only_include_frames: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Preprocess a tracking dataframe for a dataset to prepare it for model prediction.

    This function performs the following preprocessing steps:
        1. Loads the tracking dataframe for the specified dataset from the
           manifest.
        2. Selects and computes the necessary columns for model evaluation.
        3. Filters the dataframe to exclude rows with invalid bounding boxes.
        4. Adjusts the crop coordinates to be consistent with the specified
           resolution level.
        5. Groups the dataframe by zarr path, position, and timepoint, and
           converts the start and end coordinates to lists.
        6. Adds columns for which channel to load and what resolution to load it
           at.
        7. Optionally filters the dataframe to only include specified position
           indices and/or frames.
        8. Optionally adds columns for z slice bounds if specified.
        9. Removes the temporary column with position index.

    Parameters
    ----------
    dataset_config
        Dataset configuration for the dataset to preprocess the tracking
        manifest for.
    z_slice_bounds_per_position
        Optional, dictionary specifying the z slice bounds for each position
        index.
    only_include_positions
        Optional, list of position indices to include in the preprocessed
        dataframe.
    only_include_frames
        Optional, dictionary specifying which frames to include for each
        position index.

    Returns
    -------
    :
        Preprocessed dataframe ready for model evaluation.

    """
    manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
    location = get_dataframe_location_for_dataset(manifest, dataset_config.name)
    df = load_dataframe(location, delay=True)

    # define which columns to compute and keep from the dataframe
    columns_to_keep = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.SegData.LABEL,
        Column.SegData.START_X_RES_0,
        Column.SegData.START_Y_RES_0,
        Column.SegData.END_X_RES_0,
        Column.SegData.END_Y_RES_0,
        Column.IMAGE_SIZE_X,
        Column.IMAGE_SIZE_Y,
        Column.SegData.CROP_SIZE,
        Column.SegData.RESOLUTION_FOR_DIFFAE,
    ]

    columns_for_filtering = [
        Column.SegDataFilters.IS_VALID_BBOX,
    ]

    # compute the required and filtering columns
    df = df[columns_to_keep + columns_for_filtering].compute()

    # filter the dataframe in-place to remove clipped bounding boxes
    df = df[df[Column.SegDataFilters.IS_VALID_BBOX]]

    df = df[columns_to_keep]

    # Adjust the crop coordinates to be consistent with the resolution level
    resolution = sequence_to_scalar(df[Column.SegData.RESOLUTION_FOR_DIFFAE])

    # Need to confirm that this is loading at the default resolution level of 1
    # If I understand correctly, gets set by add_diffae_model_eval_crop_columns
    logger.debug("Loading images at resolution level: [ %d ]", resolution)
    columns_to_downsample = [
        Column.SegData.START_X_RES_0,
        Column.SegData.START_Y_RES_0,
        Column.SegData.END_X_RES_0,
        Column.SegData.END_Y_RES_0,
    ]
    for col in columns_to_downsample:
        df[col] = df[col] // (2**resolution)
    # group df by zarr_path and convert start and end coordinates to list
    grouped_df = (
        df.groupby([Column.DATASET, Column.POSITION, Column.TIMEPOINT])
        .agg(
            {
                Column.SegData.START_Y_RES_0: lambda x: list(x),
                Column.SegData.START_X_RES_0: lambda x: list(x),
                Column.SegData.END_Y_RES_0: lambda x: list(x),
                Column.SegData.END_X_RES_0: lambda x: list(x),
                Column.TRACK_ID: lambda x: list(x),
            }
        )
        .reset_index()
    )
    # Add which channel to load and what resolution to load it at
    grouped_df[CytoDLLoadDataKeys.CHANNELS] = ZARR_BRIGHTFIELD_CHANNEL
    grouped_df[CytoDLLoadDataKeys.RESOLUTION] = resolution

    # only run a single timepoint from zarr
    grouped_df[CytoDLLoadDataKeys.TIME_START] = grouped_df[Column.TIMEPOINT]
    grouped_df[CytoDLLoadDataKeys.TIME_END] = grouped_df[Column.TIMEPOINT]
    grouped_df = grouped_df.rename(
        {
            Column.TIMEPOINT: CytoDLLoadDataKeys.TIMEPOINT,
            Column.SegData.START_X_RES_0: CytoDLLoadDataKeys.START_X,
            Column.SegData.START_Y_RES_0: CytoDLLoadDataKeys.START_Y,
            Column.SegData.END_X_RES_0: CytoDLLoadDataKeys.END_X,
            Column.SegData.END_Y_RES_0: CytoDLLoadDataKeys.END_Y,
        },
        axis=1,
    )

    # Get mapping of dataset and position to zarr location. Prefer grabbing
    # local paths first, which is faster to load, if it exists. Otherwise, try
    # to grab the S3 URI.
    unique_datasets = grouped_df[Column.DATASET].unique()
    unique_positions = grouped_df[Column.POSITION].unique()
    zarr_file_locs = {}
    for dataset in unique_datasets:
        dataset_config = load_dataset_config(dataset)
        for position in unique_positions:
            loc = get_zarr_location_for_position(dataset_config, position)
            if loc.path is not None and loc.path.exists():
                zarr_file_locs[(dataset, position)] = loc.path.as_posix()
            elif loc.s3uri is not None:
                zarr_file_locs[(dataset, position)] = loc.s3uri

    # Add file path column based on dataset name and position
    grouped_df[CytoDLLoadDataKeys.FILE_PATH] = grouped_df[[Column.DATASET, Column.POSITION]].apply(
        lambda row: zarr_file_locs[row[Column.DATASET], row[Column.POSITION]], axis=1
    )

    # only load images for specified position indices
    if only_include_positions is not None:
        logger.debug(
            "Filtering Zarr files to only include positions: [ %s ]", only_include_positions
        )

        grouped_df = grouped_df[grouped_df[Column.POSITION].isin(only_include_positions)]

    # add column for excluding frames, if specified
    if only_include_frames is not None:
        # if position has no frames to exclude, set to None
        grouped_df[CytoDLLoadDataKeys.INCLUDE_TIMEPOINTS] = grouped_df[Column.POSITION].apply(
            lambda x: only_include_frames.get(x, None)
        )

    # if start and stop for loading z slices are specified, add to dataframe
    if z_slice_bounds_per_position is not None:
        # get z info dict for each position index
        # unpack the start, stop, and step values from those dictionaries
        grouped_df[CytoDLLoadDataKeys.Z_START] = grouped_df[Column.POSITION].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_START, 0)
        )
        grouped_df[CytoDLLoadDataKeys.Z_END] = grouped_df[Column.POSITION].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_END, -1)
        )
        grouped_df[CytoDLLoadDataKeys.Z_STEP] = grouped_df[Column.POSITION].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_STEP, 1)
        )

    # remove temporary column with dataset name and position index
    grouped_df = grouped_df.drop(columns=[Column.DATASET, Column.POSITION])

    return grouped_df


def bbox_in_image_bounds(
    df: pd.DataFrame, resolution_level: int = DIFFAE_ZARR_RESOLUTION_LEVEL
) -> pd.Series:
    """Indicate if bounding boxes fit in image bounds without being clipped.

    Parameters
    ----------
    df
        Dataframe with columns for image size and bounding box coordinates at
        resolution level 0.
    resolution_level
        Resolution level to check the bounding boxes against. The bounding box
        coordinates will be downsampled according to the resolution level before
        checking if they fit within the image bounds.

    Returns
    -------
    :
        Boolean series indicating if the bounding boxes fit in image bounds without
        being clipped.

    """
    # adjust the image size according to the desired downsample factor
    downsample_factor = 2**resolution_level
    cols_to_downsample = [
        Column.IMAGE_SIZE_X,
        Column.IMAGE_SIZE_Y,
        Column.SegData.START_X_RES_0,
        Column.SegData.START_Y_RES_0,
        Column.SegData.END_X_RES_0,
        Column.SegData.END_Y_RES_0,
        Column.SegData.CROP_SIZE,
    ]
    df_temp = df[cols_to_downsample].copy(deep=True)
    for col in cols_to_downsample:
        df_temp[col] = df[col] // downsample_factor

    # limit start and end of x and y bboxes to be within image size limits
    df_temp[Column.SegData.START_X_RES_0] = df_temp[Column.SegData.START_X_RES_0].transform(
        lambda x: max(0, x)
    )
    df_temp[Column.SegData.START_Y_RES_0] = df_temp[Column.SegData.START_Y_RES_0].transform(
        lambda y: max(0, y)
    )
    df_temp[Column.SegData.END_X_RES_0] = df_temp[
        [Column.SegData.END_X_RES_0, Column.IMAGE_SIZE_X]
    ].min(axis=1)
    df_temp[Column.SegData.END_Y_RES_0] = df_temp[
        [Column.SegData.END_Y_RES_0, Column.IMAGE_SIZE_Y]
    ].min(axis=1)

    # filter the dataframe to exclude anything where the size of
    # the bounding box does not match the downsampled crop size
    # (because the model expects identically sized square crops)
    bbox_size_y = df_temp[Column.SegData.END_Y_RES_0] - df_temp[Column.SegData.START_Y_RES_0]
    bbox_size_x = df_temp[Column.SegData.END_X_RES_0] - df_temp[Column.SegData.START_X_RES_0]
    # ask if both x and y bbox dimensions equal downsampled crop size
    bbox_size_is_correct = (bbox_size_y == df_temp[Column.SegData.CROP_SIZE]) & (
        bbox_size_x == df_temp[Column.SegData.CROP_SIZE]
    )
    return bbox_size_is_correct


def update_prediction_from_crops_with_metadata(
    dataset_name: str,
    model_manifest_name: str,
    run_name: str,
    crop_size: list[int],
    prediction_path: Path,
) -> None:
    """Add metadata columns to the prediction dataframe from grid-based crop inference.

    This function adds metadata columns to the prediction dataframe, including:

        - dataset name
        - model manifest name
        - model run name
        - resolution level
        - crop size
        - position index
        - timepoint

    The metadata column names are defined in the Column enum.

    Parameters
    ----------
    dataset_name
        Name of the dataset the predictions were made on.
    model_manifest_name
        Name of the model manifest the model used for prediction was loaded
        from.
    run_name
        Name of the model run the model used for prediction was loaded from.
    crop_size
        Size of the crops that were used for prediction.
    prediction_path
        Path to the prediction file to update with metadata (used for both
        loading and saving).

    """
    # add model and dataset information to prediction file
    pred_df = pd.read_parquet(prediction_path)
    pred_df[Column.DATASET] = dataset_name
    pred_df[Column.DiffAEData.MODEL_MANIFEST] = model_manifest_name
    pred_df[Column.DiffAEData.MODEL_RUN] = run_name

    # note: the current model loads images at resolution
    # level 0 and downsamples in the transforms.
    pred_df[Column.DiffAEData.RESOLUTION] = 1

    pred_df[Column.DiffAEData.END_Y] = pred_df[Column.DiffAEData.START_Y] + crop_size[0]
    pred_df[Column.DiffAEData.END_X] = pred_df[Column.DiffAEData.START_X] + crop_size[1]
    pred_df[Column.DiffAEData.CROP_SIZE_Y] = crop_size[0]
    pred_df[Column.DiffAEData.CROP_SIZE_X] = crop_size[1]

    pred_df[Column.POSITION] = pred_df[CytoDLSaveDataKeys.FILE_PATH].apply(
        lambda s: get_position_integer_from_zarr_file_path(s)
    )
    pred_df.rename(
        columns={
            CytoDLSaveDataKeys.FILE_PATH: Column.ZARR_PATH,
            CytoDLSaveDataKeys.TIMEPOINT: Column.TIMEPOINT,
        },
        inplace=True,
    )
    pred_df.to_parquet(prediction_path)


def update_prediction_from_tracks_with_metadata(
    dataset_name: str, model_manifest_name: str, run_name: str, prediction_path: Path
) -> None:
    """Add metadata columns to the prediction dataframe from track-based crop inference.

    This function adds metadata columns to the prediction dataframe, including:

        - dataset name
        - model manifest name
        - model run name
        - resolution level
        - crop size
        - position index
        - timepoint

    The metadata column names are defined in the Column enum.

    Parameters
    ----------
    dataset_name
        Name of the dataset the predictions were made on.
    model_manifest_name
        Name of the model manifest the model used for prediction was loaded
        from.
    run_name
        Name of the model run the model used for prediction was loaded from.
    prediction_path
        Path to the prediction file to update with metadata (used for both
        loading and saving).

    """
    # add model and dataset information to prediction file
    pred_df = pd.read_parquet(prediction_path)
    pred_df[Column.DATASET] = dataset_name
    pred_df[Column.DiffAEData.MODEL_MANIFEST] = model_manifest_name
    pred_df[Column.DiffAEData.MODEL_RUN] = run_name

    pred_df[Column.DiffAEData.RESOLUTION] = 1

    crop_size = (
        pred_df[Column.DiffAEData.END_Y].iloc[0] - pred_df[Column.DiffAEData.START_Y].iloc[0],
        pred_df[Column.DiffAEData.END_X].iloc[0] - pred_df[Column.DiffAEData.START_X].iloc[0],
    )
    pred_df[Column.DiffAEData.CROP_SIZE_Y] = crop_size[0]
    pred_df[Column.DiffAEData.CROP_SIZE_X] = crop_size[1]
    pred_df[Column.POSITION] = pred_df[CytoDLSaveDataKeys.FILE_PATH].apply(
        lambda s: get_position_integer_from_zarr_file_path(s)
    )
    pred_df.rename(
        columns={
            CytoDLSaveDataKeys.FILE_PATH: Column.ZARR_PATH,
            CytoDLSaveDataKeys.TIMEPOINT: Column.TIMEPOINT,
        },
        inplace=True,
    )
    pred_df.to_parquet(prediction_path)
