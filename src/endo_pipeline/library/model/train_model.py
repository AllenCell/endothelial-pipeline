"""Methods for loading models and preprocessing data for model training."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_available_dataset_names,
    get_subset_of_timepoint_annotations,
)
from endo_pipeline.io import build_fms_annotations, load_dataframe, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    ModelManifest,
    save_dataframe_manifest,
)

logger = logging.getLogger(__name__)


def _upload_zarr_dataframe_to_fms(
    dataframe: pd.DataFrame,
    dataset_type: Literal["training", "validation"],
    resolution_level: int,
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
) -> str:
    """Upload a dataframe containing zarr paths and metadata to FMS.

    Parameters
    ----------
    dataframe
        The dataframe to be uploaded, containing paths to zarr files and other
        metadata.
    dataset_type
        A string indicating whether the dataframe is for training or validation
        data.
    resolution_level
        The resolution level of the zarr files to be used for training.
    dataset_config_list
        A list of DatasetConfig objects for the datasets used in training, used
        to build FMS annotations.
    output_savedir
        The directory where the output dataframe will be saved as an
        intermediate file before uploading to FMS.

    Returns
    -------
    :
        The FMS ID of the uploaded dataframe.

    """
    # save the dataframes to parquet files locally as intermediates
    # use timestamp to ensure unique filenames
    output_filename = f"{dataset_type}_resolution_{resolution_level}.parquet"
    output_path = output_savedir / output_filename
    dataframe.to_parquet(output_path, index=False)
    logger.debug("Saved [ %s ] dataframe to \n %s", dataset_type, output_path)

    # upload dataframes to fms
    logger.debug("Building FMS annotations for training and validation dataframes...")
    fms_annotations = build_fms_annotations(
        dataset_config_list,
        additional_notes=f"Dataframe of images for {dataset_type} set. \
            Resolution level for zarr loading: {resolution_level}",
    )

    logger.debug("Annotations built, uploading to FMS...")
    fmsid = upload_file_to_fms(
        output_path,
        annotations=fms_annotations,
        file_type="parquet",
    )

    logger.info("Uploaded % s dataframe to FMS with ID: [ %s ]", dataset_type, fmsid)

    return fmsid


def build_and_save_dataframe_manifest_for_training(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    resolution_level: int,
    z_slice_offsets: tuple[int, int] | None,
    include_cell_piling: bool,
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
    manifest_name: str,
    workflow_name: str,
) -> None:
    """Build training and validation image loading dataframes, and upload them to FMS.

    This method updates or creates a DataframeManifest with the provided
    parameters and the FMS IDs of the uploaded training and validation
    dataframes. The manifest is then saved for use in downstream steps.

    Parameters
    ----------
    train_dataframe
        The training dataframe containing paths to zarr files and other
        metadata.
    val_dataframe
        The validation dataframe containing paths to zarr files and other
        metadata.
    resolution_level
        The resolution level of the zarr files to be used for training.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    include_cell_piling
        Include cell piling timepoints if True, exclude them if False.
    dataset_config_list
        A list of DatasetConfig objects for the datasets used in training.
    output_savedir
        The directory where the output dataframes will be saved as
        intermediates.
    manifest_name
        The name of the DataframeManifest to be created.
    workflow_name
        The name of the workflow that is creating the dataframe manifest.

    """
    # first, upload the train and val dataframes to FMS
    train_fmsid = _upload_zarr_dataframe_to_fms(
        train_dataframe,
        "training",
        resolution_level,
        dataset_config_list,
        output_savedir,
    )

    val_fmsid = _upload_zarr_dataframe_to_fms(
        val_dataframe,
        "validation",
        resolution_level,
        dataset_config_list,
        output_savedir,
    )

    # create the DataframeManifest object
    # note that this will overwrite any existing manifest with the same name
    # (intended behavior)
    dataframe_manifest = DataframeManifest(
        name=manifest_name,
        workflow=workflow_name,
        parameters={
            "z_slice_offsets": z_slice_offsets,
            "include_cell_piling": include_cell_piling,
        },
        locations={
            "training": DataframeLocation(fmsid=train_fmsid, s3uri=None),
            "validation": DataframeLocation(fmsid=val_fmsid, s3uri=None),
        },
    )

    # save the updated or new manifest
    save_dataframe_manifest(dataframe_manifest)


def get_dataset_names_used_for_training(
    train_dataframe_location: DataframeLocation,
    val_dataframe_location: DataframeLocation,
) -> list[str]:
    """Get the list of dataset names used for model training.

    Parses the provided training and validation dataframes based on the
    input dataset collection to get the list of dataset names that are
    used for training.

    Parameters
    ----------
    train_dataframe_location
        The location of the training dataframe, used to load the dataframe and
        extract dataset names.
    val_dataframe_location
        The location of the validation dataframe, used to load the dataframe and
        extract dataset names.

    Returns
    -------
    :
        A sorted list of dataset names used for training.

    """
    train_df = load_dataframe(train_dataframe_location)
    val_df = load_dataframe(val_dataframe_location)

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
    training_dataset_superset = get_available_dataset_names()

    training_dataset_names = []
    for dataset_name in training_dataset_superset:
        for date in training_dataset_dates:
            if date in dataset_name:
                training_dataset_names.append(dataset_name)

    return sorted(training_dataset_names)


def get_included_frames_for_model(
    dataset_config: DatasetConfig, model_manifest: ModelManifest
) -> dict[int, list[int]]:
    """Get list of frame numbers for model training based on timepoint annotations.

    **Method output**

    The output of this method is a dictionary where the keys are position numbers and
    the values are lists of frame numbers that should be included in model training for
    that position.

    Parameters
    ----------
    dataset_config
        The dataset config object for the dataset being used for training, used
        to access information on timepoint annotations.
    model_manifest
        Model manifest object, used to access information on whether to include
        cell piling timepoints in training.

    Returns
    -------
    :
        A dictionary of timepoints to include on a per-position basis.

    """
    # Default behavior is to remove all annotations except NOT_STEADY_STATE
    annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]

    # If cell piling is included, then exclude that annotation from filter
    if model_manifest.parameters["include_cell_piling"]:
        annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)

    # Get list of timepoint annotations to filter out
    annotations = get_subset_of_timepoint_annotations(annotations_to_ignore=annotations_to_ignore)

    # Get list of timepoints without any of the filtered annotations by position
    return get_all_unannotated_timepoints(dataset_config, annotations=annotations)
