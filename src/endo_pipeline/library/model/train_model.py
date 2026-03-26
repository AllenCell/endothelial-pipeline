import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_subset_of_timepoint_annotations,
    load_dataset_collection_config,
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
    """
    Upload training and validation image loading dataframes to FMS.

    Parameters
    ----------
    train_dataframe
        The training dataframe containing paths to zarr files and other metadata.
    val_dataframe
        The validation dataframe containing paths to zarr files and other metadata.
    resolution_level
        The resolution level of the zarr files to be used for training.
    z_slice_offsets
        Lower and upper bounds for z-slicing.
    include_cell_piling
        Include cell piling timepoints if True, exclude them if False.
    dataset_config_list
        A list of DatasetConfig objects for the datasets used in training.
    output_savedir
        The directory where the output dataframes will be saved as intermediates.
    manifest_name
        The name of the DataframeManifest to be created.
    workflow_name
        The name of the workflow that is creating the dataframe manifest.

    Returns
    -------
    :
        Saves a ``DataframeManifest`` with ``DatasetLocation`` objects containing the FMS IDs
        of the uploaded files. The manifest is saved to the default location for
        ``DataframeManifests``: [ src/endo_pipeline/manifests/dataframes/ ].

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
    dataset_collection_name: str,
) -> list[str]:
    """
    Pull list of dataset names used for model training from training
    and validation image loading dataframes.
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

    training_dataset_superset = load_dataset_collection_config(dataset_collection_name)
    training_dataset_names = []
    for dataset_name in training_dataset_superset.datasets:
        for date in training_dataset_dates:
            if date in dataset_name:
                training_dataset_names.append(dataset_name)

    return sorted(training_dataset_names)


def get_included_frames_for_model(
    dataset_config: DatasetConfig, model_manifest: ModelManifest
) -> dict[int, list[int]]:
    """Get list of frames for model training based on timepoint annotations."""

    # Default behavior is to remove all annotations except NOT_STEADY_STATE
    annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]

    # If cell piling is included, then exclude that annotation from filter
    if model_manifest.parameters["include_cell_piling"]:
        annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)

    # Get list of timepoint annotations to filter out
    annotations = get_subset_of_timepoint_annotations(annotations_to_ignore=annotations_to_ignore)

    # Get list of timepoints without any of the filtered annotations by position
    return get_all_unannotated_timepoints(dataset_config, annotations=annotations)
