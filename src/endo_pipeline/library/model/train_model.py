"""Methods for loading models and preprocessing data for model training."""

import logging
from pathlib import Path

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_available_dataset_names,
    get_subset_of_timepoint_annotations,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeLocation, ModelManifest

logger = logging.getLogger(__name__)


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
