import datetime
import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from cyto_dl.api import CytoDLModel
from omegaconf import DictConfig, ListConfig

from src.endo_pipeline.configs import DatasetConfig, load_dataset_collection_config
from src.endo_pipeline.io import (
    build_fms_annotations,
    get_local_path_from_fmsid,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from src.endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    save_dataframe_manifest,
)

logger = logging.getLogger(__name__)


def _generate_overrides_for_model_training(
    model_name: str,
    crop_size: int,
    train_dataframe_path: str,
    val_dataframe_path: str,
    max_num_epochs: int = 1000,
    log_every_n_steps: int = 50,
) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    **Workflow testing**

    If the training workflow is being run in testing mode, the model will be trained for
    only one epoch. That is, the ``max_num_epochs`` input will be set to 1, which overrides
    the configuration value of ``trainer.max_epochs`` in the training config. The value
    of ``log_every_n_steps`` will also be set to 1.

    Parameters
    ----------
    model_name
        The name of the model to train.
    crop_size
        The number of pixels in each dimension of the image crop to use for training.
        That is, the cropped image will be square with size (crop_size px, crop_size px).
    train_dataframe_path
        The path to the training dataset (image loading metadata) .parquet file.
    val_dataframe_path
        The path to the validation dataset (image loading metadata) .parquet file.
    max_num_epochs
        The maximum number of epochs to train the model for.
    log_every_n_steps
        The interval at which to log training metrics.

    Returns
    -------
    :
        A dictionary of configuration overrides for the DiffAE model training.
    """
    # create output directories if they do not exist
    training_run_output_path = get_output_path("models", model_name, "train")
    _ = get_output_path("models", model_name, "train", "logs")
    _ = get_output_path("models", model_name, "train", "checkpoints")

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.dataframe_path": train_dataframe_path,
        "data.predict_dataloaders.dataset.dataframe_path": val_dataframe_path,
        "data.val_dataloaders.dataset.dataframe_path": val_dataframe_path,
        # get repo root directory and current working directory
        "paths.root_dir": Path(__file__).resolve().parents[3],
        "paths.work_dir": os.getcwd(),
        # save outputs to user-specified directory
        "paths.output_dir": training_run_output_path / "logs",
        "paths.log_dir": "${paths.output_dir}",
        "callbacks.model_checkpoint.dirpath": training_run_output_path / "checkpoints",
        # update run name
        "run_name": model_name,
        # set crop size from input via model.image_shape,
        # the rest are populated by interpolation
        "model.image_shape": [1, crop_size, crop_size],
        # turn off config printing, will get saved locally instead
        "extras.print_config": False,
        # set the max number of epochs for training and logging interval
        "trainer.max_epochs": max_num_epochs,
        "trainer.log_every_n_steps": log_every_n_steps,
    }
    return overrides


def _generate_overrides_for_finetuning(
    model_name: str,
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    ckpt_path: Path,
) -> dict:
    """
    Generate overrides for finetuning a DiffAE model.

    Parameters
    ----------
    model_name: str
        The name of the model to finetune. This should correspond to a
        directory in `results/models/` and match the model name used during the
        `paired_data_validation` step.
    dataset_pair_type: Literal['live_fixed', '20x_40x']
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.
    train_csv_path: Path
        The path to the training CSV file containing paired data.
    val_csv_path: Path
        The path to the validation CSV file containing paired data.
    ckpt_path: Path
        The path to the DiffAE checkpoint to finetune.
    """
    # create output directories if they do not exist
    save_path = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "checkpoints",
        include_timestamp=False,
    )
    _ = get_output_path(
        "finetune_paired_dataset",
        f"finetune_{model_name}_on_{dataset_pair_type}",
        "logs",
        include_timestamp=False,
    )

    overrides = {
        # point to already projected paired dataset
        "data.train_dataloaders.dataset.csv_path": str(train_csv_path),
        "data.val_dataloaders.dataset.csv_path": str(val_csv_path),
        # load diffae checkpoint to finetune
        "checkpoint.ckpt_path": str(ckpt_path),
        "checkpoint.weights_only": True,
        "checkpoint.strict": False,
        # save to user-specified directory
        "model.save_dir": (save_path / "logs").as_posix(),
        "trainer.default_root_dir": save_path,
        "callbacks.model_checkpoint.dirpath": (save_path / "checkpoints").as_posix(),
        "paths.output_dir": (save_path / "logs").as_posix(),
        # do training
        "train": True,
        # # make sure that last ckpt is saved
        # "callbacks.model_checkpoint.monitor": None,
        # updated mlflow logger
        "logger": {
            "mlflow": {
                "_target_": "cyto_dl.loggers.MLFlowLogger",
                "tracking_uri": "https://production.int.allencell.org/mlflow/",
                "experiment_name": "endo_diffae",
                "run_name": "fixed_finetune_separate_encoder",
            }
        },
    }

    return overrides


def initialize_diffae_model(
    template_training_config: DictConfig | ListConfig,
    crop_size: int,
    model_name: str,
    train_dataframe_path: str,
    val_dataframe_path: str,
    max_num_epochs: int = 1000,
    log_every_n_steps: int = 50,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    **Workflow testing**

    If the training workflow is being run in testing mode, the model will be trained for
    only one epoch. That is, the ``max_num_epochs`` input will be set to 1, which overrides
    the configuration value of ``trainer.max_epochs`` in the training config. The value
    of ``log_every_n_steps`` will also be set to 1.

    Parameters
    ----------
    template_training_config
        The template training configuration to use.
    crop_size
        The pixel size of the square image crop along one dimension to use in training.
    model_name
        The name of the model to train.
    train_dataframe_path
        The path to the training dataset (image loading metadata) .parquet file.
    val_dataframe_path
        The path to the validation dataset (image loading metadata) .parquet file.
    max_num_epochs
        The maximum number of epochs to train the model for.
    log_every_n_steps
        The interval at which to log training metrics.

    Returns
    -------
    :
        An initialized ``CytoDLModel`` for training the DiffAE model.
    """

    if log_every_n_steps > max_num_epochs:
        logger.error("Logging interval must be less than or equal to the maximum number of epochs.")
        raise ValueError(
            "Logging interval must be less than or equal to the maximum number of epochs."
        )

    # user overrides for training
    overrides = _generate_overrides_for_model_training(
        model_name,
        crop_size,
        train_dataframe_path,
        val_dataframe_path,
        max_num_epochs=max_num_epochs,
        log_every_n_steps=log_every_n_steps,
    )

    # init model
    cytodl_model = CytoDLModel()
    # override config with workflow inputs
    cytodl_model.load_config_from_dict(template_training_config)
    cytodl_model.override_config(overrides)
    return cytodl_model


def _upload_zarr_dataframe_to_fms(
    dataframe: pd.DataFrame,
    dataset_type: Literal["training", "validation"],
    resolution_level: int,
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
) -> str:
    # save the dataframes to parquet files locally as intermediates
    # use timestamp to ensure unique filenames
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M")
    output_filename = f"{dataset_type}_resolution_{resolution_level}_{timestamp}.parquet"
    output_path = output_savedir / output_filename
    dataframe.to_parquet(output_path, index=False)
    logger.debug("Saved % s dataframe to \n %s", dataset_type, output_path)

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
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
    manifest_name: str,
) -> None:
    """
    Upload training and validation image loading dataframes to FMS.

    **Workflow testing**

    If the dataframe building workflow is being run in testing mode, the training and validation
    datasets will only keep one entry each. This is useful for testing the workflow without needing
    to load large datasets. The dataframes will be uploaded to the staging environment of FMS,
    and the resulting DataframeManifest will be saved with ``_test_workflow`` appended to the name.

    Parameters
    ----------
    train_dataframe
        The training dataframe containing paths to zarr files and other metadata.
    val_dataframe
        The validation dataframe containing paths to zarr files and other metadata.
    resolution_level
        The resolution level of the zarr files to be used for training.
    dataset_config_list
        A list of DatasetConfig objects for the datasets used in training.
    output_savedir
        The directory where the output dataframes will be saved as intermediates.
    manifest_name
        The name of the DataframeManifest to be created.

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
        workflow="create_diffae_training_dataframe",
        parameters={"resolution_level": resolution_level},
        locations={
            "training": DataframeLocation(fmsid=train_fmsid, s3uri=None),
            "validation": DataframeLocation(fmsid=val_fmsid, s3uri=None),
        },
    )

    # save the updated or new manifest
    save_dataframe_manifest(dataframe_manifest)


def get_valid_dataframe_path_for_training(dataframe_location: DataframeLocation) -> str:
    """
    Get a valid path for training or validation dataframes.

    These are the dataframes that are used for loading zarr files for training or validation.
    They are either stored in FMS or on S3 as a .parquet file, and this function retrieves the path
    to that file based on the input DataframeLocation object.

    Parameters
    ----------
    dataframe_location
        The DataframeLocation object containing either the FMS ID or S3 URI of the .parquet file.


    Returns
    -------
    :
        The path to the .parquet file for training or validation sets, rendered as a string.

        If the DataframeLocation object has an S3 URI, it will be used. Else, this
        function downloads the file from FMS using the FMS ID and returns the local path.
    """
    if dataframe_location.s3uri is not None:
        # if s3uri is provided, use that for loading
        dataframe_path = dataframe_location.s3uri
    else:
        # get local path from FMS ID
        if dataframe_location.fmsid is None:
            logger.error(
                "DataframeLocation does not have a FMS ID or S3 URI. "
                "Please provide a valid DataframeLocation object."
            )
            raise ValueError(
                "DataframeLocation does not have a FMS ID or S3 URI. "
                "Please provide a valid DataframeLocation object."
            )
        dataframe_path = get_local_path_from_fmsid(dataframe_location.fmsid).as_posix()

    return dataframe_path


def initialize_diffae_model_for_finetuning(
    template_finetune_config: DictConfig | ListConfig,
    model_name: str,
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    train_csv_path: Path,
    val_csv_path: Path,
    model_save_path: Path,
    diffae_ckpt_path: Path,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    template_finetune_config
        The template configuration for finetuning the DiffAE model.
    model_name
        The name of the model to train.
    dataset_pair_type
        The type of dataset to use for finetuning ("live_fixed" or "20X_40X").
        This should match the input used during the `paired_data_validation` step.
    train_csv_path
        The path to the training dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    val_csv_path
        The path to the validation dataset CSV file. If None, the default path
        for the output of generate_csv_for_training_diffae will be used.
    model_save_path
        The path to the directory where the checkpoints and logs will be saved.
    diffae_ckpt_path
        The path to the DiffAE checkpoint to finetune. This should be a path
        to the checkpoint downloaded from MLflow artifacts.

    Returns
    -------
    cytodl_model
        An initialized CytoDLModel for finetuning the DiffAE model.
    """
    # generate overrides for train.yaml for finetuning
    overrides = _generate_overrides_for_finetuning(
        model_name=model_name,
        dataset_pair_type=dataset_pair_type,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        ckpt_path=model_save_path / diffae_ckpt_path,
    )

    # init model
    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_dict(template_finetune_config)
    cytodl_model.override_config(overrides)

    return cytodl_model


def get_valid_csv_path_for_finetuning(
    csv_path: Path | str | None,
    csv_name: Literal["train", "val"],
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
) -> Path:
    """
    Get a valid CSV path for training or validation datasets.

    Parameters
    ----------
    csv_path: Path | str | None
        The path to the CSV file. If None, the default path for the output of
        generate_csv_for_training_diffae will be used.
    csv_name: Literal["train", "val"]
        The name of the CSV file to validate. If csv_path is not None,
        csv_name will not be used in the path generation.
        This input is mainly for the default case where csv_path is None,
        and the path will be generated based on the csv_name (train or val).
    dataset_pair_type: Literal["live_fixed", "20x_40x"]
        The type of dataset to use for finetuning. This should match the dataset
        type used during the `paired_data_validation` step.


    Returns
    -------
    Path
        A valid Path object pointing to the CSV file.
    """
    if csv_path is None:
        csv_path = (
            get_output_path("finetune_paired_dataset", dataset_pair_type, include_timestamp=False)
            / f"{csv_name}.csv"
        )

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide a valid path.")

    return csv_path


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
