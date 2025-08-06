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
    upload_file_to_fms,
)
from src.endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    save_dataframe_manifest,
)

logger = logging.getLogger(__name__)


def get_model_dir() -> Path:
    """Get the path to `src.endo_pipeline.library.model`."""
    return Path(__file__).resolve().parent


def _generate_overrides_for_model_training(
    model_name: str,
    crop_size: int,
    train_csv_path: Path,
    val_csv_path: Path,
    workflow_testing: bool = False,
) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    Parameters
    ----------
    model_name
        The name of the model to train.
    crop_size
        The number of pixels in each dimension of the image crop to use for training.
        That is, the cropped image will be square with size (crop_size px, crop_size px).
    train_csv_path
        The path to the training dataset CSV file.
    val_csv_path
        The path to the validation dataset CSV file.
    workflow_testing
        Flag to indicate if this script is being run for testing purposes (e.g., code review).
        If True, the max number of epochs will be set to 1 to speed up the run during code review.

    Returns
    -------
    :
        A dictionary of configuration overrides for the DiffAE model training.
    """
    # create output directories if they do not exist
    train_output_path = get_output_path("models", model_name, "train", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "logs", include_timestamp=False)
    _ = get_output_path("models", model_name, "train", "checkpoints", include_timestamp=False)

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.csv_path": train_csv_path.as_posix(),
        "data.predict_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        "data.val_dataloaders.dataset.csv_path": val_csv_path.as_posix(),
        # get repo root directory and current working directory
        "paths.root_dir": Path(__file__).resolve().parents[3],
        "paths.work_dir": os.getcwd(),
        # save outputs to user-specified directory
        "paths.output_dir": (train_output_path / "logs").as_posix(),
        "paths.log_dir": "${paths.output_dir}",
        "callbacks.model_checkpoint.dirpath": (train_output_path / "checkpoints").as_posix(),
        # update run name
        "run_name": model_name,
        # set crop size from input via model.image_shape,
        # the rest are populated by interpolation
        "model.image_shape": [1, crop_size, crop_size],
        # turn off config printing, will get saved locally instead
        "extras.print_config": False,
        "trainer.max_epochs": 1 if workflow_testing else 1000,
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
    train_csv_path: Path,
    val_csv_path: Path,
    workflow_testing: bool = False,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    template_training_config
        The template training configuration to use.
    crop_size
        The pixel size of the image crop to use for training.
        This is the crop size along one dimension, the image will be square
        with size (crop_size, crop_size).
    model_name
        The name of the model to train.
    train_csv_path
        The path to the training dataset CSV file.
    val_csv_path
        The path to the validation dataset CSV file.
    workflow_testing
        Flag to indicate if this script is being run for testing purposes (e.g., code review).

        If True, the training and validation datasets used will be the ones with only one entry each
        as generated by the `generate_diffae_training_csv` script with `workflow_testing=True`.
        Additionally, the model will only be trained for one epoch to speed up the
        run during code review.

    Returns
    -------
    :
        An initialized CytoDLModel for training the DiffAE model.
    """
    # user overrides for training
    overrides = _generate_overrides_for_model_training(
        model_name, crop_size, train_csv_path, val_csv_path, workflow_testing
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
    zarr_resolution: int,
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
) -> tuple[str, str]:
    # save the dataframes to csv files locally as intermediates
    # use timestamp to ensure unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_path = output_savedir / f"{dataset_type}_resolution_{zarr_resolution}_{timestamp}.csv"
    dataframe.to_csv(output_path, index=False)
    logger.debug("Saved % s CSV to \n %s", dataset_type, output_path)
    # upload dataframes to fms
    logger.debug("Building FMS annotations for training and validation CSVs...")
    fms_annotations = build_fms_annotations(
        dataset_config_list,
        additional_notes=f"Dataframe of images for {dataset_type} set. \
            Resolution level for zarr loading: {zarr_resolution}",
    )

    logger.debug("Annotations built, uploading to FMS...")
    fmsid = upload_file_to_fms(
        output_path,
        annotations=fms_annotations,
        file_type="csv",
    )

    logger.info("Uploaded % s CSV to FMS with ID: [ %s ]", dataset_type, fmsid)

    return fmsid


def build_and_save_dataframe_manifest_for_training(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    zarr_resolution: int,
    dataset_config_list: list[DatasetConfig],
    output_savedir: Path,
    workflow_testing: bool = False,
) -> None:
    """
    Upload training and validation dataframes to FMS and save a DataframeManifest
    with the DatasetLocation objects containing the FMS IDs of the uploaded files.
    """
    # first, upload the train and val dataframes to FMS
    train_fmsid = _upload_zarr_dataframe_to_fms(
        train_dataframe,
        "training",
        zarr_resolution,
        dataset_config_list,
        output_savedir,
    )

    val_fmsid = _upload_zarr_dataframe_to_fms(
        val_dataframe,
        "validation",
        zarr_resolution,
        dataset_config_list,
        output_savedir,
    )

    # create the DataframeManifest object
    # note that this will overwrite any existing manifest with the same name
    # (intended behavior)
    manifest_name = f"diffae_training_csv_resolution_{zarr_resolution}"
    if workflow_testing:
        # if workflow_testing is True, append "_test_workflow" to the manifest name
        manifest_name += "_test_workflow"
    dataframe_manifest = DataframeManifest(
        name=manifest_name,
        workflow="generate_diffae_training_csv",
        parameters={"zarr_resolution": zarr_resolution},
        locations={
            "training": DataframeLocation(fmsid=train_fmsid, s3uri=None),
            "validation": DataframeLocation(fmsid=val_fmsid, s3uri=None),
        },
    )

    # save the updated or new manifest
    save_dataframe_manifest(dataframe_manifest)


def get_valid_csv_path_for_training(dataframe_location: DataframeLocation) -> Path | str:
    """
    Get a valid CSV path for training or validation datasets.

    Parameters
    ----------
    dataframe_location: DataframeLocation
        The DataframeLocation object containing either the FMS ID of the CSV file
        or the S3 URI of the CSV file.


    Returns
    -------
    :
        A valid Path object pointing to the CSV file for training or validation sets.
        If the DataframeLocation object has an S3 URI, it will be used. Else, this
        function downloads the CSV file from FMS using the FMS ID and returns the local path.
    """
    if dataframe_location.s3uri is not None:
        # if s3uri is provided, use that for loading
        dataframe_csv_path = dataframe_location.s3uri
    else:
        # get local path from FMS ID
        dataframe_csv_path = get_local_path_from_fmsid(dataframe_location.fmsid)

    return dataframe_csv_path


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
    train_csv_path: Path, val_csv_path: Path, dataset_collection_name: str
) -> list[str]:
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
