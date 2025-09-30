import logging
import os
import typing
from pathlib import Path
from typing import Literal

import pandas as pd
from cyto_dl.api import CytoDLModel

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig

from endo_pipeline.configs import DatasetConfig, load_dataset_collection_config
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.manifests import DataframeLocation, DataframeManifest, save_dataframe_manifest

logger = logging.getLogger(__name__)


def _generate_overrides_for_model_training(
    model_manifest_name: str,
    run_name: str,
    crop_size: int,
    train_dataframe_path: str,
    val_dataframe_path: str,
    max_num_epochs: int = 1000,
    log_every_n_steps: int = 50,
    cache_rate: float = 1.0,
    replace_rate: float = 0.1,
    num_gpus: int | None = None,
) -> dict:
    """
    Generate overrides for the DiffAE model training configuration.

    Parameters
    ----------
    model_manifest_name
        The name of the model manifest to which this training session belongs.
    run_name
        The MLFlow run name for the training session.
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
    cache_rate
        The fraction of the dataset to cache in memory for training.
    replace_rate
        The replace rate for cached data.
    num_gpus
        Number of GPUs to use with the workflow.

    Returns
    -------
    :
        A dictionary of configuration overrides for the DiffAE model training.
    """
    # create output directories if they do not exist
    training_run_checkpoint_path = get_output_path(
        "models", model_manifest_name, run_name, "train", "checkpoints"
    )
    training_run_log_path = get_output_path(
        "models", model_manifest_name, run_name, "train", "logs"
    )

    # Calculate effective epochs
    multiplier = (1 - cache_rate) / (cache_rate * replace_rate) + 1
    effective_min_epochs = int(2500 * multiplier)
    effective_max_epochs = int(max_num_epochs * multiplier)
    effective_save_images_epochs = int(10 * multiplier)

    overrides = {
        # set path to train and val datasets
        "data.train_dataloaders.dataset.dataframe_path": train_dataframe_path,
        "data.train_dataloaders.dataset.cache_rate": cache_rate,
        "data.train_dataloaders.dataset.replace_rate": replace_rate,
        "data.predict_dataloaders.dataset.dataframe_path": val_dataframe_path,
        "data.val_dataloaders.dataset.dataframe_path": val_dataframe_path,
        "data.val_dataloaders.dataset.cache_rate": cache_rate,
        "data.val_dataloaders.dataset.replace_rate": replace_rate,
        # get repo root directory and current working directory
        "paths.root_dir": Path(__file__).resolve().parents[3].as_posix(),
        "paths.work_dir": os.getcwd(),
        # save outputs to user-specified directory
        "paths.output_dir": training_run_log_path.as_posix(),
        "paths.log_dir": "${paths.output_dir}",
        "callbacks.model_checkpoint.dirpath": training_run_checkpoint_path.as_posix(),
        # update run name
        "run_name": run_name,
        # set crop size from input via model.image_shape,
        # the rest are populated by interpolation
        "model.image_shape": [1, crop_size, crop_size],
        # override the effective epochs calculations
        "model.save_images_every_n_epochs": effective_save_images_epochs,
        "trainer.min_epochs": effective_min_epochs,
        "trainer.max_epochs": effective_max_epochs,
        # turn off config printing, will get saved locally instead
        "extras.print_config": False,
        # set logging interval
        "trainer.log_every_n_steps": log_every_n_steps,
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


def _generate_overrides_for_finetuning(
    finetuned_model_manifest_name: str,
    finetuned_run_name: str,
    train_dataframe_path: str,
    val_dataframe_path: str,
    max_num_epochs: int = 100,
    log_every_n_steps: int = 50,
    cache_rate: float = 1.0,
    replace_rate: float = 0.1,
    num_gpus: int | None = None,
) -> dict:
    """
    Generate overrides for finetuning a DiffAE model.

    Parameters
    ----------
    finetuned_model_manifest_name
        The name of the finetuned model manifest to which this training session belongs.
    finetuned_run_name
        The MLFlow run name for the finetuned model training session.
    train_dataframe_path
        The path to the image loading metadata file for the training dataset.
    val_dataframe_path
        The path to the image loading metadata file for the validation dataset.
    max_num_epochs
        The maximum number of epochs to train the model for.
    log_every_n_steps
        The interval at which to log training metrics.
    cache_rate
        The fraction of the dataset to cache in memory for training.
    replace_rate
        The replace rate for cached data.
    num_gpus
        Number of GPUs to use with the workflow.

    """
    # create output directories if they do not exist
    training_run_output_path = get_output_path(
        "models",
        finetuned_model_manifest_name,
        finetuned_run_name,
    )
    training_run_checkpoint_path = get_output_path(
        "models",
        finetuned_model_manifest_name,
        finetuned_run_name,
        "checkpoints",
    )
    training_run_log_path = get_output_path(
        "models",
        finetuned_model_manifest_name,
        finetuned_run_name,
        "logs",
    )

    # Calculate effective epochs
    multiplier = (1 - cache_rate) / (cache_rate * replace_rate) + 1
    effective_max_epochs = int(max_num_epochs * multiplier)

    # Calculate effective epochs
    multiplier = (1 - cache_rate) / (cache_rate * replace_rate) + 1
    effective_min_epochs = int(1000 * multiplier)
    effective_max_epochs = int(max_num_epochs * multiplier)
    effective_save_images_epochs = int(10 * multiplier)

    overrides = {
        # point to already projected paired dataset
        "data.train_dataloaders.dataset.dataframe_path": train_dataframe_path,
        "data.train_dataloaders.dataset.cache_rate": cache_rate,
        "data.train_dataloaders.dataset.replace_rate": replace_rate,
        "data.predict_dataloaders.dataset.dataframe_path": val_dataframe_path,
        "data.val_dataloaders.dataset.dataframe_path": val_dataframe_path,
        "data.val_dataloaders.dataset.cache_rate": cache_rate,
        "data.val_dataloaders.dataset.replace_rate": replace_rate,
        # Finetuneing-specific overrides
        "checkpoint.weights_only": True,
        "checkpoint.strict": False,
        # save to user-specified directory
        "model.save_dir": training_run_log_path.as_posix(),
        "trainer.default_root_dir": training_run_output_path.as_posix(),
        "callbacks.model_checkpoint.dirpath": training_run_checkpoint_path.as_posix(),
        "paths.output_dir": training_run_log_path.as_posix(),
        # do training
        "train": True,
        # turn off config printing, will get saved locally instead
        "extras.print_config": False,
        # override the effective epochs calculations
        "model.save_images_every_n_epochs": effective_save_images_epochs,
        "trainer.min_epochs": effective_min_epochs,
        "trainer.max_epochs": effective_max_epochs,
        "trainer.log_every_n_steps": log_every_n_steps,
        # updated the run name
        "run_name": finetuned_run_name,
    }

    if num_gpus:
        overrides["trainer.accelerator"] = "gpu"
        overrides["trainer.devices"] = num_gpus
        if num_gpus == 1:
            overrides["trainer.strategy"] = "auto"
    else:
        overrides["trainer.accelerator"] = "cpu"
        overrides["trainer.devices"] = 1
        overrides["trainer.strategy"] = "auto"

    return overrides


def initialize_diffae_model(
    template_training_config: "DictConfig | ListConfig",
    crop_size: int | None = None,
    model_manifest_name: str,
    run_name: str | None = None,
    train_dataframe_path: str | None = None,
    val_dataframe_path: str | None = None,
    max_num_epochs: int | None = None,
    log_every_n_steps: int | None = None,
    cache_rate: float | None = None,
    replace_rate: float | None = None,
    num_gpus: int | None = None,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    template_training_config
        The template training configuration to use.
    crop_size
        The pixel size of the square image crop along one dimension to use in training.
    model_manifest_name
        The name of the model manifest to which this training session belongs.
    run_name
        The MLFlow run name for the training session.
    train_dataframe_path
        The path to the training dataset (image loading metadata) .parquet file.
    val_dataframe_path
        The path to the validation dataset (image loading metadata) .parquet file.
    max_num_epochs
        The maximum number of epochs to train the model for.
    log_every_n_steps
        The interval at which to log training metrics.
    cache_rate
        The fraction of the dataset to cache in memory for training.
    replace_rate
        The replace rate for cached data.
    num_gpus
        Number of GPUs to use with the workflow. If None, use the CPU!

    Returns
    -------
    :
        An initialized ``CytoDLModel`` for training the DiffAE model.
    """

    import os

    def get_config_value(keys_or_path, default=None, type_=None, validator=None):
        # Accepts a single string or tuple list for nested key path (dot notation)
        value = template_training_config
        if isinstance(keys_or_path, str):
            keys = keys_or_path.split(".")
        else:
            keys = keys_or_path
        try:
            for k in keys:
                value = value[k]
        except (KeyError, IndexError, TypeError):
            return default
        if value is None:
            return default
        if type_ is not None and not isinstance(value, type_):
            return default
        if validator is not None and not validator(value):
            return default
        return value

    # Parameter extraction and validation
    crop_size = crop_size or get_config_value("model.image_shape", [1, 64, 64])[1]
    if not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError(f"crop_size must be a positive integer (got {crop_size})")

    model_name = model_name or get_config_value("run_name")
    if not model_name or not isinstance(model_name, str):
        raise ValueError("Could not extract valid model_name from config.")

    train_dataframe_path = train_dataframe_path or get_config_value(
        "data.train_dataloaders.dataset.dataframe_path"
    )
    if not train_dataframe_path or not isinstance(train_dataframe_path, str):
        raise ValueError("Could not extract valid train_dataframe_path from config.")

    val_dataframe_path = val_dataframe_path or get_config_value(
        "data.val_dataloaders.dataset.dataframe_path"
    )
    if not val_dataframe_path or not isinstance(val_dataframe_path, str):
        raise ValueError("Could not extract valid val_dataframe_path from config.")

    max_num_epochs = max_num_epochs or get_config_value("trainer.max_epochs", 1000)
    if not isinstance(max_num_epochs, int) or max_num_epochs <= 0:
        raise ValueError(f"max_num_epochs must be a positive integer (got {max_num_epochs})")

    log_every_n_steps = log_every_n_steps or get_config_value("trainer.log_every_n_steps", 50)
    if not isinstance(log_every_n_steps, int) or log_every_n_steps <= 0:
        raise ValueError(f"log_every_n_steps must be a positive integer (got {log_every_n_steps})")

    cache_rate = (
        cache_rate
        if cache_rate is not None
        else get_config_value("data.train_dataloaders.dataset.cache_rate", 1.0)
    )
    if not isinstance(cache_rate, (int, float)) or not (0 <= cache_rate <= 1):
        raise ValueError(f"cache_rate must be a float between 0 and 1 (got {cache_rate})")

    replace_rate = (
        replace_rate
        if replace_rate is not None
        else get_config_value("data.train_dataloaders.dataset.replace_rate", 0.1)
    )
    if not isinstance(replace_rate, (int, float)) or not (0 <= replace_rate <= 1):
        raise ValueError(f"replace_rate must be a float between 0 and 1 (got {replace_rate})")

    if num_gpus is None:
        accelerator = get_config_value("trainer.accelerator")
        if accelerator == "gpu":
            num_gpus = get_config_value("trainer.devices", 1)
            if not isinstance(num_gpus, int) or num_gpus <= 0:
                raise ValueError(
                    f"Invalid num_gpus extracted from config: {num_gpus}. Must be a positive integer."
                )
        else:
            num_gpus = None

    if log_every_n_steps > max_num_epochs:
        logger.error("Logging interval must be less than or equal to the maximum number of epochs.")
        raise ValueError(
            f"Logging interval ({log_every_n_steps}) must be <= maximum epochs ({max_num_epochs})"
        )

    if not os.path.exists(train_dataframe_path):
        logger.warning(f"Training dataframe path does not exist: {train_dataframe_path}")
    if not os.path.exists(val_dataframe_path):
        logger.warning(f"Validation dataframe path does not exist: {val_dataframe_path}")

    overrides = _generate_overrides_for_model_training(
        model_manifest_name,
        run_name,
        crop_size,
        train_dataframe_path,
        val_dataframe_path,
        max_num_epochs=max_num_epochs,
        log_every_n_steps=log_every_n_steps,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        num_gpus=num_gpus,
    )

    cytodl_model = CytoDLModel()
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
    exclude_cell_piling: bool,
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
    exclude_cell_piling
        Exclude cell piling timepoints if True, include them if False.
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
            "resolution_level": resolution_level,
            "z_slice_offsets": z_slice_offsets,
            "exclude_cell_piling": exclude_cell_piling,
        },
        locations={
            "training": DataframeLocation(fmsid=train_fmsid, s3uri=None),
            "validation": DataframeLocation(fmsid=val_fmsid, s3uri=None),
        },
    )

    # save the updated or new manifest
    save_dataframe_manifest(dataframe_manifest)


def initialize_diffae_model_for_finetuning(
    base_model: CytoDLModel,
    template_finetune_config: "DictConfig | ListConfig",
    finetuned_model_manifest_name: str,
    finetuned_run_name: str,
    train_dataframe_path: str,
    val_dataframe_path: str,
    max_num_epochs: int = 100,
    log_every_n_steps: int = 50,
    cache_rate: float = 1.0,
    replace_rate: float = 0.1,
    num_gpus: int | None = None,
) -> CytoDLModel:
    """
    Initialize a DiffAE model for training.

    Parameters
    ----------
    base_model
        The baseline DiffAE model to finetune.
    template_finetune_config
        The template configuration for finetuning the DiffAE model.
    finetuned_model_manifest_name
        The name of the finetuned model manifest to which this training session belongs.
    finetuned_run_name
        The MLFlow run name for the finetuned model training session.
    train_dataframe_path
        The path to the image loading metadata dataframe for the training dataset.
    val_dataframe_path
        The path to the image loading metadata dataframe for the validation dataset.
    diffae_ckpt_path
        The path to the DiffAE checkpoint to finetune.
    max_num_epochs
        The maximum number of epochs for which to train the model.
    log_every_n_steps
        The interval at which to log training metrics.
    cache_rate
        The fraction of the dataset to cache in memory for training.
    replace_rate
        The replace rate for cached data.
    num_gpus
        Number of GPUs to use for training. If None, use the CPU!

    Returns
    -------
    cytodl_model
        An initialized CytoDLModel for finetuning the DiffAE model.
    """
    # override base model with finetuning config parameters
    # ** except ** keep the checkpoint path from the loaded model
    checkpoint_override = {"checkpoint_path": base_model.cfg["checkpoint_path"]}
    template_finetune_config.update(checkpoint_override)
    # override downloaded model config with finetuning config
    base_model.override_config(template_finetune_config)

    # generate overrides for train.yaml for finetuning
    overrides = _generate_overrides_for_finetuning(
        finetuned_model_manifest_name=finetuned_model_manifest_name,
        finetuned_run_name=finetuned_run_name,
        train_dataframe_path=train_dataframe_path,
        val_dataframe_path=val_dataframe_path,
        max_num_epochs=max_num_epochs,
        log_every_n_steps=log_every_n_steps,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        num_gpus=num_gpus,
    )

    # init model
    base_model.override_config(overrides)

    return base_model


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
