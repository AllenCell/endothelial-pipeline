import logging
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
from endo_pipeline.library.model.model_config_overrides import ModelConfigOverride
from endo_pipeline.manifests import DataframeLocation, DataframeManifest, save_dataframe_manifest

logger = logging.getLogger(__name__)


def initialize_diffae_model(
    template_training_config,
    overrides: ModelConfigOverride | None = None,
) -> "CytoDLModel":
    """
    Initialize a DiffAE model for training, applying overrides and configuration.

    Parameters
    ----------
    template_training_config
        The loaded training configuration to use as the base.
    overrides
        Optionally, a ModelConfigOverride object (read from code, CLI, etc).
        If None, uses values from config as much as possible.

    Returns
    -------
    CytoDLModel
        An initialized ``CytoDLModel`` for training the DiffAE model.
    """

    cytodl_model = CytoDLModel()
    cytodl_model.load_config_from_dict(template_training_config)
    if overrides is not None:
        cytodl_model.override_config(overrides.to_dict())

    return cytodl_model


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
        # update the experiment name and run name
        "experiment_name": finetuned_model_manifest_name,
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
            "resolution_level": resolution_level,
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
