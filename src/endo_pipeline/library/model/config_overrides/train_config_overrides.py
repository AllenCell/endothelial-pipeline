"""Model config override classes for model training runs."""

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import Field
from pydantic.dataclasses import dataclass

from endo_pipeline.configs import load_model_config
from endo_pipeline.io import get_output_path, get_repository_root_dir
from endo_pipeline.io.mlflow import MLFLOW_TRACKING_URI
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
from endo_pipeline.settings.workflow_defaults import DEFAULT_NUM_LATENT_DIMENSIONS

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ModelConfigOverrideTrain:
    """CytoDL model config overrides for training."""

    run_name: str
    """Run name."""

    model_manifest_name: str
    """Manifest name."""

    template_config: str = DIFFAE_MODEL_TRAIN_CONFIG
    """Name of model config template."""

    crop_size: int | None = Field(default=None, gt=0)
    """Number of pixels in each dimension of the image crop to use for training."""

    condition_key: str | None = None
    """Key for the image channel to use for semantic conditioning of the diffusion model."""

    latent_dim: int | None = Field(default=None, gt=0)
    """Number of dimensions for the latent space of the semantic encoder."""

    train_dataframe_path: str | None = None
    """Path or URI to the training dataset (image loading metadata) parquet file."""

    val_dataframe_path: str | None = None
    """Path or URI to the validation dataset (image loading metadata) parquet file."""

    max_epochs: int | None = Field(default=None, gt=0)
    """Maximum number of epochs to train the model for."""

    epoch_multiplier: bool = True
    """Apply multiplier to number of epochs based on caching."""

    cache_rate: float | None = Field(default=None, ge=0, le=1)
    """Fraction of the dataset to cache in memory for training."""

    replace_rate: float | None = Field(default=None, ge=0, le=1)
    """Rate at which cached data is replaced."""

    log_steps: int | None = Field(default=None, gt=0)
    """Interval at which to log training metrics."""

    num_gpus: int | None = Field(default=None, gt=0)
    """Number of GPUs to use. None indicates that CPU should be used."""

    num_workers: int | None = Field(default=None, ge=0)
    """Number of workers to use. None indicates use 50% available on machine."""

    def __post_init__(self):
        """Post initialization steps for model config overrides."""
        self.task_name = "train"

        config = load_model_config(self.template_config)

        if self.train_dataframe_path is None:
            train_path = OmegaConf.select(config, "data.train_dataloaders.dataset.dataframe_path")

            if train_path is None:
                logger.error("Training dataframe could not be found in config")
                raise ValueError("Training dataframe is required and not found in the config")
            else:
                self.train_dataframe_path = train_path

            if (
                not self.train_dataframe_path.startswith("s3://")
                and Path(self.train_dataframe_path).exists()
            ):
                logger.error(
                    "Training dataframe does not exist at [ %s ]", self.train_dataframe_path
                )
                raise ValueError(f"Training dataframe not found at [ {self.train_dataframe_path}]")

        if self.val_dataframe_path is None:
            val_path = OmegaConf.select(config, "data.val_dataloaders.dataset.dataframe_path")

            if val_path is None:
                logger.error("Validation dataframe could not be found in config")
                raise ValueError("Validation dataframe is required and not found in the config")
            else:
                self.val_dataframe_path = val_path

            if (
                not self.val_dataframe_path.startswith("s3://")
                and Path(self.val_dataframe_path).exists()
            ):
                logger.error(
                    "Validation dataframe does not exist at [ %s ]", self.val_dataframe_path
                )
                raise ValueError(f"Validation dataframe not found at [ {self.val_dataframe_path}]")

        if self.cache_rate is None:
            self.cache_rate = OmegaConf.select(
                config, "data.train_dataloaders.dataset.cache_rate", default=1.0
            )

        if self.replace_rate is None:
            self.replace_rate = OmegaConf.select(
                config, "data.train_dataloaders.dataset.replace_rate", default=1.0
            )

        if self.crop_size is None:
            self.crop_size = OmegaConf.select(config, "model.image_shape[1]", default=128)

        if self.condition_key is None:
            self.condition_key = OmegaConf.select(config, "model.condition_key", default="raw_bf")

        if self.latent_dim is None:
            self.latent_dim = OmegaConf.select(
                config, "model.semantic_encoder.num_classes", default=DEFAULT_NUM_LATENT_DIMENSIONS
            )

        if self.max_epochs is None:
            self.max_epochs = OmegaConf.select(config, "trainer.max_epochs", default=1000)

        if self.log_steps is None:
            self.log_steps = OmegaConf.select(config, "trainer.log_every_n_steps", default=50)

        if self.num_gpus is None:
            accelerator = OmegaConf.select(config, "trainer.accelerator")
            if accelerator == "gpu":
                self.num_gpus = OmegaConf.select(config, "trainer.devices", default=1)

        if self.log_steps > self.max_epochs:
            logger.error("Logging interval must be less than or equal to maximum number of epochs")
            raise ValueError(
                "Logging interval is great than max number of epochs "
                f"[ {self.log_steps} > {self.max_epochs} ]"
            )

        if self.num_workers is None:
            self.num_workers = int(0.5 * (os.cpu_count() or 0))

    def to_dict(self):
        """Convert to overrides dict."""
        # Create output directories for checkpoints and logs if they do not exist.
        checkpoint_path = get_output_path(
            "models",
            self.model_manifest_name,
            self.run_name,
            "checkpoints",
            include_timestamp=False,
        )
        log_path = get_output_path(
            "models",
            self.model_manifest_name,
            self.run_name,
            "logs",
            include_timestamp=False,
        )

        assert self.cache_rate is not None
        assert self.replace_rate is not None
        assert self.max_epochs is not None
        assert self.train_dataframe_path is not None
        assert self.val_dataframe_path is not None

        # Calculate effective epochs, if requested. Otherwise, set min and max
        # number of epochs to exactly the requested number.
        if self.epoch_multiplier:
            multiplier = (1 - self.cache_rate) / (self.cache_rate * self.replace_rate) + 1
            effective_min_epochs = int(5000 * multiplier)
            effective_max_epochs = max(
                int(self.max_epochs * multiplier), int(1.5 * effective_min_epochs)
            )
            effective_save_images_epochs = int(10 * multiplier)
        else:
            effective_min_epochs = 1
            effective_max_epochs = self.max_epochs
            effective_save_images_epochs = 1

        overrides = {
            # update experiment name and run name
            "experiment_name": self.model_manifest_name,
            "run_name": self.run_name,
            # get repo root directory and current working directory
            "paths.root_dir": get_repository_root_dir().as_posix(),
            "paths.work_dir": Path.cwd().as_posix(),
            # save outputs to user-specified directory
            "paths.output_dir": log_path.as_posix(),
            "paths.log_dir": "${paths.output_dir}",
            "callbacks.model_checkpoint.dirpath": checkpoint_path.as_posix(),
            # set crop size from input via model.image_shape,
            "model.image_shape": [1, self.crop_size, self.crop_size],
            # update the callbacks,
            "callbacks.model_checkpoint.save_last": True,
            "callbacks.model_checkpoint.monitor": "val/loss",
            "callbacks.model_checkpoint.save_top_k": 3,
            # set condition key
            "model.condition_key": self.condition_key,
            # set number of latent dimensions
            "lat_dim": self.latent_dim,
            # set training and validation dataframe paths and caching parameters
            "data.train_dataloaders.dataset.dataframe_path": self.train_dataframe_path,
            "data.train_dataloaders.dataset.cache_rate": self.cache_rate,
            "data.train_dataloaders.dataset.replace_rate": self.replace_rate,
            "data.val_dataloaders.dataset.dataframe_path": self.val_dataframe_path,
            "data.val_dataloaders.dataset.cache_rate": self.cache_rate,
            "data.val_dataloaders.dataset.replace_rate": self.replace_rate,
            # override the effective epochs calculations
            "model.save_images_every_n_epochs": effective_save_images_epochs,
            "trainer.min_epochs": effective_min_epochs,
            "trainer.max_epochs": effective_max_epochs,
            # turn off config printing, will get saved locally instead
            "extras.print_config": False,
            "trainer.log_every_n_steps": self.log_steps,
            # set device usage
            "trainer.accelerator": "cpu" if self.num_gpus is None else "gpu",
            "trainer.devices": self.num_gpus or 1,
            "trainer.precision": "bf16-mixed" if self.num_gpus is None else "16-mixed",
            # set number of workers
            "data.train_dataloaders.num_workers": self.num_workers,
            "data.train_dataloaders.dataset.num_init_workers": self.num_workers,
            "data.train_dataloaders.dataset.num_replace_workers": self.num_workers,
            "data.val_dataloaders.num_workers": self.num_workers,
            "data.val_dataloaders.dataset.num_init_workers": self.num_workers,
            "data.val_dataloaders.dataset.num_replace_workers": self.num_workers,
            # set logger uri
            "logger.mlflow.tracking_uri": MLFLOW_TRACKING_URI,
        }

        # If no workers, turn off persistent workers.
        if self.num_workers == 0:
            overrides["data.train_dataloaders.persistent_workers"] = False
            overrides["data.val_dataloaders.persistent_workers"] = False

        # If single GPU or none, use "auto" strategy
        if self.num_gpus is None or self.num_gpus == 1:
            overrides["trainer.strategy"] = "auto"

        return overrides
