import logging
import os
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import Field
from pydantic.dataclasses import dataclass

from endo_pipeline.io import get_output_path, get_repository_root_dir

logger = logging.getLogger(__name__)


@dataclass
class ModelConfigOverride:
    run_name: str
    """Run name."""

    model_manifest_name: str
    """Manifest name."""

    task_name: Literal["train", "eval"]
    """Model task name."""

    template_config: str = "diffae_train.yaml"
    """Name of model config template."""

    crop_size: int | None = Field(None, gt=0)
    """Number of pixels in each dimension of the image crop to use for training."""

    train_dataframe_path: Path | None = None
    """Path to the training dataset (image loading metadata) parquet file."""

    val_dataframe_path: Path | None = None
    """Path to the validation dataset (image loading metadata) parquet file."""

    max_epochs: int | None = Field(None, gt=0)
    """Maximum number of epochs to train the model for."""

    cache_rate: float | None = Field(None, ge=0, le=1)
    """Fraction of the dataset to cache in memory for training."""

    replace_rate: float | None = Field(None, ge=0, le=1)
    """Rate at which cached data is replaced."""

    log_steps: int | None = Field(None, gt=0)
    """Interval at which to log training metrics."""

    num_gpus: int | None = Field(None, gt=0)
    """Number of GPUs to use. None indicates that CPU should be used."""

    def __post_init__(self):
        # Adding on a few more checks for the path!
        config_path = Path(self.template_config)
        if not config_path.exists():
            config_path = Path(__file__).resolve().parent / self.template_config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        config = OmegaConf.load(config_path)

        if self.train_dataframe_path is None:
            train = OmegaConf.select(config, "data.train_dataloaders.dataset.dataframe_path")

            if train is None:
                logger.error("Training dataframe could not be found in config")
                raise ValueError("Training dataframe is required and not found in the config")
            else:
                self.train_dataframe_path = Path(train)

            if self.train_dataframe_path.exists():
                logger.error(
                    "Training dataframe does not exist at [ %s ]", self.train_dataframe_path
                )
                raise ValueError(f"Training dataframe not found at [ {self.train_dataframe_path}]")

        if self.val_dataframe_path is None:
            val = OmegaConf.select(config, "data.val_dataloaders.dataset.dataframe_path")

            if val is None:
                logger.error("Validation dataframe could not be found in config")
                raise ValueError("Validation dataframe is required and not found in the config")
            else:
                self.val_dataframe_path = Path(val)

            if self.val_dataframe_path.exists():
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

    def to_dict(self, use_timestamp: bool = False):
        """Convert to overrides dict. If `use_timestamp`, dirs include a time-string."""
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
        subdir = f"{self.task_name}_{time_tag}" if use_timestamp else self.task_name

        # create output directories if they do not exist
        training_run_checkpoint_path = get_output_path(
            "models", self.model_manifest_name, self.run_name, subdir, "checkpoints"
        )
        training_run_log_path = get_output_path(
            "models", self.model_manifest_name, self.run_name, subdir, "logs"
        )

        multiplier = (1 - self.cache_rate) / (self.cache_rate * self.replace_rate) + 1
        effective_min_epochs = int(2500 * multiplier)
        effective_max_epochs = int(self.max_epochs * multiplier)
        effective_save_images_epochs = int(10 * multiplier)

        overrides = {
            # update experiment name and run name
            "experiment_name": self.model_manifest_name,
            "run_name": self.run_name,
            # get repo root directory and current working directory
            "paths.root_dir": get_repository_root_dir().as_posix(),
            "paths.work_dir": os.getcwd(),
            # save outputs to user-specified directory
            "paths.output_dir": training_run_log_path.as_posix(),
            "paths.log_dir": "${paths.output_dir}",
            "callbacks.model_checkpoint.dirpath": training_run_checkpoint_path.as_posix(),
            # set crop size from input via model.image_shape,
            "model.image_shape": [1, self.crop_size, self.crop_size],
            # set training and validation dataframe paths
            # and caching parameters
            "data.train_dataloaders.dataset.dataframe_path": self.train_dataframe_path.as_posix(),
            "data.train_dataloaders.dataset.cache_rate": self.cache_rate,
            "data.train_dataloaders.dataset.replace_rate": self.replace_rate,
            "data.val_dataloaders.dataset.dataframe_path": self.val_dataframe_path.as_posix(),
            "data.val_dataloaders.dataset.cache_rate": self.cache_rate,
            "data.val_dataloaders.dataset.replace_rate": self.replace_rate,
            # override the effective epochs calculations
            "model.save_images_every_n_epochs": effective_save_images_epochs,
            "trainer.min_epochs": effective_min_epochs,
            "trainer.max_epochs": effective_max_epochs,
            # turn off config printing, will get saved locally instead
            "extras.print_config": False,
            "trainer.log_every_n_steps": self.log_steps,
            "trainer.accelerator": "cpu" if self.num_gpus is None else "gpu",
            "trainer.devices": self.num_gpus or 1,
        }
        # If single GPU or none, use "auto" strategy
        if self.num_gpus is None or self.num_gpus == 1:
            overrides["trainer.strategy"] = "auto"
        return overrides
