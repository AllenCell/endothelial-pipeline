"""Model config override classes for model evaluation (inference) runs."""

import logging
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import Field
from pydantic.dataclasses import dataclass

from endo_pipeline.configs import load_model_config
from endo_pipeline.io import get_output_path, get_repository_root_dir
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_EVAL_CONFIG
from endo_pipeline.settings.diffae_feature_dataframes import CytoDLSaveDataKeys

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ModelConfigOverrideEval:
    """CytoDL model config overrides for evaluation."""

    run_name: str
    """Run name."""

    model_manifest_name: str
    """Manifest name."""

    template_config: str = DIFFAE_MODEL_EVAL_CONFIG
    """Name of model config template."""

    eval_dataframe_path: str | None = None
    """Path or URI to the evaluation dataset (image loading metadata) parquet file."""

    cache_rate: float | None = Field(default=None, ge=0, le=1)
    """Fraction of the dataset to cache in memory for evaluation."""

    replace_rate: float | None = Field(default=None, ge=0, le=1)
    """Rate at which cached data is replaced."""

    num_gpus: int | None = Field(default=None, gt=0)
    """Number of GPUs to use. None indicates that CPU should be used."""

    def __post_init__(self):
        """Post initialization steps for model config overrides."""
        self.task_name = "eval"

        config = load_model_config(self.template_config)

        if self.eval_dataframe_path is None:
            eval_path = OmegaConf.select(config, "data.predict_dataloaders.dataset.dataframe_path")

            if eval_path is None:
                logger.error("Evaluation dataframe could not be found in config")
                raise ValueError("Evaluation dataframe is required and not found in the config")
            else:
                self.eval_dataframe_path = eval_path

            if (
                not self.eval_dataframe_path.startswith("s3://")
                and Path(self.eval_dataframe_path).exists()
            ):
                logger.error(
                    "Evaluation dataframe does not exist at [ %s ]", self.eval_dataframe_path
                )
                raise ValueError(f"Evaluation dataframe not found at [ {self.eval_dataframe_path}]")

        if self.cache_rate is None:
            self.cache_rate = OmegaConf.select(
                config, "data.predict_dataloaders.dataset.cache_rate", default=1.0
            )

        if self.replace_rate is None:
            self.replace_rate = OmegaConf.select(
                config, "data.predict_dataloaders.dataset.replace_rate", default=1.0
            )

        if self.num_gpus is None:
            accelerator = OmegaConf.select(config, "trainer.accelerator")
            if accelerator == "gpu":
                self.num_gpus = OmegaConf.select(config, "trainer.devices", default=1)

    def to_dict(self, dataset_name: str, crop_pattern: Literal["grid", "tracked"]):
        """Convert to overrides dict."""
        # Create directories for outputs if they do not exist.
        output_path = get_output_path(
            "models",
            self.model_manifest_name,
            self.run_name,
            "outputs",
            include_timestamp=False,
        )

        # Build save suffix for outputs
        base_suffix = f"{dataset_name}_{self.model_manifest_name}_{self.run_name}"
        save_suffix = f"{base_suffix}_{crop_pattern}_features"

        assert self.eval_dataframe_path is not None

        overrides: dict[str, Any] = {
            # set task name
            "task_name": self.task_name,
            # update experiment name and run name
            "experiment_name": self.model_manifest_name,
            "run_name": self.run_name,
            # get repo root directory and current working directory
            "paths.root_dir": get_repository_root_dir().as_posix(),
            "paths.work_dir": Path.cwd().as_posix(),
            "paths.output_dir": output_path.as_posix(),
            # remove the training dataloader (which is not needed for eval)
            "data.train_dataloaders": None,
            # remove the validation dataloader (Which is not needed for eval)
            "data.val_dataloaders": None,
            # set evaluation dataframe paths and caching parameters
            "data.predict_dataloaders.dataset.dataframe_path": self.eval_dataframe_path,
            "data.predict_dataloaders.dataset.cache_rate": self.cache_rate,
            "data.predict_dataloaders.dataset.replace_rate": self.replace_rate,
            # turn off all callbacks and add prediction saver callback
            "callbacks": None,
            "callbacks.prediction_saver": {
                "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
                "save_dir": output_path.as_posix(),
                "meta_keys": [
                    CytoDLSaveDataKeys.TIMEPOINT.value,
                    Column.DiffAEData.START_Y.value,
                    Column.DiffAEData.START_X.value,
                    CytoDLSaveDataKeys.FILE_PATH.value,
                ],
                "save_suffix": save_suffix,
            },
            # turn off config printing, will get saved locally instead
            "extras.print_config": False,
            # set device usage
            "trainer.accelerator": "cpu" if self.num_gpus is None else "gpu",
            "trainer.devices": self.num_gpus or 1,
            "trainer.precision": "bf16-mixed" if self.num_gpus is None else "16-mixed",
        }

        # Additional overrides specific to track-based crops
        if crop_pattern == "tracked":
            overrides.update(
                {
                    # add prediction saver callback
                    "callbacks.prediction_saver": {
                        "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
                        "save_dir": output_path.as_posix(),
                        "meta_keys": [
                            CytoDLSaveDataKeys.TIMEPOINT.value,
                            Column.DiffAEData.START_Y.value,
                            Column.DiffAEData.START_X.value,
                            Column.DiffAEData.END_Y.value,
                            Column.DiffAEData.END_X.value,
                            CytoDLSaveDataKeys.FILE_PATH.value,
                            "track_id",
                        ],
                        "save_suffix": save_suffix,
                    },
                    # add cropping transform
                    "data.predict_dataloaders.dataset.transform.transforms[6]": {
                        "_target_": "cyto_dl.image.transforms.coordinate_crop.CropToCoordsd",
                        "keys": ["raw_bf"],
                        "start_keys": [
                            Column.DiffAEData.START_Y.value,
                            Column.DiffAEData.START_X.value,
                        ],
                        "end_keys": [Column.DiffAEData.END_Y.value, Column.DiffAEData.END_X.value],
                        "meta_keys": ["track_id"],
                    },
                    # persist coordinate data through MultiDimImageDataset
                    "data.predict_dataloaders.dataset.extra_columns": [
                        Column.DiffAEData.START_Y.value,
                        Column.DiffAEData.START_X.value,
                        Column.DiffAEData.END_Y.value,
                        Column.DiffAEData.END_X.value,
                        "track_id",
                    ],
                    # no spatial inferer needed
                    "model.spatial_inferer": None,
                }
            )

        # If single GPU or none, use "auto" strategy
        if self.num_gpus is None or self.num_gpus == 1:
            overrides["trainer.strategy"] = "auto"

        return overrides
