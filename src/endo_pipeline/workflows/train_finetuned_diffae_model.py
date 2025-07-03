from pathlib import Path
from typing import Any, Literal

import fire
from cyto_dl.api import CytoDLModel

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import load_model_config
from src.endo_pipeline.library.model.mlflow import download_mlflow_artifact, get_ckpt_path


def _generate_overrides(
    user_overrides,
    save_path: Path,
    train_data_path: str,
    val_data_path: str,
    ckpt_path: str,
) -> dict:

    for subdir in ["logs", "checkpoints"]:
        (save_path / subdir).mkdir(parents=True, exist_ok=True)

    overrides = {
        # point to already projected paired dataset
        "data.train_dataloaders.dataset.csv_path": train_data_path,
        "data.val_dataloaders.dataset.csv_path": val_data_path,
        # change model target path
        "model._target_": "src.endo_pipeline.library.model.diffae_finetune.DiffAEFinetune",
        # load diffae checkpoint to finetune
        "checkpoint.ckpt_path": ckpt_path,
        "checkpoint.weights_only": True,
        "checkpoint.strict": False,
        # save to user-specified directory
        "model.save_dir": (save_path / "logs").as_posix(),
        "trainer.default_root_dir": save_path,
        "callbacks.model_checkpoint.dirpath": (save_path / "checkpoints").as_posix(),
        "paths.output_dir": (save_path / "logs").as_posix(),
        # do training
        "train": True,
        # make sure that last ckpt is saved
        "callbacks.model_checkpoint.monitor": None,
        # add mlflow logger
        "logger": {
            "mlflow": {
                "_target_": "cyto_dl.loggers.MLFlowLogger",
                "tracking_uri": "https://production.int.allencell.org/mlflow/",
                "experiment_name": "diffae",
                "run_name": "fixed_finetune_separate_encoder",
            }
        },
    }
    overrides.update(user_overrides)
    return overrides


def main(
    model_name: str = "diffae_04_10",
    dataset_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    model_template_name: str = "live_fixed_finetune_template",
    overrides: dict[str, Any] = {},
):
    """
    Finetune a DiffAE model with paired live/fixed data.

    Parameters
    ----------
    dataset_type: Literal['live_fixed', '20x_40x']
        The type of dataset to use for finetuning. This should match the dataset type used during the `paired_data_validation` step.
    model_name: str
        The name of the model to use for finetuning. This should correspond to a directory in `outputs/models/` and match the model name used during the `paired_data_validation` step.
    model_template_name: str
        The name of the model template to use for finetuning. This should correspond to a model in `model_config.yaml`
    overrides: dict[str, Any]
        Additional overrides for the training configuration. This can include any parameters that can be set in the config file, such as learning rate, batch size, etc.
    """
    save_dir = Path(
        get_output_path(f"finetune_paired_dataset/finetune_{model_name}_on_{dataset_type}")
    )

    manifest_path = Path(get_output_path(f"finetune_paired_dataset/{dataset_type}"))

    # download model to finetune
    finetune_run_id = load_model_config(model_name).mlflow_run_id
    diffae_ckpt_path = get_ckpt_path(run_id=finetune_run_id)
    download_mlflow_artifact(finetune_run_id, diffae_ckpt_path, save_dir)

    overrides = _generate_overrides(
        user_overrides=overrides,
        save_path=save_dir,
        train_data_path=manifest_path / "train.csv",
        val_data_path=manifest_path / "val.csv",
        ckpt_path=save_dir / diffae_ckpt_path,
    )

    # download template config
    template_run_id = load_model_config(model_template_name).mlflow_run_id
    download_mlflow_artifact(
        run_id=template_run_id, artifact_path="config/train.yaml", dst_path=save_dir
    )

    model = CytoDLModel()
    model.load_config_from_file(save_dir / "config" / "train.yaml")
    model.override_config(overrides)
    model.train()


if __name__ == "__main__":
    fire.Fire(main)
