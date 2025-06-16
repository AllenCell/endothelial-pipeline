from pathlib import Path
from typing import Any

import fire
from cyto_dl.api import CytoDLModel

from cellsmap.model_features.utils.mlflow_utils import download_mlflow_artifact


def generate_overrides(
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
    save_dir: str,
    train_csv_path: str,
    val_csv_path: str,
    diffae_ckpt_path: str,
    base_model_run_id: str = "a5b6274c683948b5a3e6ef892b0c88d5",
    overrides: dict[str, Any] = {},
):
    """
    Finetune a DiffAE model with paired live/fixed data.

    Parameters
    ----------
    save_dir: str
        The directory where the model and logs will be saved. This can be any empty folder.
    train_csv_path: str
        Path to the training CSV file containing paired data.
    val_csv_path: str
        Path to the validation CSV file containing paired data.
    diffae_ckpt_path: str
        Path to the DiffAE checkpoint to finetune.
    base_model_run_id: str
        The MLflow run ID of the base DiffAE model. This is used to download the template config and requirements.
    overrides: dict[str, Any]
        Additional overrides for the training configuration. This can include any parameters that can be set in the config file, such as learning rate, batch size, etc.
    """
    save_dir = Path(save_dir)

    overrides = generate_overrides(
        user_overrides=overrides,
        save_path=save_dir,
        train_data_path=train_csv_path,
        val_data_path=val_csv_path,
        ckpt_path=diffae_ckpt_path,
    )

    # download template config
    download_mlflow_artifact(
        run_id=base_model_run_id, artifact_path="config/train.yaml", dst_path=save_dir
    )

    model = CytoDLModel()
    model.load_config_from_file(save_dir / "config" / "train.yaml")
    model.override_config(overrides)
    model.train()


if __name__ == "__main__":
    fire.Fire(main)
