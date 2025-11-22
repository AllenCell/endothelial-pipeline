import math
import os
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger as _LightningMLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.file_utils import local_file_uri_to_path
from omegaconf import OmegaConf


class MLFlowLogger(_LightningMLFlowLogger):
    """
    Logger is copied from cyto-DL (commit hash: cd55c519b6a18018077d54e0da1871263f1c1c5c),
    with several enhancements:
    - Added support for log-log scaled plots.
    - The `_pylogger` function has been redefined to avoid circular imports.
    - Warning handling logic has been adjusted in a function with a check on the logger.
    """

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        fault_tolerant: bool = True,
        log_log_scale: bool = True,
    ):
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
        )
        self.fault_tolerant = fault_tolerant
        self.log_log_scale = log_log_scale

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def _pylogger(self):
        try:
            from cyto_dl import utils

            return utils.get_pylogger(__name__)
        except Exception:
            return None

    def _warn(self, msg: str):
        log = self._pylogger()
        if log:
            log.warn(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace, mode: str = "train") -> None:
        requirements = params.pop("requirements", [])
        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_path = Path(tmp_dir) / f"{mode}.yaml"
            reqs_path = Path(tmp_dir) / f"{mode}-requirements.txt"

            OmegaConf.save(OmegaConf.create(params), conf_path)
            reqs_path.write_text("\n".join(requirements))

            self.experiment.log_artifact(self.run_id, str(conf_path), "config")
            self.experiment.log_artifact(self.run_id, str(reqs_path), "requirements")

    # Metrics + log-log version
    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        # Original
        try:
            super().log_metrics(metrics, step)
        except Exception as e:
            if self.fault_tolerant:
                self._warn(f"Original log_metrics failed: {e}")
            else:
                raise

        # log-log
        if not self.log_log_scale or step is None or step <= 0:
            return

        try:
            log_step = math.log10(step)
            log_metrics = {
                f"log10_{k}": math.log10(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v > 0
            }
            if log_metrics:
                super().log_metrics(log_metrics, step=int(log_step * 1_000_000))
        except Exception as e:
            if self.fault_tolerant:
                self._warn(f"Log-log metrics failed: {e}")
            else:
                raise

    # Checkpoint handling
    def after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        try:
            self._after_save_checkpoint(ckpt_callback)
        except Exception as e:
            if self.fault_tolerant:
                self._warn(f"after_save_checkpoint failed: {e}")
            else:
                raise

    def _after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        monitor = ckpt_callback.monitor
        if monitor is not None:
            artifact_path = f"checkpoints/{monitor}"
            existing = {
                a.path.split("/")[-1]
                for a in self.experiment.list_artifacts(self.run_id, artifact_path)
            }
            top_k = {k.split("/")[-1] for k in ckpt_callback.best_k_models.keys()}
            to_delete = existing - top_k
            to_upload = top_k - existing

            repo = get_artifact_repository(self.experiment.get_run(self.run_id).info.artifact_uri)

            for ckpt in to_delete:
                if isinstance(repo, LocalArtifactRepository):
                    _delete_local_artifact(repo, f"checkpoints/{monitor}/{ckpt}")
                elif hasattr(repo, "delete_artifacts"):
                    repo.delete_artifacts(f"checkpoints/{monitor}/{ckpt}")
                else:
                    warnings.warn(
                        "Artifact deletion not supported...keeping all checkpoints.", stacklevel=2
                    )

            for ckpt in to_upload:
                self.experiment.log_artifact(
                    self.run_id,
                    local_path=os.path.join(ckpt_callback.dirpath, ckpt),
                    artifact_path=artifact_path,
                )

            best_path = Path(ckpt_callback.best_model_path).with_name("best.ckpt")
            os.link(ckpt_callback.best_model_path, best_path)
            self.experiment.log_artifact(self.run_id, str(best_path), artifact_path)
            best_path.unlink()

        else:

            fp = ckpt_callback.best_model_path
            artifact_path = "checkpoints"
            if ckpt_callback.save_top_k == 1:
                last_path = Path(fp).with_name("last.ckpt")
                os.link(fp, last_path)
                self.experiment.log_artifact(self.run_id, str(last_path), artifact_path)
                last_path.unlink()
            else:
                self.experiment.log_artifact(self.run_id, fp, artifact_path)


def _delete_local_artifact(repo, artifact_path: str):
    p = Path(
        local_file_uri_to_path(
            os.path.join(repo._artifact_dir, artifact_path) if artifact_path else repo._artifact_dir
        )
    )
    if p.is_file():
        p.unlink()
