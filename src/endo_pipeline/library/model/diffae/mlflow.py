import math
import os
import re
import shutil
import tempfile
import warnings
from argparse import Namespace
from collections.abc import Mapping
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

            params_dict = vars(params) if isinstance(params, Namespace) else params
            OmegaConf.save(OmegaConf.create(params_dict), conf_path)
            reqs_path.write_text("\n".join(requirements))

            self.experiment.log_artifact(self.run_id, str(conf_path), "config")
            self.experiment.log_artifact(self.run_id, str(reqs_path), "requirements")

    # Metrics + log-log version
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
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
            log_metrics = {f"log10_{k}": math.log10(v) for k, v in metrics.items() if v > 0}
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

    def _resolve_last_checkpoint(self, ckpt_callback: ModelCheckpoint) -> str | None:
        """Determine the path of the most recently saved checkpoint."""
        # Try last_model_path (set when save_last=True)
        last_path = getattr(ckpt_callback, "last_model_path", "")
        if last_path and Path(last_path).is_file():
            return last_path

        # Try best_model_path (always the last saved when monitor=None)
        best_path = ckpt_callback.best_model_path
        if best_path and Path(best_path).is_file():
            return best_path

        # Scan dirpath for the checkpoint with the highest epoch number
        dirpath = ckpt_callback.dirpath
        if dirpath and Path(dirpath).is_dir():
            epoch_re = re.compile(r"epoch[=_](\d+)")
            ckpt_files = [
                f
                for f in Path(dirpath).iterdir()
                if f.is_file()
                and f.suffix == ".ckpt"
                and not f.name.startswith("best_")
                and f.name != "last.ckpt"
            ]
            # Pair each file with its parsed epoch number (if present)
            parsed = []
            for f in ckpt_files:
                m = epoch_re.search(f.stem)
                if m:
                    parsed.append((int(m.group(1)), f))
            if parsed:
                return str(max(parsed, key=lambda t: t[0])[1])
            # Fall back to most recently modified if no epoch number is found
            if ckpt_files:
                return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))

        return None

    def _log_named_checkpoint_as_artifact(
        self,
        source_path: str,
        target_name: str,
        artifact_path: str,
    ) -> None:
        """Log a checkpoint under a given name as an MLflow artifact.

        Creates a temporary copy of the checkpoint at ``source_path`` renamed
        to ``target_name``, uploads it to MLflow under ``artifact_path``, and
        cleans up the copy afterwards.  If a file with ``target_name`` already
        exists in the same directory (e.g. the source itself), it is logged
        directly and no copy is made.

        Parameters
        ----------
        source_path
            Absolute or relative path to the source checkpoint file.
        target_name
            Filename to use for the uploaded artifact (e.g. ``"best.ckpt"``).
        artifact_path
            Destination directory inside the MLflow artifact store
            (e.g. ``"checkpoints/val_loss"``).
        """
        # Get name of target path by renaming file in source path
        target_path = Path(source_path).with_name(target_name)

        # If the target path does not exist, make a temporary copy of the source path
        if not target_path.exists():
            shutil.copy2(source_path, target_path)

        # Log the target path as an artifact
        self.experiment.log_artifact(self.run_id, str(target_path), artifact_path)

        # Delete the temporary copy, if one was created.
        if Path(source_path).resolve() != target_path.resolve():
            target_path.unlink()

    def _after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        monitor = ckpt_callback.monitor

        if monitor is not None:
            artifact_path = f"checkpoints/{monitor}"
            # Sanitize monitor name for use in filenames (e.g. "val/loss" → "val_loss")
            safe_monitor = monitor.replace("/", "_")

            # Top-k management (upload new, delete old)
            existing = {
                a.path.split("/")[-1]
                for a in self.experiment.list_artifacts(self.run_id, artifact_path)
            }
            top_k = {k.split("/")[-1] for k in ckpt_callback.best_k_models.keys()}
            best_name = f"best_{safe_monitor}.ckpt"
            to_delete = existing - top_k - {best_name, "last.ckpt"}
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
                if ckpt_callback.dirpath is not None:
                    self.experiment.log_artifact(
                        self.run_id,
                        local_path=os.path.join(ckpt_callback.dirpath, ckpt),
                        artifact_path=artifact_path,
                    )
                else:
                    raise ValueError("ckpt_callback.dirpath must not be None")

            # Log best_{monitor}.ckpt when a metric is monitored
            best_src = ckpt_callback.best_model_path
            if best_src and Path(best_src).is_file():
                self._log_named_checkpoint_as_artifact(best_src, best_name, artifact_path)
            else:
                self._warn(
                    f"best_model_path is empty or missing ({best_src!r}). "
                    f"Skipping {best_name} logging."
                )

        else:
            # When monitor is None there is no metric to rank checkpoints by,
            # so no "best" checkpoint is saved.  Only last.ckpt (below) is
            # relevant in this mode.
            artifact_path = "checkpoints"

        # --- Log last.ckpt only when save_last is enabled in the callback config ---
        if ckpt_callback.save_last:
            last_src = self._resolve_last_checkpoint(ckpt_callback)
            if last_src:
                self._log_named_checkpoint_as_artifact(last_src, "last.ckpt", artifact_path)
            else:
                self._warn("Could not determine last checkpoint path. Skipping last.ckpt logging.")


def _delete_local_artifact(repo, artifact_path: str):
    p = Path(
        local_file_uri_to_path(
            os.path.join(repo._artifact_dir, artifact_path) if artifact_path else repo._artifact_dir
        )
    )
    if p.is_file():
        p.unlink()
