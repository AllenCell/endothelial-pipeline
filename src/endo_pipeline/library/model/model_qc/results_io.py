"""Parquet persistence for model QC inference results.

The supplementary Model-QC figure pipeline persists every
``(model, seed, example)`` evaluation row to disk so the companion
plot workflow can render the bar chart without re-running GPU
diffusion inference.

On-disk layout
--------------

One long-format parquet per ``(manifest_name, run_name)`` is written
under the workflow output directory using the filename pattern
``{manifest_name}_{run_name}.parquet``.  All parquets for a given
``manifest_name`` are catalogued in a single dataframe manifest named
``diffae_model_comparison_metrics_<manifest_name>``, whose
``locations`` dict is keyed by ``run_name``.  This per-manifest grouping
mirrors the model-manifest layout and makes it easy to add or remove
individual runs without re-running everything.

Long-format schema (one row per example x seed x model):

- ``manifest_name``, ``run_name``     -- model identity
- ``random_seed``                     -- noise/RNG seed used for this row
- ``example_set``                     -- e.g. ``"rep_2_positions"``
- ``example_idx``                     -- 0-based position in the curated list
- ``dataset_name``, ``position``, ``timepoint``, ``crop_x_start``,
  ``crop_y_start``, ``description``   -- :class:`ExampleImage` metadata
- ``correlation_100``, ``ssim_100``, ``lpips_100`` -- denoise-from-100 % metrics
- ``baseline_correlation``, ``baseline_ssim``, ``baseline_lpips``
  -- next-timepoint baseline (nullable when baseline could not be computed)
"""

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from endo_pipeline.settings.column_names import ColumnName

if TYPE_CHECKING:
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey
    from endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)


def per_model_parquet_filename(manifest_name: str, run_name: str) -> str:
    """Filename of the per-(manifest, run) metrics parquet."""
    return f"{manifest_name}_{run_name}.parquet"


def per_model_parquet_path(out_dir: Path, manifest_name: str, run_name: str) -> Path:
    """Absolute path of the per-(manifest, run) metrics parquet under ``out_dir``."""
    return Path(out_dir) / per_model_parquet_filename(manifest_name, run_name)


DATAFRAME_COLUMNS: list[str] = [
    "manifest_name",
    "run_name",
    ColumnName.ModelQC.RANDOM_SEED,
    ColumnName.ModelQC.EXAMPLE_SET,
    ColumnName.ModelQC.EXAMPLE_IDX,
    ColumnName.ModelQC.DATASET_NAME,
    "position",
    "timepoint",
    "crop_x_start",
    "crop_y_start",
    "description",
    "correlation_100",
    "ssim_100",
    "lpips_100",
    "baseline_correlation",
    "baseline_ssim",
    "baseline_lpips",
]


def _dataframe_to_result(df: pd.DataFrame, model_key: "ModelKey", seed: int) -> dict:
    """Inverse of :func:`_result_to_dataframe` for one ``(model, seed)`` slice.

    Reconstructs the dict shape that :func:`aggregate_seed_metrics` expects.
    """
    sub = df[
        (df["manifest_name"] == model_key.manifest_name)
        & (df["run_name"] == model_key.run_name)
        & (df["random_seed"] == seed)
    ]
    example_set_labels = sub["example_set"].drop_duplicates().tolist()

    result: dict = {
        "model_key": model_key,
        "random_seed": int(seed),
        "example_set_labels": example_set_labels,
    }
    for label in example_set_labels:
        rows = sub[sub["example_set"] == label].sort_values("example_idx")
        bucket: dict[str, list[float]] = {
            "correlations_100": [float(v) for v in rows["correlation_100"]],
            "ssims_100": [float(v) for v in rows["ssim_100"]],
            "lpips_100": [float(v) for v in rows["lpips_100"]],
        }
        if rows["baseline_correlation"].notna().any():
            bucket["baseline_correlations"] = [
                float(v) for v in rows["baseline_correlation"] if not _is_nan(v)
            ]
            bucket["baseline_ssims"] = [float(v) for v in rows["baseline_ssim"] if not _is_nan(v)]
            bucket["baseline_lpips"] = [float(v) for v in rows["baseline_lpips"] if not _is_nan(v)]
        result[label] = bucket
    return result


def _is_nan(v: object) -> bool:
    try:
        return bool(math.isnan(float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def load_results_from_manifests(
    manifests: Iterable["DataframeManifest"],
) -> tuple[dict["ModelKey", dict[int, dict]], list["ModelKey"], list[int]]:
    """Load every ``(model, seed)`` result from one or more dataframe manifests.

    Each manifest is expected to catalogue the per-(manifest, run) parquets
    for a single ``manifest_name``, with ``locations`` keyed by ``run_name``.
    Each location is resolved via :func:`endo_pipeline.io.load_dataframe`,
    so the underlying parquet can come from FMS, local path, or S3.

    Returns
    -------
    all_seed_results
        ``{ModelKey: {seed: result_dict}}`` reshaped to match the in-memory
        layout produced by :func:`evaluate_single_model`.
    model_keys
        Model keys in the order they first appear across the manifests.
    seeds
        Sorted union of seeds present in the parquets.
    """
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey

    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        if not manifest.locations:
            raise FileNotFoundError(f"Dataframe manifest {manifest.name!r} has no locations.")
        for run_name, loc in manifest.locations.items():
            df_run = load_dataframe(loc)
            if df_run.empty:
                logger.warning(
                    "Dataframe manifest %r entry %r resolved to an empty parquet.",
                    manifest.name,
                    run_name,
                )
                continue
            frames.append(df_run)

    if not frames:
        raise FileNotFoundError(
            "No non-empty parquets resolved from the supplied dataframe manifests."
        )

    df = pd.concat(frames, ignore_index=True)
    pairs = df[["manifest_name", "run_name"]].drop_duplicates().itertuples(index=False)
    model_keys = [ModelKey(str(mn), str(rn)) for mn, rn in pairs]
    all_seeds = sorted(int(s) for s in df["random_seed"].unique())

    all_seed_results: dict[ModelKey, dict[int, dict]] = {
        mk: {seed: _dataframe_to_result(df, mk, seed) for seed in all_seeds} for mk in model_keys
    }
    return all_seed_results, model_keys, all_seeds
