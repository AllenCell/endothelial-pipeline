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
    from endo_pipeline.settings.examples import ExampleImage

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


def _aligned_baseline_values(
    metric_values: list[float], baseline_values: list[float], label: str
) -> list[float | None]:
    """Pad ``baseline_values`` to align with ``metric_values`` by index.

    Per-example baselines can be skipped when the next-timepoint image
    fails to load (the ``compute_baseline_for_example`` call is wrapped
    in a ``try/except`` in :func:`evaluate_single_model`).  Model metrics
    are *not* wrapped, so they can never be fewer than the baselines.
    The two scenarios are therefore:

    - ``len(baseline_values) == len(metric_values)`` -- normal case.
    - ``len(baseline_values) <  len(metric_values)`` -- baseline lookup
      failed on at least one example; pad the tail with ``None`` so the
      output dataframe stays rectangular and the example indices still
      line up.  A warning is logged so the misalignment is visible.
    """
    if not baseline_values:
        return [None] * len(metric_values)
    if len(baseline_values) == len(metric_values):
        return [float(v) for v in baseline_values]
    logger.warning(
        "Baseline list length (%d) does not match metric length (%d) for %s; "
        "padding with NaN -- per-example index alignment is best-effort.",
        len(baseline_values),
        len(metric_values),
        label,
    )
    padded: list[float | None] = [float(v) for v in baseline_values]
    padded.extend([None] * (len(metric_values) - len(baseline_values)))
    return padded


def _result_to_dataframe(
    model_key: "ModelKey",
    seed: int,
    result: dict,
    examples_by_set: dict[str, list["ExampleImage"]],
) -> pd.DataFrame:
    """Flatten one ``evaluate_single_model`` result into a long-format dataframe.

    Returns
    -------
    :
        One row per ``(example_set, example_idx)`` pair for the given
        ``(model_key, seed)``, with the ExampleImage metadata columns
        attached and baseline metrics aligned by index.
    """
    rows: list[dict] = []
    for example_set_label in result["example_set_labels"]:
        bucket = result[example_set_label]
        examples = examples_by_set.get(example_set_label, [])

        corrs = list(bucket.get("correlations_100", []))
        ssims = list(bucket.get("ssims_100", []))
        lpips = list(bucket.get("lpips_100", []))
        n = len(corrs)
        if not (len(ssims) == n and len(lpips) == n):
            raise ValueError(
                f"correlations_100/ssims_100/lpips_100 length mismatch for "
                f"{model_key.manifest_name}/{model_key.run_name} seed={seed} "
                f"set={example_set_label}: {len(corrs)}/{len(ssims)}/{len(lpips)}"
            )
        if examples and len(examples) != n:
            logger.warning(
                "Example list length (%d) does not match metric length (%d) for "
                "%s seed=%d set=%s; metadata columns may be sparse.",
                len(examples),
                n,
                model_key.manifest_name,
                seed,
                example_set_label,
            )

        base_label = (
            f"{model_key.manifest_name}/{model_key.run_name} "
            f"seed={seed} set={example_set_label}"
        )
        baseline_corrs = _aligned_baseline_values(
            corrs, list(bucket.get("baseline_correlations", [])), base_label + " corr"
        )
        baseline_ssims = _aligned_baseline_values(
            corrs, list(bucket.get("baseline_ssims", [])), base_label + " ssim"
        )
        baseline_lpips = _aligned_baseline_values(
            corrs, list(bucket.get("baseline_lpips", [])), base_label + " lpips"
        )

        for idx in range(n):
            example = examples[idx] if idx < len(examples) else None
            rows.append(
                {
                    "manifest_name": model_key.manifest_name,
                    "run_name": model_key.run_name,
                    ColumnName.ModelQC.RANDOM_SEED: int(seed),
                    ColumnName.ModelQC.EXAMPLE_SET: example_set_label,
                    ColumnName.ModelQC.EXAMPLE_IDX: idx,
                    ColumnName.ModelQC.DATASET_NAME: example.dataset_name if example else None,
                    "position": example.position if example else None,
                    "timepoint": example.timepoint if example else None,
                    "crop_x_start": example.crop_x_start if example else None,
                    "crop_y_start": example.crop_y_start if example else None,
                    "description": example.description if example else None,
                    "correlation_100": float(corrs[idx]),
                    "ssim_100": float(ssims[idx]),
                    "lpips_100": float(lpips[idx]),
                    "baseline_correlation": baseline_corrs[idx],
                    "baseline_ssim": baseline_ssims[idx],
                    "baseline_lpips": baseline_lpips[idx],
                }
            )

    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)


def write_per_model_parquets(
    out_dir: Path,
    results: list[tuple["ModelKey", int, dict]],
    examples_by_set: dict[str, list["ExampleImage"]] | None = None,
) -> dict["ModelKey", Path]:
    """Persist results to one parquet per ``(manifest_name, run_name)``.

    All rows for the same ``ModelKey`` (across seeds and examples) are
    concatenated into a single long-format parquet so each model can be
    added/removed independently without re-running every other model.

    Returns
    -------
    :
        Mapping from each ``ModelKey`` to the parquet path just written
        under ``out_dir``.
    """
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey

    eb = examples_by_set or {}
    rows_by_model: dict[ModelKey, list[pd.DataFrame]] = {}
    for mk, seed, result in results:
        rows_by_model.setdefault(mk, []).append(_result_to_dataframe(mk, seed, result, eb))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[ModelKey, Path] = {}
    for mk, frames in rows_by_model.items():
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=DATAFRAME_COLUMNS)
        )
        out_path = per_model_parquet_path(out_dir, mk.manifest_name, mk.run_name)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote per-model metrics parquet (%d rows) to %s", len(df), out_path)
        paths[mk] = out_path
    return paths


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
