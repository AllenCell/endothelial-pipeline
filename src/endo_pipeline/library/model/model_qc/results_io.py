"""Parquet persistence for model QC inference results.

The supplementary Model-QC figure pipeline persists every
``(model, seed, example)`` evaluation row to a single long-format
parquet so the companion plot workflow can render the bar chart
without re-running GPU diffusion inference.

On-disk layout
--------------

A single ``model_qc_metrics.parquet`` under the workflow output
directory holds all rows for every model x seed x example.  That one
file is the sole entry of the emitted :class:`DataframeManifest` and
the sole thing optionally uploaded to FMS.

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
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey
    from endo_pipeline.manifests import DataframeManifest
    from endo_pipeline.settings.examples import ExampleImage

logger = logging.getLogger(__name__)


COMBINED_PARQUET_FILENAME = "model_qc_metrics.parquet"
COMBINED_MANIFEST_LOCATION_KEY = "metrics"

DATAFRAME_COLUMNS: list[str] = [
    "manifest_name",
    "run_name",
    "random_seed",
    "example_set",
    "example_idx",
    "dataset_name",
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


def combined_parquet_path(out_dir: Path) -> Path:
    """Absolute path of the single combined metrics parquet under ``out_dir``."""
    return Path(out_dir) / COMBINED_PARQUET_FILENAME


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
                    "random_seed": int(seed),
                    "example_set": example_set_label,
                    "example_idx": idx,
                    "dataset_name": example.dataset_name if example else None,
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


def write_combined_parquet(
    out_dir: Path,
    results: list[tuple["ModelKey", int, dict]],
    examples_by_set: dict[str, list["ExampleImage"]] | None = None,
) -> Path:
    """Persist every ``(model_key, seed, result)`` row to one combined parquet.

    Returns
    -------
    :
        Path of the combined parquet just written under ``out_dir``.
    """
    eb = examples_by_set or {}
    frames = [_result_to_dataframe(mk, seed, result, eb) for mk, seed, result in results]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DATAFRAME_COLUMNS)
    out_path = combined_parquet_path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote combined metrics parquet (%d rows) to %s", len(df), out_path)
    return out_path


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


def load_results_from_manifest(
    manifest: "DataframeManifest",
) -> tuple[dict["ModelKey", dict[int, dict]], list["ModelKey"], list[int]]:
    """Load every ``(model, seed)`` result from a :class:`DataframeManifest`.

    ``manifest.locations`` is expected to hold a single entry pointing
    at the combined metrics parquet written by the production inference
    workflow.  The location is resolved via
    :func:`endo_pipeline.io.load_dataframe`, so the file can come from
    FMS, local path, or S3.

    Returns
    -------
    all_seed_results
        ``{ModelKey: {seed: result_dict}}`` reshaped to match the in-memory
        layout produced by :func:`evaluate_single_model`.
    model_keys
        Model keys in the order they first appear in the parquet.
    seeds
        Sorted union of seeds present in the parquet.
    """
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey

    if not manifest.locations:
        raise FileNotFoundError(f"Dataframe manifest {manifest.name!r} has no locations.")
    if len(manifest.locations) > 1:
        logger.warning(
            "Dataframe manifest %r has %d locations; expected a single combined "
            "metrics parquet. Loading all and concatenating.",
            manifest.name,
            len(manifest.locations),
        )

    frames = [load_dataframe(loc) for loc in manifest.locations.values()]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise FileNotFoundError(
            f"Dataframe manifest {manifest.name!r} resolved to an empty parquet."
        )

    pairs = df[["manifest_name", "run_name"]].drop_duplicates().itertuples(index=False)
    model_keys = [ModelKey(str(mn), str(rn)) for mn, rn in pairs]
    all_seeds = sorted(int(s) for s in df["random_seed"].unique())

    all_seed_results: dict[ModelKey, dict[int, dict]] = {
        mk: {seed: _dataframe_to_result(df, mk, seed) for seed in all_seeds} for mk in model_keys
    }
    return all_seed_results, model_keys, all_seeds
