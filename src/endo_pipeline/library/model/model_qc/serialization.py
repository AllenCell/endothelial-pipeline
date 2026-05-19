"""Parquet persistence for model QC inference results.

Used to hand off per-(model, seed) evaluation output from the
``fig-model-qc-inference`` workflow to the ``fig-model-qc-plot`` workflow,
so that plotting can run without re-executing GPU diffusion inference.

On disk, each ``(model, seed)`` evaluation is stored as a small long-format
parquet file (one row per example image).  After a run completes, all
per-seed files are concatenated into a single ``model_qc_metrics.parquet``
that serves as the canonical FMS-uploadable artifact.

Schema of the long-format dataframe (per row = one example × seed × model):

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

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from endo_pipeline.settings.examples import ExampleImage

    from .evaluation import ModelKey

logger = logging.getLogger(__name__)


MANIFEST_FILENAME = "inference_manifest.json"
COMBINED_DATAFRAME_FILENAME = "model_qc_metrics.parquet"

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


def seed_result_filename(model_key: "ModelKey", seed: int) -> str:
    """Return the parquet filename used to persist one ``(model_key, seed)`` result."""
    return f"{model_key.manifest_name}__{model_key.run_name}__seed{seed}.parquet"


def seed_result_path(run_dir: Path, model_key: "ModelKey", seed: int) -> Path:
    """Absolute path of the persisted parquet for one ``(model_key, seed)`` result."""
    return Path(run_dir) / seed_result_filename(model_key, seed)


def _aligned_baseline_values(
    metric_values: list[float], baseline_values: list[float], label: str
) -> list[float | None]:
    """Pad ``baseline_values`` to align with ``metric_values`` by index.

    Baselines are appended in the same example loop as metrics, but can be
    skipped on a per-example basis if the next-timepoint image fails to
    load.  In current runs the lists always match length-wise; if they
    don't, we fall back to NaN-padding so the dataframe stays rectangular
    and log a warning so the misalignment is visible.
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
    return padded[: len(metric_values)]


def _result_to_dataframe(
    model_key: "ModelKey",
    seed: int,
    result: dict,
    examples_by_set: dict[str, list["ExampleImage"]],
) -> pd.DataFrame:
    """Flatten one ``evaluate_single_model`` result into a long-format dataframe."""
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

        base_label = f"{model_key.manifest_name}/{model_key.run_name} " \
                     f"seed={seed} set={example_set_label}"
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


def save_seed_result(
    run_dir: Path,
    model_key: "ModelKey",
    seed: int,
    result: dict,
    examples_by_set: dict[str, list["ExampleImage"]] | None = None,
) -> Path:
    """Persist one ``evaluate_single_model`` result dict to parquet.

    Parameters
    ----------
    run_dir
        Directory holding all per-(model, seed) files for one inference run.
    model_key, seed, result
        Identity + payload coming back from :func:`evaluate_single_model`.
    examples_by_set
        Mapping ``{example_set_label: [ExampleImage, ...]}`` used to
        decorate each row with the originating crop's metadata.  Omit when
        the metadata is unavailable (the metric columns will still be
        written; the ``dataset_name``/``position``/... columns will be
        null).
    """
    df = _result_to_dataframe(model_key, seed, result, examples_by_set or {})
    out_path = seed_result_path(run_dir, model_key, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def _dataframe_to_result(
    df: pd.DataFrame, model_key: "ModelKey", seed: int
) -> dict:
    """Inverse of :func:`_result_to_dataframe` for one ``(model, seed)`` slice.

    Reconstructs the dict shape that :func:`aggregate_seed_metrics` expects.
    """
    sub = df[(df["manifest_name"] == model_key.manifest_name)
             & (df["run_name"] == model_key.run_name)
             & (df["random_seed"] == seed)]
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
            bucket["baseline_ssims"] = [
                float(v) for v in rows["baseline_ssim"] if not _is_nan(v)
            ]
            bucket["baseline_lpips"] = [
                float(v) for v in rows["baseline_lpips"] if not _is_nan(v)
            ]
        result[label] = bucket
    return result


def _is_nan(v: object) -> bool:
    try:
        return bool(math.isnan(float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def load_seed_results(
    run_dir: Path,
) -> tuple[dict["ModelKey", dict[int, dict]], list["ModelKey"], list[int]]:
    """Load all persisted ``(model_key, seed)`` results from ``run_dir``.

    Prefers the consolidated ``model_qc_metrics.parquet`` when present;
    otherwise concatenates the per-seed parquet files listed by the
    inference manifest.  Returns the data shaped like ``model_qc.py``
    builds in memory:

    - ``all_seed_results``: ``{ModelKey: {seed: result_dict}}`` where each
      ``result_dict`` matches the shape produced by
      :func:`evaluate_single_model`.
    - ``model_keys``: ordered list of ``ModelKey`` taken from the inference
      manifest (preserves the curated 10-model sweep order).
    - ``seeds``: sorted list of seeds discovered.
    """
    from .evaluation import ModelKey

    manifest = read_inference_manifest(run_dir)
    model_keys = [
        ModelKey(entry["manifest_name"], entry["run_name"]) for entry in manifest["model_keys"]
    ]
    seeds = sorted(int(s) for s in manifest["seeds"])

    df = _load_combined_or_glob(run_dir, model_keys, seeds)

    all_seed_results: dict[ModelKey, dict[int, dict]] = {key: {} for key in model_keys}
    for model_key in model_keys:
        for seed in seeds:
            sub = df[(df["manifest_name"] == model_key.manifest_name)
                     & (df["run_name"] == model_key.run_name)
                     & (df["random_seed"] == seed)]
            if sub.empty:
                logger.warning(
                    "Missing rows for %s/%s seed=%d", model_key.manifest_name,
                    model_key.run_name, seed,
                )
                continue
            all_seed_results[model_key][seed] = _dataframe_to_result(df, model_key, seed)

    return all_seed_results, model_keys, seeds


def _load_combined_or_glob(
    run_dir: Path, model_keys: list["ModelKey"], seeds: list[int]
) -> pd.DataFrame:
    combined = Path(run_dir) / COMBINED_DATAFRAME_FILENAME
    if combined.exists():
        logger.info("Loading combined dataframe: %s", combined)
        return pd.read_parquet(combined)

    frames: list[pd.DataFrame] = []
    for model_key in model_keys:
        for seed in seeds:
            path = seed_result_path(run_dir, model_key, seed)
            if not path.exists():
                logger.warning("Missing seed result file: %s", path)
                continue
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=DATAFRAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_combined_dataframe(run_dir: Path) -> Path:
    """Concatenate all per-(model, seed) parquets into a single FMS-ready file.

    Writes ``<run_dir>/model_qc_metrics.parquet``.  Safe to call multiple
    times (idempotent overwrite).
    """
    manifest = read_inference_manifest(run_dir)
    from .evaluation import ModelKey

    model_keys = [
        ModelKey(entry["manifest_name"], entry["run_name"]) for entry in manifest["model_keys"]
    ]
    seeds = sorted(int(s) for s in manifest["seeds"])

    frames: list[pd.DataFrame] = []
    for model_key in model_keys:
        for seed in seeds:
            path = seed_result_path(run_dir, model_key, seed)
            if not path.exists():
                logger.warning("Skipping missing seed result file: %s", path)
                continue
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No per-seed parquet files found under {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    out_path = Path(run_dir) / COMBINED_DATAFRAME_FILENAME
    df.to_parquet(out_path, index=False)
    logger.info("Wrote combined dataframe (%d rows) to %s", len(df), out_path)
    return out_path


def read_combined_dataframe(run_dir: Path) -> pd.DataFrame:
    """Read the consolidated long-format dataframe written for FMS handoff."""
    return pd.read_parquet(Path(run_dir) / COMBINED_DATAFRAME_FILENAME)


def write_inference_manifest(
    run_dir: Path,
    model_keys: list["ModelKey"],
    seeds: list[int],
    example_set_labels: list[str],
) -> Path:
    """Write the manifest enumerating which ``(model_key, seed)`` files exist."""
    payload = {
        "model_keys": [
            {"manifest_name": k.manifest_name, "run_name": k.run_name} for k in model_keys
        ],
        "seeds": [int(s) for s in seeds],
        "example_set_labels": list(example_set_labels),
    }
    path = Path(run_dir) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def read_inference_manifest(run_dir: Path) -> dict:
    """Read the manifest written by :func:`write_inference_manifest`."""
    path = Path(run_dir) / MANIFEST_FILENAME
    with path.open() as handle:
        return json.load(handle)


def find_latest_inference_run_dir(workflow_stem: str) -> Path:
    """Return the most recent date-stamped output directory for an inference workflow.

    Scans ``<results_root>/<date>/<workflow_stem>/`` (matching the layout
    produced by :func:`endo_pipeline.io.output.get_output_path` with
    ``include_timestamp=True``) and returns the lexicographically largest
    date that contains an ``inference_manifest.json``.  ``workflow_stem``
    may itself contain ``/`` to point at a nested subdirectory
    (e.g. ``"model_qc_supp/metrics"``).
    """
    from endo_pipeline.io.output import get_output_dir

    results_root = get_output_dir()
    candidates = sorted(
        results_root.glob(f"*/{workflow_stem}"),
        key=lambda p: p.relative_to(results_root).parts[0],
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / MANIFEST_FILENAME).exists():
            return candidate
    raise FileNotFoundError(
        f"No inference run with {MANIFEST_FILENAME!r} found under "
        f"{results_root}/*/{workflow_stem}"
    )
