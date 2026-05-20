"""Parquet persistence for model QC inference results.

The supplementary Model-QC figure pipeline persists the per-(model, seed)
evaluation output to disk so the companion plot workflow can render the
bar chart without re-running GPU diffusion inference.

On-disk layout
--------------

- One **per-model** parquet per ``(manifest_name, run_name)`` pair, named
  ``<manifest>__<run>.parquet``.  Each file holds rows for every seed
  evaluated for that model.  The set of per-model parquets is what gets
  enumerated in a :class:`DataframeManifest` and (optionally) uploaded
  to FMS.

- A transient ``shards/`` subdirectory holding per-seed parquets named
  ``<manifest>__<run>__seed<N>.parquet``.  Shards are the finest-grained
  resume unit: the inference workflow skips any seed whose shard already
  exists.  Once every seed for a given model finishes, the shards are
  concatenated into the per-model parquet and can be deleted.  Shards
  never enter the ``DataframeManifest``.

Long-format schema (per row = one example x seed x model):

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


SHARDS_SUBDIR = "shards"

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


def model_key_str(model_key: "ModelKey") -> str:
    """Return the canonical ``manifest__run`` string used in filenames and manifest keys."""
    return f"{model_key.manifest_name}__{model_key.run_name}"


def model_result_filename(model_key: "ModelKey") -> str:
    """Per-model parquet filename (one file per ``(manifest, run)`` pair)."""
    return f"{model_key_str(model_key)}.parquet"


def model_result_path(out_dir: Path, model_key: "ModelKey") -> Path:
    """Absolute path of the per-model parquet under ``out_dir``."""
    return Path(out_dir) / model_result_filename(model_key)


def shard_filename(model_key: "ModelKey", seed: int) -> str:
    """Per-(model, seed) shard filename used for resume granularity."""
    return f"{model_key_str(model_key)}__seed{seed}.parquet"


def shard_path(out_dir: Path, model_key: "ModelKey", seed: int) -> Path:
    """Absolute path of a per-seed shard under ``out_dir / SHARDS_SUBDIR``."""
    return Path(out_dir) / SHARDS_SUBDIR / shard_filename(model_key, seed)


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


def save_shard(
    out_dir: Path,
    model_key: "ModelKey",
    seed: int,
    result: dict,
    examples_by_set: dict[str, list["ExampleImage"]] | None = None,
) -> Path:
    """Persist one ``(model_key, seed)`` ``evaluate_single_model`` result to a shard parquet.

    Returns
    -------
    :
        Path of the shard file written under ``out_dir / SHARDS_SUBDIR``.
    """
    df = _result_to_dataframe(model_key, seed, result, examples_by_set or {})
    out_path = shard_path(out_dir, model_key, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def write_model_parquet_from_shards(
    out_dir: Path,
    model_key: "ModelKey",
    seeds: list[int],
) -> Path:
    """Concatenate the per-seed shards for one model into its per-model parquet.

    Raises ``FileNotFoundError`` if any expected shard is missing.

    Returns
    -------
    :
        Path of the per-model parquet just written under ``out_dir``.
    """
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        sp = shard_path(out_dir, model_key, seed)
        if not sp.exists():
            raise FileNotFoundError(
                f"Missing shard for {model_key_str(model_key)} seed={seed}: {sp}"
            )
        frames.append(pd.read_parquet(sp))
    df = pd.concat(frames, ignore_index=True)
    out_path = model_result_path(out_dir, model_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(
        "Wrote per-model parquet (%d rows, %d seeds) to %s",
        len(df),
        len(seeds),
        out_path,
    )
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
    """Load per-(model, seed) results from a :class:`DataframeManifest`.

    Each entry in ``manifest.locations`` is expected to point at a
    per-model parquet (one file per ``(manifest_name, run_name)`` pair)
    written by the production inference workflow.  Locations are
    resolved via :func:`endo_pipeline.io.load_dataframe`, so files can
    come from FMS, local path, or S3.

    Returns
    -------
    all_seed_results
        ``{ModelKey: {seed: result_dict}}`` reshaped to match the in-memory
        layout produced by :func:`evaluate_single_model`.
    model_keys
        Model keys in the order they appear in ``manifest.locations``.
    seeds
        Sorted union of seeds found across all loaded per-model parquets.
    """
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.model.model_qc.evaluation import ModelKey

    all_seed_results: dict[ModelKey, dict[int, dict]] = {}
    model_keys: list[ModelKey] = []
    all_seeds: set[int] = set()

    for key, location in manifest.locations.items():
        df = load_dataframe(location)
        if df.empty:
            logger.warning("Empty per-model parquet for key %r (location=%r)", key, location)
            continue

        manifest_names = df["manifest_name"].unique().tolist()
        run_names = df["run_name"].unique().tolist()
        if len(manifest_names) != 1 or len(run_names) != 1:
            raise ValueError(
                f"Per-model parquet for key {key!r} must contain exactly one "
                f"(manifest_name, run_name) pair; got "
                f"manifest_names={manifest_names}, run_names={run_names}"
            )
        model_key = ModelKey(str(manifest_names[0]), str(run_names[0]))
        model_keys.append(model_key)

        seeds_for_model = sorted(int(s) for s in df["random_seed"].unique())
        all_seeds.update(seeds_for_model)
        all_seed_results[model_key] = {
            seed: _dataframe_to_result(df, model_key, seed) for seed in seeds_for_model
        }

    if not model_keys:
        raise FileNotFoundError(
            f"Dataframe manifest {manifest.name!r} resolved to zero usable per-model parquets."
        )

    return all_seed_results, model_keys, sorted(all_seeds)
