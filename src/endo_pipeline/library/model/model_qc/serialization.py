"""JSON persistence for model QC inference results.

Used to hand off per-(model, seed) evaluation output from the
``fig-model-qc-inference`` workflow to the ``fig-model-qc-plot`` workflow,
so that plotting can run without re-executing GPU diffusion inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluation import ModelKey

logger = logging.getLogger(__name__)


MANIFEST_FILENAME = "inference_manifest.json"


def seed_result_filename(model_key: "ModelKey", seed: int) -> str:
    """Return the JSON filename used to persist one ``(model_key, seed)`` result."""
    return f"{model_key.manifest_name}__{model_key.run_name}__seed{seed}.json"


def seed_result_path(run_dir: Path, model_key: "ModelKey", seed: int) -> Path:
    """Absolute path of the persisted JSON for one ``(model_key, seed)`` result."""
    return Path(run_dir) / seed_result_filename(model_key, seed)


def save_seed_result(run_dir: Path, model_key: "ModelKey", seed: int, result: dict) -> Path:
    """Persist one ``evaluate_single_model`` result dict to JSON.

    The ``ModelKey`` (a ``NamedTuple``) is flattened to ``{manifest_name,
    run_name}`` so the file is portable.  Float metric lists are written
    as-is.
    """
    payload = {
        "model_key": {
            "manifest_name": model_key.manifest_name,
            "run_name": model_key.run_name,
        },
        "random_seed": int(result["random_seed"]),
        "example_set_labels": list(result["example_set_labels"]),
        "metrics_by_example_set": {
            label: {
                metric_name: [float(v) for v in values]
                for metric_name, values in result[label].items()
            }
            for label in result["example_set_labels"]
        },
    }

    out_path = seed_result_path(run_dir, model_key, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def load_seed_results(
    run_dir: Path,
) -> tuple[dict["ModelKey", dict[int, dict]], list["ModelKey"], list[int]]:
    """Load all persisted ``(model_key, seed)`` JSON results from ``run_dir``.

    Returns the data shaped like ``model_qc.py`` builds in memory:

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

    all_seed_results: dict[ModelKey, dict[int, dict]] = {key: {} for key in model_keys}
    for model_key in model_keys:
        for seed in seeds:
            path = seed_result_path(run_dir, model_key, seed)
            if not path.exists():
                logger.warning("Missing seed result file: %s", path)
                continue
            with path.open() as handle:
                payload = json.load(handle)
            result: dict = {
                "model_key": model_key,
                "random_seed": int(payload["random_seed"]),
                "example_set_labels": list(payload["example_set_labels"]),
            }
            for label, bucket in payload["metrics_by_example_set"].items():
                result[label] = {k: list(v) for k, v in bucket.items()}
            all_seed_results[model_key][seed] = result

    return all_seed_results, model_keys, seeds


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

    Scans ``<results_root>/*/<workflow_stem>/`` (matching the layout produced
    by :func:`endo_pipeline.io.output.get_output_path` with
    ``include_timestamp=True``) and returns the lexicographically largest
    date that contains an ``inference_manifest.json``.
    """
    from endo_pipeline.io.output import get_output_dir

    results_root = get_output_dir()
    candidates = sorted(
        results_root.glob(f"*/{workflow_stem}"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / MANIFEST_FILENAME).exists():
            return candidate
    raise FileNotFoundError(
        f"No inference run with {MANIFEST_FILENAME!r} found under {results_root}/*/{workflow_stem}"
    )
