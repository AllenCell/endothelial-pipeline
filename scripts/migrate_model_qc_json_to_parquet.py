"""One-shot migration: convert per-(model, seed) JSON files written by the
old ``fig-model-qc-inference`` workflow into the new long-format parquet
representation, then emit the consolidated ``model_qc_metrics.parquet``.

Usage
-----

    python scripts/migrate_model_qc_json_to_parquet.py \
        results/2026-05-15/fig_model_qc_inference

The JSON files are left in place but moved into a ``_legacy_json/``
subdirectory for provenance; delete that directory once the parquet
artifacts are verified.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from endo_pipeline.library.model.model_qc import (
    ModelKey,
    save_seed_result,
    write_combined_dataframe,
)
from endo_pipeline.library.model.model_qc.serialization import (
    MANIFEST_FILENAME,
    seed_result_path,
)
from endo_pipeline.settings.examples import MODEL_QC_EXAMPLES_REP_2_POSITIONS

logger = logging.getLogger(__name__)

EXAMPLES_BY_SET = {
    "rep_2_positions": MODEL_QC_EXAMPLES_REP_2_POSITIONS,
}


def _json_to_result(payload: dict, model_key: ModelKey) -> dict:
    """Rebuild an ``evaluate_single_model``-shaped dict from JSON payload."""
    result: dict = {
        "model_key": model_key,
        "random_seed": int(payload["random_seed"]),
        "example_set_labels": list(payload["example_set_labels"]),
    }
    for label, bucket in payload["metrics_by_example_set"].items():
        result[label] = {k: list(v) for k, v in bucket.items()}
    return result


def migrate(run_dir: Path) -> None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        sys.exit(f"No {MANIFEST_FILENAME} found in {run_dir}")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != MANIFEST_FILENAME)
    if not json_files:
        sys.exit(f"No legacy JSON files found in {run_dir}")
    logger.info("Converting %d JSON files to parquet", len(json_files))

    converted: list[Path] = []
    for json_path in json_files:
        with json_path.open() as handle:
            payload = json.load(handle)
        model_key = ModelKey(
            payload["model_key"]["manifest_name"],
            payload["model_key"]["run_name"],
        )
        seed = int(payload["random_seed"])
        result = _json_to_result(payload, model_key)
        out = save_seed_result(
            run_dir,
            model_key,
            seed,
            result,
            examples_by_set=EXAMPLES_BY_SET,
        )
        converted.append(out)
        logger.debug("Wrote %s", out.name)

    # Verify every expected parquet now exists.
    for json_path in json_files:
        with json_path.open() as handle:
            payload = json.load(handle)
        model_key = ModelKey(
            payload["model_key"]["manifest_name"], payload["model_key"]["run_name"]
        )
        expected = seed_result_path(run_dir, model_key, int(payload["random_seed"]))
        assert expected.exists(), f"Missing converted parquet: {expected}"

    combined_path = write_combined_dataframe(run_dir)
    logger.info("Wrote combined dataframe: %s", combined_path)

    # Archive JSONs for provenance (do NOT delete -- user can rm later).
    legacy_dir = run_dir / "_legacy_json"
    legacy_dir.mkdir(exist_ok=True)
    for json_path in json_files:
        shutil.move(str(json_path), str(legacy_dir / json_path.name))
    logger.info("Moved %d JSON files into %s", len(json_files), legacy_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Inference output directory to migrate")
    args = parser.parse_args()
    migrate(args.run_dir.resolve())


if __name__ == "__main__":
    main()
