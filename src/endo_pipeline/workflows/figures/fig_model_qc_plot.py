"""Plotting half of the supplementary Model-QC figure pipeline.

Loads the per-(model, seed) parquets written by ``fig-model-qc-inference``
and renders the single Rep-2 Pearson-correlation bar chart used in the
supplementary figure.  No GPU work; runs in seconds.
"""

import logging


def main(
    inference_run_dir: str | None = None,
    fms_id: str | None = None,
    include_baseline: bool = False,
) -> None:
    """Render the Rep-2 correlation bar chart from a prior inference run.

    #diffae #figure

    Parameters
    ----------
    inference_run_dir
        Path to the output directory of a prior ``fig-model-qc-inference``
        run (the directory containing the per-(model, seed) parquet files
        and the consolidated ``model_qc_metrics.parquet``).  When ``None``
        the next source in the precedence chain is used.
    fms_id
        FMS ID of a previously uploaded ``model_qc_metrics.parquet``.
        Used only when ``inference_run_dir`` is not provided.  Defaults
        to :data:`DEFAULT_MODEL_QC_FMS_ID` so reviewers can regenerate
        the figure without re-running inference.  Pass an empty string
        (``--fms-id ""``) to bypass FMS entirely and force local
        auto-discovery.
    include_baseline
        If ``True``, also compute the next-timepoint baseline statistics
        and attach them to ``models_data`` (the current renderer does not
        draw them but downstream consumers can use them).  Default:
        ``False`` -- skips the baseline aggregation entirely so the plot
        run does no redundant work.
    """
    from pathlib import Path

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import (
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
        find_latest_inference_run_dir,
        load_seed_results,
    )
    from endo_pipeline.library.visualize.model_qc_plots import create_rep2_correlation_bar_plot
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_FMS_ID,
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
    )

    logger = logging.getLogger(__name__)

    # Source precedence:
    #   1. --inference-run-dir (explicit local path)
    #   2. --fms-id (defaults to DEFAULT_MODEL_QC_FMS_ID; pass "" to skip)
    #   3. auto-discover the latest local fig-model-qc-inference output
    if inference_run_dir is not None:
        run_dir: Path = Path(inference_run_dir)
        logger.info("Using explicit inference run dir: %s", run_dir)
    else:
        resolved_fms_id = DEFAULT_MODEL_QC_FMS_ID if fms_id is None else fms_id
        if resolved_fms_id:
            from endo_pipeline.io.fms import get_local_path_from_fmsid

            run_dir = get_local_path_from_fmsid(resolved_fms_id)
            logger.info("Resolved FMS ID %s -> %s", resolved_fms_id, run_dir)
        else:
            run_dir = find_latest_inference_run_dir("model_qc_supp/metrics")
            logger.info("Auto-discovered latest inference run dir: %s", run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Inference run dir does not exist: {run_dir}")

    all_seed_results, model_keys, seeds = load_seed_results(run_dir)
    example_sets_for_metrics = {"rep_2_positions"}

    # Map each ModelKey to its curated short label by positional order in the
    # default sweep.  We require an exact match because the plot only makes
    # sense for the publication 10-model panel.
    sweep_label_map = {
        (m, r): lbl
        for m, r, lbl in zip(
            DEFAULT_MODEL_QC_MANIFEST_NAMES,
            DEFAULT_MODEL_QC_RUN_NAMES,
            DEFAULT_MODEL_QC_LABELS,
            strict=True,
        )
    }
    missing = [k for k in model_keys if (k.manifest_name, k.run_name) not in sweep_label_map]
    if missing:
        raise ValueError(
            "fig-model-qc-plot expects only the curated DEFAULT_MODEL_QC sweep "
            f"models. Unexpected entries: {missing}"
        )

    # Reorder model_keys / labels to follow the curated sweep order
    # (8 BF -> 1024 BF, then CDH5) regardless of the parquet's
    # first-occurrence order.
    discovered = {(k.manifest_name, k.run_name): k for k in model_keys}
    ordered_pairs = [
        (m, r)
        for m, r in zip(DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True)
        if (m, r) in discovered
    ]
    model_keys = [discovered[p] for p in ordered_pairs]
    model_labels = [sweep_label_map[p] for p in ordered_pairs]

    all_metrics, _ = aggregate_seed_metrics(
        all_seed_results, model_keys, example_sets_for_metrics, seeds
    )
    baseline_data = compute_baseline_data(all_metrics, compute_baseline=include_baseline)
    models_data = build_models_data(
        all_metrics, model_keys, baseline_data, compute_baseline=include_baseline
    )

    output_path = get_output_path(__file__)
    create_rep2_correlation_bar_plot(
        models_data=models_data,
        model_labels=model_labels,
        output_path=output_path,
        filename="rep2_correlation_100_noise",
        title="Correlation Analysis",
    )
    logger.info("Saved figure to %s", output_path)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
