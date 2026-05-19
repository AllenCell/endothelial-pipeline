"""Plotting half of the supplementary Model-QC figure pipeline.

Loads the per-(model, seed) JSON results written by
``fig-model-qc-inference`` and renders the single Rep-2 Pearson-correlation
bar chart used in the supplementary figure.  No GPU work; runs in seconds.
"""

import logging


def main(inference_run_dir: str | None = None) -> None:
    """Render the Rep-2 correlation bar chart from a prior inference run.

    #diffae #figure

    Parameters
    ----------
    inference_run_dir
        Path to the output directory of a prior ``fig-model-qc-inference``
        run (the directory containing ``inference_manifest.json`` and the
        per-(model, seed) JSON files).  When ``None`` (default), the most
        recent date-stamped output directory of ``fig-model-qc-inference``
        is auto-discovered under the standard results root.
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
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
    )

    logger = logging.getLogger(__name__)

    if inference_run_dir is None:
        run_dir = find_latest_inference_run_dir("model_qc_supp/metrics")
        logger.info("Auto-discovered latest inference run dir: %s", run_dir)
    else:
        run_dir = Path(inference_run_dir)
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
    model_labels = [sweep_label_map[(k.manifest_name, k.run_name)] for k in model_keys]

    all_metrics, _ = aggregate_seed_metrics(
        all_seed_results, model_keys, example_sets_for_metrics, seeds
    )
    # Baseline data is computed for completeness / future use but not drawn.
    baseline_data = compute_baseline_data(all_metrics, compute_baseline=True)
    models_data = build_models_data(
        all_metrics, model_keys, baseline_data, compute_baseline=True
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
