"""Plot cross-model DiffAE comparisons from precomputed metrics."""

import logging

from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_QC_MANIFEST_NAMES,
    DEFAULT_MODEL_QC_RUN_NAMES,
)


def main(
    model_manifest_name: list[str] = DEFAULT_MODEL_QC_MANIFEST_NAMES,
    run_name: list[str] = DEFAULT_MODEL_QC_RUN_NAMES,
    correlation_only: bool = False,
    compute_baseline: bool = True,
) -> None:
    r"""
    Plot the cross-model DiffAE comparison from precomputed metrics.

    #diffae #model-comparison

    Plotting-only: consumes the per-manifest metrics emitted by the
    ``calculate-model-comparison-metrics`` workflow (no GPU, no recompute).

    Two breadths, controlled by ``--correlation-only``:

    - **default (full)** — one bar plot per metric (correlation / SSIM /
      LPIPS) across the validation and rep-2 splits, plus the summary table
      (the complete cross-model comparison).
    - ``--correlation-only`` — just the single rep-2 Pearson-correlation bar
      chart (the supplemental figure's panel-B view).

    The single-model qualitative QC (denoising contact sheet + negative
    controls) lives in the ``visualize-diffae-model-performance`` workflow.

    Note: the full comparison needs the validation split, which
    ``calculate-model-comparison-metrics`` only persists after its
    validation-split update -- regenerate the metrics manifests first if they
    predate it. ``--correlation-only`` works with rep-2-only metrics.

    Parameters
    ----------
    model_manifest_name
        Model manifest name(s) to compare. One dataframe manifest is loaded
        per unique name. Defaults to the curated QC sweep.
    run_name
        Run name(s), one per ``model_manifest_name`` entry, selecting which
        run of each manifest to plot. Defaults to the curated QC runs.
    correlation_only
        Plot only the rep-2 correlation bar chart instead of the full
        per-metric comparison.
    compute_baseline
        Overlay the next-timepoint baseline reference (full comparison only).
    """
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import (
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
        create_comparison_plots_and_summary,
    )
    from endo_pipeline.library.model.model_qc.results_io import load_results_from_manifests
    from endo_pipeline.library.visualize.model_qc_plots import create_rep2_correlation_bar_plot
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
        DEFAULT_MODEL_QC_LABELS,
    )

    logger = logging.getLogger(__name__)

    manifest_names = list(model_manifest_name)
    run_names = list(run_name)
    if len(run_names) != len(manifest_names):
        raise ValueError(
            f"Number of run_names ({len(run_names)}) must match "
            f"number of model_manifest_names ({len(manifest_names)})"
        )
    requested_pairs = list(zip(manifest_names, run_names, strict=True))

    # One dataframe manifest per unique manifest_name, produced by
    # calculate-model-comparison-metrics. Preserve first-seen order.
    unique_manifest_names = list(dict.fromkeys(manifest_names))
    dataframe_manifests = [
        load_dataframe_manifest(f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{mn}")
        for mn in unique_manifest_names
    ]
    for dfm in dataframe_manifests:
        logger.info(
            "Loaded dataframe manifest [ %s ] with %d locations.",
            dfm.name,
            len(dfm.locations),
        )

    all_seed_results, discovered_model_keys, seeds = load_results_from_manifests(
        dataframe_manifests
    )

    # Order the requested (manifest, run) pairs, keeping only those present in
    # the loaded metrics so the bars come out in the requested order.
    discovered = {(k.manifest_name, k.run_name): k for k in discovered_model_keys}
    missing = [p for p in requested_pairs if p not in discovered]
    if missing:
        raise ValueError(
            f"Requested models not found in the metrics manifests: {missing}. "
            "Run calculate-model-comparison-metrics for these first."
        )
    model_keys = [discovered[p] for p in requested_pairs]

    # Full comparison needs both splits; correlation-only needs rep-2 alone.
    if correlation_only:
        example_sets_for_metrics = {"rep_2_positions"}
        use_baseline = False
    else:
        example_sets_for_metrics = {"validation_positions", "rep_2_positions"}
        use_baseline = compute_baseline

    all_metrics, _ = aggregate_seed_metrics(
        all_seed_results, model_keys, example_sets_for_metrics, seeds
    )
    baseline_data = compute_baseline_data(all_metrics, use_baseline)
    models_data = build_models_data(all_metrics, model_keys, baseline_data, use_baseline)

    if correlation_only:
        # Curated short labels when a model is part of the default sweep,
        # otherwise the ModelKey's own ``manifest\nrun`` label.
        sweep_label_map = dict(
            zip(
                zip(DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True),
                DEFAULT_MODEL_QC_LABELS,
                strict=True,
            )
        )
        model_labels = [
            sweep_label_map.get(pair, key.label)
            for pair, key in zip(requested_pairs, model_keys, strict=True)
        ]
        output_path = get_output_path("model_qc", "comparison", f"models_{len(model_keys)}")
        create_rep2_correlation_bar_plot(
            models_data=models_data,
            model_labels=model_labels,
            output_path=output_path,
            filename="Model_Comparison_Rep2_Correlation_Bars",
            file_format=".png",
        )
        logger.info("Wrote rep-2 correlation comparison to %s", output_path)
    else:
        create_comparison_plots_and_summary(
            models_data, model_keys, seeds, baseline_data, use_baseline
        )
        logger.info("Wrote full cross-model comparison plots + summary table.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
