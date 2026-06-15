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
    """
    Compare DiffAE model performance across models.

    #diffae #model-comparison #visualization

    This workflow uses the per-example metrics precomputed by the
    `calculate-model-comparison-metrics` workflow, so it runs no model
    inference. Run that workflow for each model you want to compare before
    visualizing.

    Visualization outputs include:

    - Bar plots comparing correlation, SSIM, and LPIPS across models, on both
      the validation and rep-2 splits
    - A text summary of the per-model metrics, saved alongside the plots
    - With `--correlation-only`, only the rep-2 correlation bar plot (the view
      used in the supplemental figure)

    ## Example usage

    To compare the default set of models:

    ```bash
    uv run endopipe visualize-diffae-model-comparison
    ```

    To plot only the rep-2 correlation bars:

    ```bash
    uv run endopipe visualize-diffae-model-comparison --correlation-only
    ```

    ## Metrics requirement

    The full comparison reads both the validation and rep-2 splits from the
    metrics manifests; `--correlation-only` reads rep-2 alone.

    Parameters
    ----------
    model_manifest_name
        Model manifest name(s) to compare, one per model. Defaults to the
        curated set of QC models.
    run_name
        Run name(s), one per ``model_manifest_name`` entry. Defaults to the
        curated QC runs.
    correlation_only
        Plot only the rep-2 correlation bar chart instead of the full
        per-metric comparison.
    compute_baseline
        Overlay the next-timepoint baseline reference on the full comparison.
    """
    import logging

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import (
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
    )
    from endo_pipeline.library.model.model_qc.results_io import load_results_from_manifests
    from endo_pipeline.library.visualize.model_qc_plots import (
        create_comparison_plots_and_summary,
        create_rep2_correlation_bar_plot,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
        DEFAULT_MODEL_QC_LABEL_MAP,
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
    # calculate-model-comparison-metrics. Load order does not matter -- models
    # are reordered to requested_pairs below.
    dataframe_manifests = [
        load_dataframe_manifest(f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{mn}")
        for mn in set(manifest_names)
    ]
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
        # Curated short label for sweep models; the model's own label otherwise.
        model_labels = [
            DEFAULT_MODEL_QC_LABEL_MAP.get(pair, key.label)
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
