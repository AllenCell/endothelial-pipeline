"""Plotting half of the supplementary Model-QC figure pipeline.

Loads the per-model parquets catalogued by the production
``run-model-qc-inference`` workflow (via its
:class:`DataframeManifest`) and renders the Rep-2 Pearson-correlation
bar chart used in the supplementary figure.  No GPU work; runs in
seconds.
"""


def main() -> None:
    """Render the Rep-2 correlation bar chart from the curated dataframe manifest.

    #diffae #figure
    """
    import logging

    from matplotlib import pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import (
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
    )
    from endo_pipeline.library.model.model_qc.results_io import load_results_from_manifest
    from endo_pipeline.library.visualize.model_qc_plots import create_rep2_correlation_bar_plot
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_NAME,
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
    )

    logger = logging.getLogger(__name__)

    dataframe_manifest = load_dataframe_manifest(DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_NAME)
    logger.info(
        "Loaded dataframe manifest [ %s ] with %d per-model parquets.",
        dataframe_manifest.name,
        len(dataframe_manifest.locations),
    )

    all_seed_results, discovered_model_keys, seeds = load_results_from_manifest(dataframe_manifest)
    example_sets_for_metrics = {"rep_2_positions"}

    # Map each discovered ModelKey to its curated short label by positional
    # order in the default sweep.  We require an exact match because the
    # plot only makes sense for the publication 10-model panel.
    sweep_label_map = {
        (m, r): lbl
        for m, r, lbl in zip(
            DEFAULT_MODEL_QC_MANIFEST_NAMES,
            DEFAULT_MODEL_QC_RUN_NAMES,
            DEFAULT_MODEL_QC_LABELS,
            strict=True,
        )
    }
    missing = [
        k for k in discovered_model_keys if (k.manifest_name, k.run_name) not in sweep_label_map
    ]
    if missing:
        raise ValueError(
            "fig-model-qc-plot expects only the curated DEFAULT_MODEL_QC sweep "
            f"models. Unexpected entries: {missing}"
        )

    # Reorder model_keys / labels to follow the curated sweep order
    # (8 BF -> 1024 BF, then CDH5) regardless of the manifest's
    # insertion order.
    discovered = {(k.manifest_name, k.run_name): k for k in discovered_model_keys}
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
    # The supplementary figure renders only the Rep-2 bars, so skip the
    # baseline aggregation entirely.
    baseline_data = compute_baseline_data(all_metrics, compute_baseline=False)
    models_data = build_models_data(all_metrics, model_keys, baseline_data, compute_baseline=False)

    output_path = get_output_path(__file__)
    with plt.style.context("endo_pipeline.figure"):
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
