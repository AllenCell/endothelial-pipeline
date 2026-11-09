def main(background_alpha: float = 0.25) -> None:
    """
    Plot flow switch fixed points and migration coherence over the shear stress summary.

    For each shear stress regime in the `flow_switch` dataset collection (e.g.
    `min_to_max`, `max_to_min`), fixed points colored by migration coherence are
    plotted in the summary plot style. Fixed points are split out by flow
    condition within each dataset, and all datasets in the regime are placed on
    one plot. The summary plot of all datasets in the `shear_stress` dataset
    collection is drawn in the background at low opacity.

    Datasets without bootstrapped fixed points or optical flow features are
    skipped with a warning, and regimes with no remaining datasets are not
    plotted.

    #migration-coherence #fixed-points #visualization #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-flow-switch-summary -d
    ```

    To run the workflow with a more transparent background:

    ```bash
    uv run endopipe plot-flow-switch-summary --background-alpha 0.15
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will limit the
    background to the first three datasets in the `shear_stress` collection and
    plot only the first shear stress regime.

    Parameters
    ----------
    background_alpha
        Opacity of the background summary of the `shear_stress` datasets.
    """

    import logging
    from collections import defaultdict

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
    )
    from endo_pipeline.library.visualize.summary_plot_flow_switch import plot_flow_switch_summaries
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        OPTICAL_FLOW_MANIFEST_NAMES,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    feature_manifest = load_dataframe_manifest(
        FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )
    bootstrap_manifest = load_dataframe_manifest(
        BOOTSTRAPPING_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )
    optical_flow_manifest = load_dataframe_manifest(
        OPTICAL_FLOW_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )

    def has_required_data(dataset_name: str) -> bool:
        """Check that dataset has features, bootstrapped fixed points, and optical flow."""
        missing = [
            name
            for name, manifest in [
                ("features", feature_manifest),
                ("bootstrapped fixed points", bootstrap_manifest),
                ("optical flow features", optical_flow_manifest),
            ]
            if dataset_name not in manifest.locations
        ]
        if missing:
            logger.warning(
                "Dataset [ %s ] is missing %s. Skipping.", dataset_name, " and ".join(missing)
            )
        return not missing

    # Group flow switch datasets by shear stress regime (e.g. "min_to_max")
    datasets_by_regime: dict[str, list[str]] = defaultdict(list)
    for dataset_name in get_datasets_in_collection("flow_switch"):
        if not has_required_data(dataset_name):
            continue
        regime = load_dataset_config(dataset_name).shear_stress_regime
        datasets_by_regime["_to_".join(r.value for r in regime)].append(dataset_name)

    if not datasets_by_regime:
        logger.error("No flow switch datasets with the required data were found.")
        return

    background_datasets = [
        dataset_name
        for dataset_name in get_datasets_in_collection("shear_stress")
        if has_required_data(dataset_name)
    ]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to three background datasets and one regime")
        background_datasets = background_datasets[:3]
        first_regime = next(iter(datasets_by_regime))
        datasets_by_regime = {first_regime: datasets_by_regime[first_regime]}

    column_names = [ColumnName.DiffAEData.POLAR_ANGLE, ColumnName.DiffAEData.POLAR_RADIUS]
    summary_kwargs = {
        "feature_dataframe_manifest": feature_manifest,
        "bootstrap_dataframe_manifest": bootstrap_manifest,
        "column_names": column_names,
        "stable_only": True,
    }

    df_background = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=background_datasets, **summary_kwargs
    )
    if df_background.empty:
        logger.error("No background fixed points were found.")
        return

    output_path = get_output_path(__file__)

    for regime, dataset_names in datasets_by_regime.items():
        df_flow_switch = build_dataframe_for_fixed_point_dataset_summary(
            dataset_names=dataset_names, all_flow_conditions=True, **summary_kwargs
        )
        if df_flow_switch.empty:
            logger.warning("No fixed points found for regime [ %s ]. Skipping.", regime)
            continue

        plot_flow_switch_summaries(
            df_flow_switch,
            df_background,
            output_path=output_path,
            regime_label=regime,
            column_names=column_names,
            background_alpha=background_alpha,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
