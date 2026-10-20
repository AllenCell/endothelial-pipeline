from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Visualize noise statistics from feature displacements at steady state.

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-noise-statistics -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-noise-statistics --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Prerequisites

    This workflow loads the noise correlation and measurement noise covariance
    statistics, the full two-timepoint noise correlation functions, and the
    residual periodograms from the manifests produced by
    `compute-noise-statistics`, so that workflow must be run first for the
    selected `patch_type`.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only
    visualize the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    datasets
        List of datasets or dataset collections to compute correlations for.
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, slugify
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_noise_correlation_vs_lag,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.manifest_names import NOISE_STATISTICS_MANIFEST_NAME_PREFIXES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning("DEMO MODE - Limiting to %d dataset(s)", len(dataset_names))

    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Load manifests produced by compute-noise-statistics: the tidy noise
    # correlation / measurement noise covariance statistics dataframe, the full
    # two-timepoint noise correlation functions, and the residual periodograms.
    name_suffix = join_sorted_strings(column_names)
    noise_statistics_manifest_name = (
        f"{NOISE_STATISTICS_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_statistics_manifest = load_dataframe_manifest(noise_statistics_manifest_name)

    for dataset_name in dataset_names:
        if dataset_name not in noise_statistics_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in the noise statistics manifest. Skipping.",
                dataset_name,
            )
            continue
        print("\n Processing dataset:", dataset_name)
        dataset_config = load_dataset_config(dataset_name)
        noise_statistics_dataframe = load_dataframe(
            noise_statistics_manifest.locations[dataset_name], delay=False
        )

        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            # noise correlation / measurement noise summary statistics for this
            # flow condition, loaded from compute-noise-statistics
            noise_statistics_flow = noise_statistics_dataframe[
                noise_statistics_dataframe[Column.SHEAR_STRESS] == flow_condition.shear_stress
            ]
            plot_noise_correlation_vs_lag(
                noise_statistics_dataframe=noise_statistics_flow,
                column_names=column_names,
                plot_title=f"Noise correlation vs lag: {dataset_name_flow}",
                output_path=output_path,
                file_name=slugify(f"noise_correlation_vs_lag_{dataset_name_flow}"),
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
