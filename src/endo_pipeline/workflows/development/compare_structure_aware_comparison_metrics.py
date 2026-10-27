def main() -> None:
    """
    Compare structure-aware comparison metrics between DiffAE model runs.

    #diffae #model-comparison

    This workflow loads the structure-aware comparison metrics computed by the
    `calculate-structure-aware-comparison-metrics` workflow for the default
    DiffAE model and the CDH5-conditioned model. Make sure to run the
    `calculate-structure-aware-comparison-metrics` workflow before running this
    workflow.

    ## Example usage

    To run the workflow:

    ```bash
    uv run endopipe compare-structure-aware-comparison-metrics
    ```
    """

    import pandas as pd

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        CDH5_CONDITIONED_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_STRUCTURE_AWARE_QC_DATAFRAME_MANIFEST_PREFIX,
    )

    models_to_compare = [DEFAULT_MODEL_MANIFEST_NAME, CDH5_CONDITIONED_MODEL_MANIFEST_NAME]
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load the structure-aware comparison metrics dataframe for each model manifest
    all_metrics = []
    for model_manifest_name in models_to_compare:
        dataframe_manifest = load_dataframe_manifest(
            f"{DEFAULT_STRUCTURE_AWARE_QC_DATAFRAME_MANIFEST_PREFIX}_{model_manifest_name}"
        )
        location = get_dataframe_location_for_dataset(dataframe_manifest, run_name)
        all_metrics.append(load_dataframe(location))

    # Combine into a single dataframe, with model manifest and run name columns identifying rows
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)  # noqa: F841


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
