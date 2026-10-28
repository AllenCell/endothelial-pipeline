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

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
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
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Melt into long format with one row per (example, seed, comparison, metric), splitting each
    # metric column name into the comparison type and the metric name
    id_columns = [
        Column.DiffAEData.MODEL_MANIFEST,
        Column.DiffAEData.MODEL_RUN,
        Column.EXAMPLE_KEY,
        Column.RANDOM_SEED,
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        Column.DiffAEData.START_X,
        Column.DiffAEData.START_Y,
    ]
    all_metrics_df_long = all_metrics_df.melt(
        id_vars=id_columns, var_name="column", value_name="value"
    )
    all_metrics_df_long[["comparison", "metric"]] = all_metrics_df_long["column"].str.extract(
        r"^(self|baseline|denoised|scrambled_image|scrambled_latent)_(.+)$"
    )

    # Plot each metric across comparison types, split by model
    for metric, df_one_metric in all_metrics_df_long.groupby("metric"):
        for comparison, df_one_metric_one_comparison in df_one_metric.groupby("comparison"):
            fig, ax = plt.subplots()
            sns.violinplot(
                data=df_one_metric_one_comparison,
                x=Column.DATASET,
                y="value",
                hue=Column.DiffAEData.MODEL_MANIFEST,
                ax=ax,
            )

            ax.set_ylabel(str(metric))
            ax.set_xlabel(Column.DATASET)
            ax.set_title(f"Comparison of {metric} for {comparison}")
        # NOTE: NEED TO UPDATE THE XLIM AND YLIM VALUES IN THE COLUMN METADATA
        # FOR THESE METRICS


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
