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
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
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

    # Define column names for the long-format dataframe
    comparison_col, metric_col, value_col = "comparison", "metric", "value"

    # Split out the existing wide-format column names into comparison and metric components
    comparison_pattern = "|".join(Column.ModelComparisonType)
    metric_pattern = "|".join(Column.ModelComparisonMetric)
    all_metrics_df_long[[comparison_col, metric_col]] = all_metrics_df_long["column"].str.extract(
        rf"^({comparison_pattern})_({metric_pattern})$"
    )

    # Exclude the self-comparison rows which were incuded as metric validation only
    all_metrics_df_long = all_metrics_df_long[
        all_metrics_df_long[comparison_col] != Column.ModelComparisonType.SELF
    ]

    # Plot each metric across comparison types, split by model
    for metric, df_one_metric in all_metrics_df_long.groupby(metric_col):
        num_cols = len(df_one_metric[comparison_col].unique())
        fig, ax = plt.subplots(
            nrows=1,
            ncols=num_cols,
            figsize=(5 * num_cols, 5),
            sharex=True,
            sharey=True,
        )

        for col_idx, (comparison, df_one_metric_one_comparison) in enumerate(
            df_one_metric.groupby(comparison_col)
        ):
            # Add legend to the last subplot only
            add_legend = True if col_idx == num_cols - 1 else False

            sns.violinplot(
                data=df_one_metric_one_comparison,
                x=Column.DATASET,
                y=value_col,
                hue=Column.DiffAEData.MODEL_MANIFEST,
                ax=ax[col_idx],
                split=True,
                legend=add_legend,
            )
            # Get column metadata
            comparison_col_metadata = COLUMN_METADATA[str(comparison)]
            metric_col_metadata = COLUMN_METADATA[str(metric)]
            # Set axis limits where appropriate
            if metric_col_metadata.min is not None and metric_col_metadata.max is not None:
                ax[col_idx].set_ylim(metric_col_metadata.min, metric_col_metadata.max)

            # Set axis labels and title for the subplot
            ax[col_idx].set_ylabel(metric_col_metadata.label)
            ax[col_idx].set_xlabel(COLUMN_METADATA[str(Column.DATASET)].label)
            ax[col_idx].set_title(f"{comparison_col_metadata.name}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
