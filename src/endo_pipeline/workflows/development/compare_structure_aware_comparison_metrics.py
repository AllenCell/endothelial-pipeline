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

    def add_dataset_position_timepoint_from_example_key(metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add dataset, position, and timepoint columns parsed from the example key.

        Temporary until the metrics are recalculated with these columns included at creation.
        Example keys are formatted as `{dataset_name}_P{position}_T{timepoint}_X{x}_Y{y}`
        (see `ExampleImage.__str__`).

        Parameters
        ----------
        metrics_df
            Dataframe containing an example key column.

        Returns
        -------
        :
            Copy of the dataframe with dataset, position, and timepoint columns added.
        """

        example_key_parts = metrics_df[Column.EXAMPLE_KEY].str.extract(
            r"^(?P<dataset>.+)_P(?P<position>\d+)_T(?P<timepoint>\d+)_X\d+_Y\d+$"
        )

        return metrics_df.assign(
            **{
                Column.DATASET: example_key_parts["dataset"],
                Column.POSITION: example_key_parts["position"].astype(int),
                Column.TIMEPOINT: example_key_parts["timepoint"].astype(int),
            }
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

    # Add dataset, position, and timepoint columns parsed from the example key
    all_metrics_df = add_dataset_position_timepoint_from_example_key(all_metrics_df)

    # Rename columns saved under old names to match the current column names
    all_metrics_df = all_metrics_df.rename(
        columns={
            "scrambled_image_correlation": Column.MODEL_COMPARISON_SCRAMBLED_IMAGE_CORRELATION,
            "scrambled_latent_correlation": Column.MODEL_COMPARISON_SCRAMBLED_LATENT_CORRELATION,
        }
    )

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
    # Temporary until the metrics are recalculated with crop index and start x/y columns included
    id_columns = [column for column in id_columns if column in all_metrics_df.columns]
    all_metrics_df_long = all_metrics_df.melt(
        id_vars=id_columns, var_name="column", value_name="value"
    )
    all_metrics_df_long[["comparison", "metric"]] = all_metrics_df_long["column"].str.extract(
        r"^(self|baseline|denoised|scrambled_image|scrambled_latent)_(.+)$"
    )

    # Plot each metric across comparison types, split by model
    for metric, df_one_metric in all_metrics_df_long.groupby("metric"):
        plt.figure()
        sns.boxplot(
            data=df_one_metric, x="comparison", y="value", hue=Column.DiffAEData.MODEL_MANIFEST
        )
        plt.title(str(metric))


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
