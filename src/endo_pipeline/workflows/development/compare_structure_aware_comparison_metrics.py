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
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
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

    # Define which comparisons to include in the plots
    comparisons_to_keep = [
        Column.ModelComparisonType.BASELINE,
        Column.ModelComparisonType.DENOISED,
        Column.ModelComparisonType.SCRAMBLED_LATENT,
        Column.ModelComparisonType.SCRAMBLED_IMAGE,
    ]
    # Define which metrics to include in the plots
    metrics_to_keep = [
        Column.ModelComparisonMetric.CORRELATION,
        Column.ModelComparisonMetric.MEAN_ORIENTATION_DIFFERENCE,
        Column.ModelComparisonMetric.MEAN_RELATIVE_ASPECT_RATIO,
    ]

    all_metrics_df_long = all_metrics_df_long[
        all_metrics_df_long[comparison_col].isin(comparisons_to_keep)
    ]
    all_metrics_df_long = all_metrics_df_long[all_metrics_df_long[metric_col].isin(metrics_to_keep)]

    # Summarize the variation across random seeds for each crop. Example key is used rather than
    # crop index because crop indices repeat across positions and timepoints.
    crop_mean_col, crop_sem_col = "crop_mean", "crop_sem"
    crop_stats_df = (
        all_metrics_df_long.groupby(
            [
                Column.DiffAEData.MODEL_MANIFEST,
                Column.DATASET,
                Column.EXAMPLE_KEY,
                Column.CROP_INDEX,
                comparison_col,
                metric_col,
            ]
        )[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    crop_stats_df[crop_mean_col] = crop_stats_df["mean"]
    crop_stats_df[crop_sem_col] = crop_stats_df["std"] / np.sqrt(crop_stats_df["count"])

    # Fix the dataset order and model colors so all plotting options are consistent
    dataset_order = ["20250618_20X", "20250319_20X", "20260304_20X", "20250611_20X"]
    palette = dict(
        zip(models_to_compare, sns.color_palette(n_colors=len(models_to_compare)), strict=True)
    )
    hue_offset = 0.2

    def plot_boxes(ax, df_crops, add_legend):
        # Option 1: box plots of each crop's mean over seeds for each dataset and model
        sns.boxplot(
            data=df_crops,
            x=Column.DATASET,
            y=crop_mean_col,
            hue=Column.DiffAEData.MODEL_MANIFEST,
            order=dataset_order,
            hue_order=models_to_compare,
            palette=palette,
            legend=add_legend,
            ax=ax,
        )

    def plot_crop_seed_sems(ax, df_crops, add_legend):
        # Option 2: each crop's mean over seeds with the standard error of the mean across its
        # seeds. Crops are spread out horizontally within each dataset and model.
        offsets = np.linspace(-hue_offset, hue_offset, len(models_to_compare))
        for dataset_idx, dataset in enumerate(dataset_order):
            for model, offset in zip(models_to_compare, offsets, strict=True):
                df_group = df_crops[
                    (df_crops[Column.DATASET] == dataset)
                    & (df_crops[Column.DiffAEData.MODEL_MANIFEST] == model)
                ]
                x = (
                    dataset_idx
                    + offset
                    + np.linspace(-0.8 * hue_offset, 0.8 * hue_offset, len(df_group))
                )
                ax.errorbar(
                    x,
                    df_group[crop_mean_col],
                    yerr=df_group[crop_sem_col],
                    fmt="o",
                    markersize=2,
                    elinewidth=0.5,
                    color=palette[model],
                    alpha=0.3,
                    label=model if dataset_idx == 0 else None,
                )
        ax.set_xticks(range(len(dataset_order)), dataset_order)
        ax.set_xlim(-0.5, len(dataset_order) - 0.5)
        if add_legend:
            ax.legend(title=str(Column.DiffAEData.MODEL_MANIFEST))

    def plot_violins(ax, df_crops, add_legend):
        # Option 3: split violins of each crop's mean over seeds for each dataset and model
        sns.violinplot(
            data=df_crops,
            x=Column.DATASET,
            y=crop_mean_col,
            hue=Column.DiffAEData.MODEL_MANIFEST,
            order=dataset_order,
            hue_order=models_to_compare,
            palette=palette,
            split=True,
            legend=add_legend,
            ax=ax,
        )

    plot_options = {
        "Boxes of crop means": plot_boxes,
        "Per-crop points with SEM": plot_crop_seed_sems,
        "Violins of crop means": plot_violins,
    }

    # Plot a grid of metrics (rows) by comparison types (columns), split by model, with one
    # figure per plotting option
    panel_size = 4
    num_rows, num_cols = len(metrics_to_keep), len(comparisons_to_keep)
    out_path = get_output_path(__file__)
    for option_name, plot_option in plot_options.items():
        fig, ax = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(panel_size * num_cols, panel_size * num_rows),
            sharex=True,
            sharey="row",
            squeeze=False,
        )

        for row_idx, metric in enumerate(metrics_to_keep):
            metric_col_metadata = COLUMN_METADATA[str(metric)]

            for col_idx, comparison in enumerate(comparisons_to_keep):
                axis = ax[row_idx, col_idx]
                axis.axhline(0, color="gray", linestyle="--")

                df_crops = crop_stats_df[
                    (crop_stats_df[metric_col] == metric)
                    & (crop_stats_df[comparison_col] == comparison)
                ]

                # Add legend to the top right subplot only
                add_legend = row_idx == 0 and col_idx == num_cols - 1

                plot_option(axis, df_crops, add_legend)

                # Set axis limits where appropriate
                if metric_col_metadata.min is not None and metric_col_metadata.max is not None:
                    axis.set_ylim(metric_col_metadata.min, metric_col_metadata.max)

                # Label the metric on the first column, the comparison on the top row, and the
                # dataset on the bottom row
                axis.set_ylabel(metric_col_metadata.label if col_idx == 0 else "")
                if row_idx == 0:
                    axis.set_title(COLUMN_METADATA[str(comparison)].name)
                if row_idx == num_rows - 1:
                    axis.set_xlabel(COLUMN_METADATA[str(Column.DATASET)].label)
                    axis.tick_params(axis="x", labelrotation=45)
                else:
                    axis.set_xlabel("")

        save_plot_to_path(
            figure=fig,
            output_path=out_path,
            figure_name=option_name.lower().replace(" ", "_"),
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
