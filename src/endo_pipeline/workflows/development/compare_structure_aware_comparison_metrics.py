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
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.workflow_defaults import (
        CDH5_CONDITIONED_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_QC_LABEL_MAP,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_STRUCTURE_AWARE_QC_DATAFRAME_MANIFEST_PREFIX,
    )

    def _get_shear_replicate_label(dataset_name: str) -> str:
        """Get "[shear stress bin] replicate [replicate number]" label for the given dataset."""
        dataset_config = load_dataset_config(dataset_name)
        shear_stress_bin = dataset_config.flow_conditions[-1].shear_stress_bin
        shear_units = COLUMN_METADATA[Column.SHEAR_REPLICATE_NAME].unit
        return f"{shear_stress_bin} {shear_units}\n[repl. {dataset_config.replicate_number}]"

    models_to_compare = [DEFAULT_MODEL_MANIFEST_NAME, CDH5_CONDITIONED_MODEL_MANIFEST_NAME]
    run_name = DEFAULT_MODEL_RUN_NAME
    model_labels = [
        DEFAULT_MODEL_QC_LABEL_MAP[(model, run_name)].replace("\n", "")
        for model in models_to_compare
    ]

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

    # Define column names for the per-crop statistics and the summary across crops
    crop_mean_col, crop_sem_col = "crop_mean", "crop_sem"
    mean_crop_sem_col, unit_col = "mean_crop_sem", "unit"
    mean_crop_sem_fold_change_col = "mean_crop_sem_fold_change"

    output_path = get_output_path(__file__)

    # Fix the dataset order and model colors so all plotting options are consistent
    dataset_order = ["20250618_20X", "20250319_20X", "20260304_20X", "20250611_20X"]
    shear_replicate_order = [_get_shear_replicate_label(dataset) for dataset in dataset_order]

    # Plot a grid of metrics (rows) by comparison types (columns), with box plots of each crop's
    # mean over seeds for each dataset and model
    num_rows, num_cols = len(metrics_to_keep), len(comparisons_to_keep)
    panel_size = MAX_FIGURE_WIDTH / num_cols

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(panel_size * num_cols, panel_size * num_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    sem_summaries = []

    # Style of the horizontal line at each metric's reference value (i.e. identical images)
    reference_line_color, reference_line_style = "black", ":"

    # Plot a subset of the comparison metrics (1 per row)
    for row_idx, metric in enumerate(metrics_to_keep):
        metric_col_metadata = COLUMN_METADATA[str(metric)]
        log_base = metric_col_metadata.log_base

        # Summarize the variation across random seeds for each crop. Example key is used rather
        # than crop index because crop indices repeat across positions and timepoints. Metrics
        # plotted on a log scale (e.g. ratios) are summarized in log space, so that the crop mean
        # is a geometric mean and the SEM is in log fold-change units.
        df_metric = all_metrics_df_long[all_metrics_df_long[metric_col] == metric]
        if log_base is not None:
            df_metric = df_metric.assign(
                **{value_col: np.log(df_metric[value_col]) / np.log(log_base)}
            )
        crop_stats_df = (
            df_metric.groupby(
                [
                    Column.DiffAEData.MODEL_MANIFEST,
                    Column.DATASET,
                    Column.EXAMPLE_KEY,
                    Column.CROP_INDEX,
                    comparison_col,
                ]
            )[value_col]
            .agg(["mean", "sem"])
            .rename(columns={"mean": crop_mean_col, "sem": crop_sem_col})
            .reset_index()
        )
        # Label each row by shear stress and replicate, loading each dataset config only once
        shear_replicate_labels = {
            dataset: _get_shear_replicate_label(dataset)
            for dataset in crop_stats_df[Column.DATASET].unique()
        }
        crop_stats_df[Column.SHEAR_REPLICATE_NAME] = crop_stats_df[Column.DATASET].map(
            shear_replicate_labels
        )
        if log_base is not None:
            crop_stats_df[crop_mean_col] = log_base ** crop_stats_df[crop_mean_col]

        # Summarize the SEM across crops for each model and comparison. For log-scaled metrics,
        # also convert the log fold-change SEM to a fold-change factor (i.e. multiply or divide).
        sem_summary = (
            crop_stats_df.groupby([Column.DiffAEData.MODEL_MANIFEST, comparison_col])[crop_sem_col]
            .mean()
            .rename(mean_crop_sem_col)
            .reset_index()
        )
        sem_summary[metric_col] = metric
        if log_base is not None:
            sem_summary[unit_col] = f"log{log_base:g} fold-change"
            sem_summary[mean_crop_sem_fold_change_col] = log_base ** sem_summary[mean_crop_sem_col]
        else:
            sem_summary[unit_col] = metric_col_metadata.unit
        sem_summaries.append(sem_summary)

        for col_idx, comparison in enumerate(comparisons_to_keep):
            ax = axes[row_idx, col_idx]
            if metric_col_metadata.reference_value is not None:
                ax.axhline(
                    metric_col_metadata.reference_value,
                    color=reference_line_color,
                    linestyle=reference_line_style,
                    zorder=4,  # draw in front of the box plots but behind the legend (zorder 5)
                )

            df_crops = crop_stats_df[crop_stats_df[comparison_col] == comparison]

            # Add legend to the top right subplot only
            add_legend = row_idx == 0 and col_idx == num_cols - 1

            sns.boxplot(
                data=df_crops,
                x=Column.SHEAR_REPLICATE_NAME,
                y=crop_mean_col,
                hue=Column.DiffAEData.MODEL_MANIFEST,
                order=shear_replicate_order,
                hue_order=models_to_compare,
                legend=add_legend,
                log_scale=metric_col_metadata.log_base,
                ax=ax,
            )

            # Replace the model manifest names in the legend with their display labels and add
            # the reference line, which marks the value for identical images
            if add_legend:
                handles = [
                    *ax.get_legend().legend_handles,
                    Line2D([], [], color=reference_line_color, linestyle=reference_line_style),
                ]
                labels = [*model_labels, "Identical\nimages"]
                ax.legend(handles, labels, loc="best", fontsize=FONTSIZE_XSMALL)
            ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_SMALL)

            # Turn on y axis minor ticks. Matplotlib adds no minor ticks to base 2 log axes, so
            # place three minor ticks between each power of the log base instead.
            if log_base is None:
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            else:
                minor_subs = np.linspace(1, log_base, 5)[1:-1].tolist()
                ax.yaxis.set_minor_locator(LogLocator(base=log_base, subs=minor_subs))
                ax.yaxis.set_minor_formatter(NullFormatter())

            # Set axis limits where appropriate
            if metric_col_metadata.min is not None and metric_col_metadata.max is not None:
                ax.set_ylim(metric_col_metadata.min, metric_col_metadata.max)

            # Label the metric on the first column, the comparison on the top row, and the
            # dataset on the bottom row
            ax.set_ylabel(
                metric_col_metadata.label if col_idx == 0 else "", fontsize=FONTSIZE_SMALL
            )
            if row_idx == 0:
                ax.set_title(COLUMN_METADATA[str(comparison)].label, fontsize=FONTSIZE_SMALL)
            if row_idx == num_rows - 1:
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
            else:
                ax.set_xlabel("")

    save_plot_to_path(
        figure=fig,
        output_path=output_path,
        figure_name="boxplots_crop_means",
    )

    # Save the mean SEM across crops, which summarizes the variation across random seeds
    sem_summary_df = pd.concat(sem_summaries, ignore_index=True)[
        [
            metric_col,
            comparison_col,
            Column.DiffAEData.MODEL_MANIFEST,
            mean_crop_sem_col,
            unit_col,
            mean_crop_sem_fold_change_col,
        ]
    ]
    sem_summary_df.to_csv(output_path / "mean_crop_sem_summary.csv", index=False)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
