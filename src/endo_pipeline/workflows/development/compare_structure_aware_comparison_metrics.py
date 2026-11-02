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
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_comparison import (
        calculate_structure_aware_crop_stats,
        load_structure_aware_comparison_metrics,
    )
    from endo_pipeline.library.visualize.model_comparison import (
        plot_structure_aware_comparison_boxplots,
        save_structure_aware_crop_sem_summary,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        CDH5_CONDITIONED_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    models_to_compare = [DEFAULT_MODEL_MANIFEST_NAME, CDH5_CONDITIONED_MODEL_MANIFEST_NAME]
    run_name = DEFAULT_MODEL_RUN_NAME

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
        Column.ModelComparisonMetric.LOG2_FOLD_CHANGE_MEAN_ASPECT_RATIO,
    ]

    # Fix the dataset order so all plotting options are consistent
    dataset_order = ["20250618_20X", "20250319_20X", "20260304_20X", "20250611_20X"]

    output_path = get_output_path(__file__)

    # Load the metrics in long format and summarize each crop's variation across random seeds
    metrics_df = load_structure_aware_comparison_metrics(
        models_to_compare, run_name, comparisons_to_keep, metrics_to_keep
    )
    crop_stats_df = calculate_structure_aware_crop_stats(
        metrics_df, baseline_model_manifest_name=models_to_compare[0]
    )

    # Plot a grid of metrics (rows) by comparison types (columns), with box plots of each crop's
    # mean over seeds for each dataset and model
    plot_structure_aware_comparison_boxplots(
        output_path,
        crop_stats_df,
        models_to_compare,
        run_name,
        comparisons_to_keep,
        metrics_to_keep,
        dataset_order,
    )

    # Save the mean SEM across crops, which summarizes the variation across random seeds
    save_structure_aware_crop_sem_summary(
        output_path, crop_stats_df, models_to_compare, run_name, metrics_to_keep
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
