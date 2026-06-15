from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.settings.workflow_defaults import DEFAULT_MODEL_COMPARISON_RUNS


def main(
    model_run: list[tuple[str, str]] = DEFAULT_MODEL_COMPARISON_RUNS,
    example_groups: Annotated[
        list[Literal["training", "validation", "replicate"]] | None,
        Parameter(consume_multiple=True, negative_iterable=[]),
    ] = None,
    metrics: Annotated[
        list[Literal["corr", "ssim", "lpips"]] | None,
        Parameter(consume_multiple=True, negative_iterable=[]),
    ] = None,
    include_baseline: Annotated[bool, Parameter(negative="--exclude-baseline")] = True,
) -> None:
    """
    Plot DiffAE model comparison metrics for select model runs.

    #diffae #model-comparison #visualization

    This workflow plots the per-example model comparison metrics computed by the
    `calculate-model-comparison-metrics` workflow for the select model runs.
    Make sure to run the `calculate-model-comparison-metrics` workflow for each
    model manifest and run that you want to plot before running this workflow.

    Outputs include:

    - Bar plot for each selected metric across model runs per example group
    - Text summary of aggregated metrics

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-model-comparion -vd
    ```

    To run the workflow for select model runs:

    ```bash
    uv run endopipe plot-model-comparion \
        --model-run MODEL_MANIFEST_NAME RUN_NAME \
        --model-run MODEL_MANIFEST_NAME RUN_NAME \
        --model-run MODEL_MANIFEST_NAME RUN_NAME
    ```

    To run the workflow for select metrics:

    ```bash
    uv run endopipe plot-model-comparion --metrics METRIC METRIC
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only plot
    the correlation metric.

    Parameters
    ----------
    model_run
        List of model runs as (model_manifest_name, run_name).
    example_groups
        Example groups to include when plotting comparison metrics.
    metrics
        Metrics to include when plotting comparison metrics.
    include_baseline
        True to include baseline metrics, False to exclude.
    """

    import logging

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_comparison import (
        aggregate_model_comparison_metrics,
        group_aggregate_model_comparison_metrics,
        load_model_comparison_metrics,
    )
    from endo_pipeline.library.visualize.model_comparison import (
        plot_model_comparison_bars,
        save_model_comparison_summary,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    plt.style.use("endo_pipeline.figure")

    # Load all metrics and then aggregate by example group and random seed
    all_metrics_df = load_model_comparison_metrics(model_run, example_groups)
    aggregate_df = aggregate_model_comparison_metrics(all_metrics_df)
    aggregate_dict = group_aggregate_model_comparison_metrics(aggregate_df)

    # If running in demo mode, only plot correlation metric
    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one metric")
        metrics = ["corr"]
    else:
        metrics = metrics or ["corr", "ssim", "lpips"]

    # Plot bar plot of metrics
    plot_model_comparison_bars(output_path, model_run, aggregate_dict, metrics, include_baseline)

    # Save summary file
    save_model_comparison_summary(output_path, model_run, aggregate_dict, metrics, include_baseline)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
