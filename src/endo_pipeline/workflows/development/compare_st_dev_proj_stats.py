from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Compare log-normalized standard deviation projection statistics across datasets.

    Loads the per-frame statistics of the log-normalized brightfield (BF)
    standard deviation projection computed by the `compute-st-dev-proj-stats`
    workflow, summarizes them for each dataset, and plots the summaries to
    compare BF illumination across datasets.

    #diffae #quality-control #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-st-dev-proj-stats -d
    ```

    To run the workflow for select datasets:

    ```bash
    uv run endopipe compare-st-dev-proj-stats --datasets DATASET_NAME_1 DATASET_NAME_2
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on the first two datasets, using the results of the full
    (non-demo) `compute-st-dev-proj-stats` workflow.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    """

    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.model_noise_sensitivity import summarize_bf_frame_intensities
    from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.manifest_names import (
        GRID_BASED_ST_DEV_PROJ_INTENSITY_COMPARISONS_MANIFEST_NAME,
    )

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two datasets")
        dataset_names = dataset_names[:2]

    # Load the log-normd std dev projection intensities for each dataset
    manifest = load_dataframe_manifest(GRID_BASED_ST_DEV_PROJ_INTENSITY_COMPARISONS_MANIFEST_NAME)
    intensities = pd.concat(
        [load_dataframe(manifest.locations[dataset_name]) for dataset_name in dataset_names],
        ignore_index=True,
    )

    # Summarize and plot the log-normd std dev projection intensities for each dataset
    summary = summarize_bf_frame_intensities(intensities)
    plot_cross_dataset_summaries(
        summary,
        output_path=output_path,
        column_names=[Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEAN],
        axis_mode="replicate",
        point_color="black",
        ylabel_rotation=90,
        ylabel_horizontal_alignment="center",
        ylabel_vertical_alignment="bottom",
        yaxis_for_fixed_points=False,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
