from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, max_workers: int | None = None) -> None:
    """
    Compare brightfield intensities across datasets.

    Uses the mean brightfield (BF) intensity over XY for each z-slice and
    timepoint, as computed for the timepoint outlier annotations, to compare BF
    illumination across datasets. Intensities are averaged over z-slices to get
    a single intensity per frame.

    #diffae #quality-control #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-brightfield-intensities -d
    ```

    To run the workflow for select datasets:

    ```bash
    uv run endopipe compare-brightfield-intensities --datasets DATASET_NAME_1 DATASET_NAME_2
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on the first two datasets.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    max_workers
        Maximum number of worker processes to use for processing. If None, defaults to 1.
    """

    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.model_noise_sensitivity import (
        get_bf_frame_intensities_multiproc_wrapper,
        summarize_bf_frame_intensities,
    )
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
    from endo_pipeline.settings.column_names import ColumnName as Column

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two datasets")
        dataset_names = dataset_names[:2]

    # Build analysis queue for multiprocessing
    analysis_queue = build_analysis_queue(
        dataset_names=dataset_names,
        out_dir=output_path,
        image_validation_frequency=None,
        t_start=0,
        t_final=50 if DEMO_MODE else None,
        max_positions=2 if DEMO_MODE else None,
    )

    # Get log-normd std dev projection intensities at each timepoint (with multiprocessing)
    intensity_dfs: list[dict] = process_task_queue(  # type: ignore
        get_bf_frame_intensities_multiproc_wrapper,
        analysis_queue,
        description="Computing log-normd std dev projection intensities",
        num_processes=max_workers or 1,  # Use specified number of processes or default to 1,
        chunksize=1,
    )
    intensities = pd.DataFrame(intensity_dfs)

    # Save summary statistics of BF intensities for each dataset
    summary = summarize_bf_frame_intensities(intensities)
    summary.to_csv(output_path / "bf_intensity_summary.csv", index=False)

    # Plot BF intensity summary for each dataset
    plot_cross_dataset_summaries(
        summary,
        output_path=output_path,
        column_names=[Column.ModelNoiseSensitivity.LOG_NORMD_STDEV_MEAN],
        point_color="black",
        ylabel_rotation=90,
        ylabel_horizontal_alignment="center",
        ylabel_vertical_alignment="bottom",
        yaxis_for_fixed_points=False,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
