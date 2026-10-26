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

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.analyze.model_noise_sensitivity import (
        get_bf_frame_intensities_multiproc_wrapper,
        summarize_bf_frame_intensities,
    )
    from endo_pipeline.library.process.general_image_preprocessing import (
        build_analysis_queue,
        process_task_queue,
    )
    from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        save_dataframe_manifest,
    )
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

    # Save the original dataframe to a .parquet file
    demo_suffix = "_demo" if DEMO_MODE else ""

    for dataset_name, df in intensities.groupby(Column.DATASET):
        dataset_name = str(dataset_name)
        # Create save path and save the dataframe for a single dataset to a parquet file
        filename = f"{dataset_name}_log_normd_stdev_stats_per_patch_data{demo_suffix}.parquet"
        save_path = output_path / filename
        df.to_parquet(save_path, index=False)

        # Create location object with output path
        manifest = create_dataframe_manifest(
            f"{GRID_BASED_ST_DEV_PROJ_INTENSITY_COMPARISONS_MANIFEST_NAME}{demo_suffix}", __file__
        )
        location = DataframeLocation()
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            dataset_config = load_dataset_config(dataset_name)
            annotations = build_fms_annotations(dataset_config)
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None
            logger.info("Uploaded log-normd std dev intensities to FMS with id '%s'", fmsid)

        # Add dataframe location to dataframe manifest and save
        manifest.locations[dataset_name] = location
        save_dataframe_manifest(manifest)

    # Summarize and plot the log-normd std dev projection intensities for each dataset
    summary = summarize_bf_frame_intensities(intensities)
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
