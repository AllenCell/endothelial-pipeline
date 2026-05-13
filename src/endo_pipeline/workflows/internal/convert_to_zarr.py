from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    output_path: str | None = None,
    channel_names: list[str] = ["EGFP", "BF"],
) -> None:
    """
    Convert datasets to Zarr format.

    #internal #test-ready #cpu-only

    Zarrs are saved in the following channel order: 488, BF, 405, 561, 640

    ## Dataset collection

    If datasets are not provided, the workflow will process the
    `live_20X_objective_3i_microscope` dataset collection.

    ## Example usage

    ```bash
    # run workflow in demo mode
    endopipe convert-to-zarr -vd

    # run workflow for single dataset
    endopipe convert-to-zarr --datasets 20250224_20X

    # run workflow to converted files to program location
    export OUTPUT_PATH=//allen/aics/endothelial/morphological_features/image_data/converted_zarrs
    endopipe convert-to-zarr --output-path $OUTPUT_PATH
    ```

    ## Workflow demo

    The ``--demo-mode`` (``-d``) flag can be used to run a simplified version of
    this workflow for testing purposes (e.g. during code review). The workflow
    will only process the first 10 timepoints of the first position of the first
    dataset. The output will be saved to `zarr_conversion_demo` even if an
    output path is provided.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to convert to Zarr.
    output_path
        Path to save the converted Zarr files.
    channel_names
        List of channel names to include in the Zarr files.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.convert_to_zarr.convert_dataset import convert_dataset

    logger = logging.getLogger(__name__)

    if datasets is None:
        logger.info(
            "No datasets provided. "
            "Converting datasets in the 'live_20X_objective_3i_microscope' dataset collection."
        )
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    if DEMO_MODE:
        logger.info("Running in DEMO MODE. Only converting the first dataset.")
        output_path = get_output_path("zarr_conversion_demo").as_posix()
        datasets = datasets[:1]
        max_timepoints = 10
        max_positions = 1
    else:
        max_timepoints = None
        max_positions = None

    if output_path is None:
        logger.info("DEMO_MODE is ON or no output path provided. Using default output path.")
        output_path = get_output_path("zarr_conversion").as_posix()

    for dataset_name in datasets:
        logger.info(f"Converting dataset: {dataset_name}")

        dataset_config = load_dataset_config(dataset_name)
        convert_dataset(
            dataset_config=dataset_config,
            output_path=output_path,
            channel_names=channel_names,
            max_timepoints=max_timepoints,
            max_positions=max_positions,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
