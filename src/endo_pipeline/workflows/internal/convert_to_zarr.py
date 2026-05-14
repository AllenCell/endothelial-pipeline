from pathlib import Path

from endo_pipeline.cli import Datasets, StrList


def main(
    datasets: Datasets | None = None,
    output_path: Path | None = None,
    channel_names: StrList = ["EGFP", "BF"],
) -> None:
    """
    Convert datasets to Zarr format.

    #zarr-conversion #internal #test-ready #cpu-only

    Zarrs are saved in the following channel order: 488, BF, 405, 561, 640

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe convert-to-zarr -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe convert-to-zarr --datasets DATASET_NAME
    ```

    To run the workflow and save converted Zarrs to the internal location:

    ```bash
    export OUTPUT_PATH=//allen/aics/endothelial/morphological_features/image_data/converted_zarrs
    uv run endopipe convert-to-zarr --output-path $OUTPUT_PATH
    ```
    ## Dataset collection

    If datasets are not provided, the workflow will process the
    `live_20X_objective_3i_microscope` dataset collection.

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
    from endo_pipeline.library.process.zarr_conversion import convert_dataset_to_zarr

    logger = logging.getLogger(__name__)

    if datasets is None:
        logger.info(
            "No datasets provided. "
            "Converting datasets in the 'live_20X_objective_3i_microscope' dataset collection."
        )
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    if DEMO_MODE:
        logger.info(
            "Running in DEMO MODE. "
            "Only converting the first 10 timepoints of the first scene in the first dataset."
        )
        output_path = get_output_path("zarr_conversion_demo")
        datasets = datasets[:1]
        max_timepoints = 10
        max_positions = 1
    else:
        max_timepoints = None
        max_positions = None

    if output_path is None:
        logger.info("No output path provided. Using default output path.")
        output_path = get_output_path("zarr_conversion")

    for dataset_name in datasets:
        logger.info(f"Converting dataset: {dataset_name}")

        dataset_config = load_dataset_config(dataset_name)
        convert_dataset_to_zarr(
            dataset_config=dataset_config,
            output_path=output_path,
            channel_names=channel_names,
            max_timepoints=max_timepoints,
            max_positions=max_positions,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
