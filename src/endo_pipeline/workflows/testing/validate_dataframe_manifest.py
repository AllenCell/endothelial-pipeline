def main(manifest_name: str) -> None:
    """
    Validate a given dataframe manifest.

    #manifests #validation

    For the given dataframe manifest, confirm:

    - Each dataset in the manifest has at least one valid location
    - The dataframe can be loaded

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-dataframe-manifest MANIFEST_NAME -vd
    ```

    To run the full workflow:

    ```bash
    uv run endopipe validate-dataframe-manifest MANIFEST_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on the first two datasets in the manifest.

    Parameters
    ----------
    manifest_name
        Name of the dataframe manifest to validate.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import load_dataframe_manifest

    logger = logging.getLogger(__name__)

    dataframe_manifest = load_dataframe_manifest(manifest_name)

    dataset_names = list(dataframe_manifest.locations.keys())

    if DEMO_MODE:
        dataset_names = dataset_names[:2]

    for dataset_name in dataset_names:
        print(f"Running validation for dataset '{dataset_name}'")
        location = dataframe_manifest.locations[dataset_name]

        # Confirm that at least one location in available
        if location.fmsid is None and location.s3uri is None and location.path is None:
            logger.error(
                "Dataset '%s' in dataframe manifest '%s' does not have a location supplied.",
                dataset_name,
                manifest_name,
            )

        # Confirm the dataframe can be loaded
        try:
            load_dataframe(location, delay=True)
        except Exception:
            logger.error(
                "Unable to load dataframe for dataset '%s' in dataframe manifest '%s'",
                dataset_name,
                manifest_name,
            )
