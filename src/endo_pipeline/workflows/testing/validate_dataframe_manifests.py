from endo_pipeline.cli import UniqueStrList


def main(manifests: UniqueStrList | None = None) -> None:
    """
    Validate dataframe manifests.

    #manifests #validation

    For each dataframe manifest, confirm:

    - All dataframes have at least one non-null location
    - All dataframes can be loaded

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-dataframe-manifest -d
    ```

    To run the workflow for a specific manifest:

    ```bash
    uv run endopipe validate-dataframe-manifest MANIFEST_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on at most two locations for two manifests.

    Parameters
    ----------
    manifest_name
        Name of the dataframe manifest to validate.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.process.progress_bar import ProgressBar
    from endo_pipeline.manifests import get_available_dataframe_manifests, load_dataframe_manifest

    logger = logging.getLogger(__name__)

    manifest_names = manifests or get_available_dataframe_manifests()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two locations for two manifests")
        manifest_names = manifest_names[:2]
        max_locations = 2
    else:
        max_locations = None

    for manifest_name in manifest_names:
        # Load dataframe manifest and location keys
        dataframe_manifest = load_dataframe_manifest(manifest_name)
        location_keys = list(dataframe_manifest.locations.keys())
        if max_locations is not None:
            location_keys = location_keys[:max_locations]

        # Check that the manifest name matches the file name
        if manifest_name != dataframe_manifest.name:
            logger.error(
                "Manifest file name '%s' does not match name field '%s'",
                manifest_name,
                dataframe_manifest.name,
            )

        progress_bar = ProgressBar(location_keys, "Validating", manifest_name)
        for location_key in progress_bar:
            progress_bar.set_iteration_name(location_key)
            location = dataframe_manifest.locations[location_key]

            # Confirm that at least one location in available
            progress_bar.set_step_description("Checking that at least one location is available")
            if location.fmsid is None and location.path is None and location.s3uri is None:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - All locations are null",
                    manifest_name,
                    location_key,
                )
                continue

            # Confirm the dataframe can be loaded
            progress_bar.set_step_description("Checking dataframe can be loaded")
            try:
                load_dataframe(location, delay=True)
            except Exception:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - Unable to load dataframe",
                    manifest_name,
                    location_key,
                )

            if location_key == location_keys[-1]:
                progress_bar.set_step_description("Finished validating all locations")
                progress_bar.clear_iteration_name()
