from endo_pipeline.cli import UniqueStrList


def main(manifests: UniqueStrList | None = None) -> None:
    """
    Validate image manifests.

    #manifests #validation

    For each image manifest, confirm:

    - All images have at least one non-null location
    - All images can be loaded

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-image-manifest -vd
    ```

    To run the workflow for a specific manifest:

    ```bash
    uv run endopipe validate-image-manifest MANIFEST_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on at most two locations of the first manifest. If the location
    has positions and/or timepoint placeholders, only run for a maximum of two
    positions and 10 timepoints.

    Parameters
    ----------
    manifest_name
        Name of the image manifest to validate.
    """

    import logging

    from termcolor import colored
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image
    from endo_pipeline.manifests import (
        get_available_image_manifests,
        get_image_location_for_dataset,
        load_image_manifest,
    )

    logger = logging.getLogger(__name__)

    manifest_names = manifests or get_available_image_manifests()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two locations of the first manifest")
        manifest_names = manifest_names[:1]
        max_locations = 2
        max_positions = 2
        max_timepoints = 10
    else:
        max_locations = None
        max_positions = None
        max_timepoints = None

    for manifest_name in manifest_names:
        # Load image manifest and location keys
        image_manifest = load_image_manifest(manifest_name)
        location_keys = list(image_manifest.locations.keys())
        if max_locations is not None:
            location_keys = location_keys[:max_locations]

        # Check that the manifest name matches the file name
        if manifest_name != image_manifest.name:
            logger.error(
                "Manifest file name '%s' does not match name field '%s'",
                manifest_name,
                image_manifest.name,
            )

        manifest_name_color = colored(manifest_name, color="cyan", attrs=["bold"])
        progress_bar = tqdm(location_keys, desc=f"Validating {manifest_name_color}")

        for location_key in progress_bar:
            progress_bar.set_postfix_str(colored(location_key, color="cyan"))
            location = image_manifest.locations[location_key]

            dataset = load_dataset_config(location_key)

            # Confirm that at least one location in available
            if location.path is None and location.s3uri is None:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - All locations are null",
                    manifest_name,
                    location_key,
                )
                continue

            if (location.path is not None and "{{position}}" in location.path.name) or (
                location.s3uri is not None and "{{position}}" in location.s3uri
            ):
                positions = dataset.zarr_positions
                if max_positions is not None:
                    positions = positions[:max_positions]
            else:
                positions = [None]

            if (location.path is not None and "{{timepoint}}" in location.path.name) or (
                location.s3uri is not None and "{{timepoint}}" in location.s3uri
            ):
                timepoints = list(range(dataset.duration))
                if max_timepoints is not None:
                    timepoints = timepoints[:max_timepoints]
            else:
                timepoints = [None]

            # Confirm the image can be loaded for all placeholders
            for position in tqdm(positions, desc="  Positions", leave=False):
                for timepoint in tqdm(timepoints, desc="  Timepoints", leave=False):
                    location_placeholder = get_image_location_for_dataset(
                        image_manifest, dataset, position=position, timepoint=timepoint
                    )

                    try:
                        load_image(location_placeholder, read=False)
                    except Exception:
                        logger.error(
                            "Validation failed for manifest '%s' key '%s' - Unable to load image",
                            manifest_name,
                            location_key,
                        )
