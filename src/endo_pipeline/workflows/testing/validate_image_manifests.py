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
    validation on at most two locations for two manifests. If the location
    has positions and/or timepoint placeholders, only run for a maximum of two
    positions and 10 timepoints.

    Parameters
    ----------
    manifest_name
        Name of the image manifest to validate.
    """

    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image
    from endo_pipeline.library.process.progress_bar import ProgressBar
    from endo_pipeline.manifests import (
        get_available_image_manifests,
        get_image_location_for_dataset,
        load_image_manifest,
    )

    logger = logging.getLogger(__name__)

    manifest_names = manifests or get_available_image_manifests()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two locations for two manifests")
        manifest_names = manifest_names[:2]
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

        progress_bar = ProgressBar(location_keys, "Validating", manifest_name)
        for location_key in progress_bar:
            progress_bar.set_iteration_name(location_key)
            location = image_manifest.locations[location_key]

            dataset = load_dataset_config(location_key)

            # Calculate expected pixel size
            expected_pixel_size = dataset.pixel_size_xy_in_um
            if "grid_seg" in manifest_name:
                expected_pixel_size *= 2

            # Confirm that at least one location in available
            progress_bar.set_step_description("Checking that at least one location is available")
            if location.path is None and location.s3uri is None:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - All locations are null",
                    manifest_name,
                    location_key,
                )
                continue

            # Check if there are any position placeholders to iterate through
            positions: list[int] | list[None] = [None]
            if (location.path is not None and "{{position}}" in location.path.name) or (
                location.s3uri is not None and "{{position}}" in location.s3uri
            ):
                positions = dataset.zarr_positions
                if max_positions is not None:
                    positions = positions[:max_positions]

            # Check if there are any timepoint placeholders to iterate through
            timepoints: list[int] | list[None] = [None]
            if (location.path is not None and "{{timepoint}}" in location.path.name) or (
                location.s3uri is not None and "{{timepoint}}" in location.s3uri
            ):
                timepoints = list(range(dataset.duration))
                if max_timepoints is not None:
                    timepoints = timepoints[:max_timepoints]

            # Confirm the image can be loaded for all placeholders
            progress_bar.set_step_description("Checking image can be loaded for all placeholders")
            pixel_size_mismatchs = set()
            for position in tqdm(positions, desc="  Positions", leave=False):
                for timepoint in tqdm(timepoints, desc="  Timepoints", leave=False):
                    location_placeholder = get_image_location_for_dataset(
                        image_manifest, dataset, position=position, timepoint=timepoint
                    )

                    try:
                        image = load_image(location_placeholder, read=False)

                        # Check for any mismatches in pixel size
                        if timepoint == 0 or timepoint is None:
                            if not round(image.physical_pixel_sizes.X, 3) == expected_pixel_size:
                                pixel_size_mismatchs.add(position)
                            if not round(image.physical_pixel_sizes.Y, 3) == expected_pixel_size:
                                pixel_size_mismatchs.add(position)
                    except Exception:
                        logger.error(
                            "Validation failed for manifest '%s' key '%s' - Unable to load image",
                            manifest_name,
                            location_key,
                        )

            if any(pixel_size_mismatchs):
                logger.error(
                    "Manifest '%s' key '%s' pixel sizes in image metadata and dataset config "
                    "do not match for positions: %s",
                    manifest_name,
                    location_key,
                    sorted(pixel_size_mismatchs),
                )

            if location_key == location_keys[-1]:
                progress_bar.set_step_description("Finished validating all locations")
                progress_bar.clear_iteration_name()
