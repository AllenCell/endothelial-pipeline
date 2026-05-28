from endo_pipeline.cli import UniqueStrList


def main(manifests: UniqueStrList | None = None) -> None:
    """
    Validate model manifests.

    #manifests #validation

    For each model manifest, confirm:

    - All models have at least one non-null location
    - All models can be loaded

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-model-manifest -vd
    ```

    To run the workflow for a specific manifest:

    ```bash
    uv run endopipe validate-model-manifest MANIFEST_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on at most two locations of the first manifest.

    Parameters
    ----------
    manifest_name
        Name of the model manifest to validate.
    """

    import logging

    from termcolor import colored
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import load_model
    from endo_pipeline.manifests import get_available_model_manifests, load_model_manifest

    logger = logging.getLogger(__name__)

    manifest_names = manifests or get_available_model_manifests()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two locations of the first manifest")
        manifest_names = manifest_names[:1]
        max_locations = 2
    else:
        max_locations = None

    for manifest_name in manifest_names:
        # Load model manifest and location keys
        model_manifest = load_model_manifest(manifest_name)
        location_keys = list(model_manifest.locations.keys())
        if max_locations is not None:
            location_keys = location_keys[:max_locations]

        # Check that the manifest name matches the file name
        if manifest_name != model_manifest.name:
            logger.error(
                "Manifest file name '%s' does not match name field '%s'",
                manifest_name,
                model_manifest.name,
            )

        manifest_name_color = colored(manifest_name, color="cyan", attrs=["bold"])
        progress_bar = tqdm(location_keys, desc=f"Validating {manifest_name_color}")

        for location_key in progress_bar:
            progress_bar.set_postfix_str(colored(location_key, color="cyan"))
            location = model_manifest.locations[location_key]

            # Confirm that at least one location in available
            if (
                location.mlflowid is None
                and location.fmsid is None
                and location.path is None
                and location.s3uri is None
            ):
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - All locations are null",
                    manifest_name,
                    location_key,
                )
                continue

            # Confirm the model can be loaded
            try:
                load_model(location, instantiate=False)
            except Exception:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - Unable to load model",
                    manifest_name,
                    location_key,
                )
