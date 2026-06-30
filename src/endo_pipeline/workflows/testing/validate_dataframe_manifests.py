from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import UniqueStrList


def main(
    manifests: UniqueStrList | None = None,
    staging_only: Annotated[bool, Parameter(negative="--all-available")] = True,
) -> None:
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

    To run the workflow for all available manifests:

    ```bash
    uv run endopipe validate-dataframe-manifest --all-available
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on at most two locations for two manifests.

    Parameters
    ----------
    manifest_name
        Name of the dataframe manifest to validate.
    staging_only
        True to only validate dataframe manifests valid for staging, False to
        validate all available dataframe manifests.
    """

    import logging
    import re

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_available_dataset_names
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.process.progress_bar import ProgressBar
    from endo_pipeline.manifests import get_available_dataframe_manifests, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.manifest_staging import STAGING_DATAFRAME_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    available_dataset_names = get_available_dataset_names()
    manifest_names = manifests or (
        STAGING_DATAFRAME_MANIFEST_NAMES if staging_only else get_available_dataframe_manifests()
    )

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two locations for two manifests")
        manifest_names = manifest_names[:2]
        max_locations = 2
    else:
        max_locations = None

    extra_nuclei_columns = [
        *[ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP % i for i in range(7)],
        *[ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP_CENTROID_X % i for i in range(7)],
        *[ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP_CENTROID_Y % i for i in range(7)],
    ]
    optional_ann_columns = [
        Column.Annotations.BF_SCOPE_ERROR,
        Column.Annotations.BF_TEMP_ARTIFACT,
        Column.Annotations.GFP_SCOPE_ERROR,
        Column.Annotations.CELL_PILING,
        Column.Annotations.NOT_STEADY_STATE,
        Column.Annotations.UNFED,
        Column.Annotations.XY_SHIFT,
        Column.Annotations.Z_SHIFT,
    ]
    optional_gfp_columns = [
        Column.Annotations.AUTO_GFP_SCOPE_ERROR,
        Column.Annotations.GFP_ROLLING_MEDIAN,
        Column.Annotations.GFP_LOWER_THRESHOLD,
        Column.Annotations.GFP_UPPER_THRESHOLD,
        Column.Annotations.GFP_TIMEPOINT_MEANS,
        Column.Annotations.GFP_DARK_OUTLIERS,
        Column.Annotations.GFP_BRIGHT_OUTLIERS,
    ]
    okay_if_missing_columns = {
        "nuclei_labelfree_segmentation": set(extra_nuclei_columns),
        "cell_centered_features_filtered": set(extra_nuclei_columns + optional_ann_columns),
        "cell_centered_features_unfiltered": set(extra_nuclei_columns + optional_ann_columns),
        "merged_segmentation_features": set(extra_nuclei_columns + optional_ann_columns),
        "timepoint_outlier_annotations": set(optional_gfp_columns),
    }

    for manifest_name in manifest_names:
        # Load dataframe manifest and location keys
        dataframe_manifest = load_dataframe_manifest(manifest_name)
        location_keys = list(dataframe_manifest.locations.keys())
        if max_locations is not None:
            location_keys = location_keys[:max_locations]

        # Get list of expected columns
        expected_columns = set(dataframe_manifest.columns.keys())

        # Check that the manifest name matches the file name
        if manifest_name != dataframe_manifest.name:
            logger.error(
                "Manifest file name '%s' does not match name field '%s'",
                manifest_name,
                dataframe_manifest.name,
            )

        # For dataset location keys, confirm the dataset config is available
        for location_key in location_keys:
            if not re.match(r"20[0-9]{6}_", location_key):
                continue

            if location_key not in available_dataset_names:
                logger.error(
                    "Manifest '%s' contains dataset '%s' that does not have dataset config",
                    manifest_name,
                    location_key,
                )
                location_keys.remove(location_key)

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

            # Confirm that columns match expected columns listed in manifest
            progress_bar.set_step_description("Checking dataframe columns are consistent")
            columns = set(load_dataframe(location, delay=True).columns)
            if columns - expected_columns:
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - Extra column(s): %s",
                    manifest_name,
                    location_key,
                    columns - expected_columns,
                )
            if expected_columns - columns - okay_if_missing_columns.get(manifest_name, set()):
                logger.error(
                    "Validation failed for manifest '%s' key '%s' - Missing column(s): %s",
                    manifest_name,
                    location_key,
                    expected_columns - columns,
                )

            if location_key == location_keys[-1]:
                progress_bar.set_step_description("Finished validating all locations")
                progress_bar.clear_iteration_name()
