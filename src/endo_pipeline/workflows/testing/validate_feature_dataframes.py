from endo_pipeline.cli import PatchType


def main(patch_type: PatchType = "grid_based") -> None:
    """
    Validate positions and timepoints in DiffAE feature dataframes.

    #diffae #pca #cell-centered #grid-based #validation

    For the "filtered" dataframes, we expect all positions and timepoints with
    annotations (except for "NOT_STEADY_STATE") to be excluded from the
    dataframe. Otherwise, we expect all positions and timepoints to be included
    in the dataframe.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-feature-dataframes PATCH_TYPE -d
    ```

    To run the full workflow:

    ```bash
    uv run endopipe validate-feature-dataframes PATCH_TYPE
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run
    validation on the base manifest containing the raw latent features.

    Parameters
    ----------
    patch_type
        Patch type used for model evaluation.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_subset_of_timepoint_annotations,
        get_unannotated_positions,
        get_unannotated_timepoints_for_position,
        load_dataset_config,
    )
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
        FEATURES_UNFILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    manifest_names = [
        (f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{patch_type}", False),
        (FEATURES_UNFILTERED_MANIFEST_NAMES[patch_type], False),
        (FEATURES_FILTERED_MANIFEST_NAMES[patch_type], True),
    ]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Only validating the first two datasets")
        manifest_names = manifest_names[:1]

    for manifest_name, is_filtered in manifest_names:
        logger.info("Validating dataframe manifest '%s'", manifest_name)

        manifest = load_dataframe_manifest(manifest_name)

        # If manifest is filtered, we want only unannotated positions, so pass
        # None. If manifest is not filtered, we want all positions, so pass
        # empty list
        position_annotations: list | None = None if is_filtered else []

        # If manifest is filtered, we want only unannotated positions (ignoring
        # the NOT_STEADY_STATE annotation), so pass the subset of annotations
        # that excludes NOT_STEADY_STATE. If manifest is not filtered, we
        # want all timepoints, so pass empty list.
        timepoint_annotations: list | None = (
            get_subset_of_timepoint_annotations(
                annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
            )
            if is_filtered
            else []
        )

        for dataset_name in manifest.locations.keys():
            # Load dataset config
            dataset_config = load_dataset_config(dataset_name)

            # Load dataframe
            location = manifest.locations[dataset_name]
            df = load_dataframe(location)

            # Check expected positions
            positions_in_df = sorted(df[Column.POSITION].unique())
            expected_positions = sorted(
                get_unannotated_positions(dataset_config, position_annotations)
            )

            if positions_in_df != expected_positions:
                if set(positions_in_df) - set(expected_positions):
                    logger.warning(
                        "Positions in dataframe for '%s' not expected: '%s'",
                        dataset_name,
                        sorted(set(positions_in_df) - set(expected_positions)),
                    )
                if set(expected_positions) - set(positions_in_df):
                    logger.warning(
                        "Expected positions for '%s' not in dataframe: '%s'",
                        dataset_name,
                        sorted(set(expected_positions) - set(positions_in_df)),
                    )

            for position, df_pos in df.groupby(Column.POSITION):
                timepoints_in_df_pos = sorted(df_pos[Column.TIMEPOINT].unique())
                expected_timepoints = sorted(
                    get_unannotated_timepoints_for_position(
                        dataset_config, position, timepoint_annotations
                    )
                )

                if timepoints_in_df_pos != expected_timepoints:
                    if set(timepoints_in_df_pos) - set(expected_timepoints):
                        logger.warning(
                            "Timepoints in dataframe for '%s' position '%s' not expected: '%s'",
                            dataset_name,
                            position,
                            sorted(set(timepoints_in_df_pos) - set(expected_timepoints)),
                        )
                    if set(expected_timepoints) - set(timepoints_in_df_pos):
                        logger.warning(
                            "Expected timepoints for '%s' position '%s' not in dataframe: '%s'",
                            dataset_name,
                            position,
                            sorted(set(expected_timepoints) - set(timepoints_in_df_pos)),
                        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
