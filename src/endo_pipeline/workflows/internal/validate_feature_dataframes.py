from endo_pipeline.cli import CropPattern
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    model_run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
) -> None:
    """
    Validate that the DiffAE feature dataframes for each dataset contain the
    expected timepoints and positions based on the dataset config.

    For the "filtered" PCA feature dataframes, we expect all positions and
    timepoints with annotations (except for "NOT_STEADY_STATE") to be excluded
    from the dataframe. Else, we expect all positions and timepoints to be
    included in the dataframe.

    Parameters
    ----------
    manifest_name
        The name of the dataframe manifest that specifies the locations of the
        DiffAE feature dataframes to validate.

    """
    import logging

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
        FEATURES_FILTERED_MANIFEST_NAMES,
        FEATURES_UNFILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    base_feature_manifest_name = f"{model_manifest_name}_{model_run_name}_{crop_pattern}"
    pca_feature_manifest_name = FEATURES_UNFILTERED_MANIFEST_NAMES[crop_pattern]
    pca_filtered_feature_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]

    for manifest_name, is_filtered in [
        (base_feature_manifest_name, False),
        (pca_feature_manifest_name, False),
        (pca_filtered_feature_manifest_name, True),
    ]:
        logger.info("Validating dataframe manifest [ %s ]", manifest_name)
        manifest = load_dataframe_manifest(manifest_name)

        # If "filtered" is in the manifest name, want to get all un-annotated
        # positions for each dataset, so pass in None to get_unannotated_positions.
        # If "filtered" is not in the manifest name, want to get all positions
        # for each dataset, so pass in empty list to get_unannotated_positions.
        position_annotations: list | None = None if is_filtered else []

        # If "filtered" is in the manifest name, want to get all un-annotated
        # timepoints for each position ignoring the NOT_STEADY_STATE annotation, so
        # pass in subset of timepoint annotations that excludes NOT_STEADY_STATE to
        # get_unannotated_timepoints_for_position. If "filtered" is not in the
        # manifest name, want to get all timepoints for each position, so pass in
        # empty list to get_unannotated_timepoints_for_position.
        timepoint_annotations: list | None = (
            get_subset_of_timepoint_annotations(
                annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
            )
            if is_filtered
            else []
        )

        for dataset_name in manifest.locations.keys():
            dataset_config = load_dataset_config(dataset_name)

            # load dataframe and check that it has the expected timepoints and positions
            # based on the dataset config

            # If "filtered" is in the manifest name, we expect all position and
            # timepoint annotations (except for "NOT_STEADY_STATE") to be excluded
            # from the dataframe. If "filtered" is not in the manifest name, we
            # expect all position and timepoint annotations to be included in the
            # dataframe.

            location = manifest.locations[dataset_name]
            df = load_dataframe(location)
            positions_in_df = sorted(df[Column.POSITION].unique())
            expected_positions = sorted(
                get_unannotated_positions(dataset_config, position_annotations)
            )

            if positions_in_df != expected_positions:
                logger.warning(
                    "Positions in dataframe for dataset [ %s ] do not match expected positions.",
                    dataset_name,
                )
                if set(positions_in_df) - set(expected_positions):
                    logger.warning(
                        "Positions in dataframe for dataset [ %s ] that are not expected: [ %s ].",
                        dataset_name,
                        sorted(set(positions_in_df) - set(expected_positions)),
                    )
                if set(expected_positions) - set(positions_in_df):
                    logger.warning(
                        "Expected positions for dataset [ %s ] that are not in dataframe: [ %s ].",
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
                    logger.warning(
                        "Timepoints in dataframe for dataset [ %s ], position [ %s ] do not match expected timepoints.",
                        dataset_name,
                        position,
                    )
                    if set(timepoints_in_df_pos) - set(expected_timepoints):
                        logger.warning(
                            "Timepoints in dataframe for dataset [ %s ], position [ %s ] that are not expected: [ %s ].",
                            dataset_name,
                            position,
                            sorted(set(timepoints_in_df_pos) - set(expected_timepoints)),
                        )
                    if set(expected_timepoints) - set(timepoints_in_df_pos):
                        logger.warning(
                            "Expected timepoints for dataset [ %s ], position [ %s ] that are not in dataframe: [ %s ].",
                            dataset_name,
                            position,
                            sorted(set(expected_timepoints) - set(timepoints_in_df_pos)),
                        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
