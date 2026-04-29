def main(manifest_name: str) -> None:
    """
    Validate that the PCA feature dataframes for each dataset contain the
    expected timepoints and positions based on the dataset config.

    If "filtered" is in the manifest name, we expect all positions and
    timepoints with annotations (except for "NOT_STEADY_STATE") to be excluded
    from the dataframe. If "filtered" is not in the manifest name, we expect all
    positions and timepoints to be included in the dataframe.

    Parameters
    ----------
    manifest_name
        The name of the dataframe manifest that specifies the locations of the
        PCA feature dataframes to validate.

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

    logger = logging.getLogger(__name__)

    manifest = load_dataframe_manifest(manifest_name)

    is_filtered = "filtered" in manifest_name

    # If "filtered" is in the manifest name, want to get all un-annotated
    # positions for each dataset, so pass in None to get_unannotated_positions.
    # If "filtered" is not in the manifest name, want to get all positions
    # for each dataset, so pass in empty list to get_unannotated_positions.
    position_annotations = None if is_filtered else []

    # If "filtered" is in the manifest name, want to get all un-annotated
    # timepoints for each position ignoring the NOT_STEADY_STATE annotation, so
    # pass in subset of timepoint annotations that excludes NOT_STEADY_STATE to
    # get_unannotated_timepoints_for_position. If "filtered" is not in the
    # manifest name, want to get all timepoints for each position, so pass in
    # empty list to get_unannotated_timepoints_for_position.
    timepoint_annotations = (
        get_subset_of_timepoint_annotations(
            exclude_annotations=[TimepointAnnotation.NOT_STEADY_STATE]
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
        expected_positions = sorted(get_unannotated_positions(dataset_config, position_annotations))

        if positions_in_df != expected_positions:
            logger.warning(
                "Positions in dataframe for dataset [ %s ] do not match expected positions. "
                "Positions in dataframe: [ %s ]. Expected positions: [ %s ].",
                dataset_name,
                positions_in_df,
                expected_positions,
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
                    "Timepoints in dataframe for dataset [ %s ], position [ %s ] do not match expected timepoints. "
                    "Timepoints in dataframe: [ %s ]. Expected timepoints: [ %s ].",
                    dataset_name,
                    position,
                    timepoints_in_df_pos,
                    expected_timepoints,
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
