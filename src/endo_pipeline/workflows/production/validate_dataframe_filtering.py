TAGS = ["validation"]


def main(
    model_manifest_name: str = "diffae_baseline_include_cell_piling",
    run_name: str = "20250918_no_log_norm",
    dataset_name: str = "20250618_20X",
) -> None:
    """Validate filtering of annotated timepoints and positions from a feature dataframe."""
    import logging

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.diffae_dataframe import (
        remove_annotated_timepoints_and_positions,
    )
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import POSITION_COLUMN_NAME, TIMEPOINT_COLUMN_NAME

    logger = logging.getLogger(__name__)

    # load dataframe manifest for model, get location of dataframe for dataset, and load dataframe
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(model_manifest, run_name)
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)
    df = load_dataframe(dataset_location)

    # log some information about the dataframe before and after filtering

    logger.info("Validating unfiltered dataframe.")
    logger.info(
        "Dataframe contains features for positions [ %s ]", df[POSITION_COLUMN_NAME].unique()
    )
    for position, df_pos in df.groupby(POSITION_COLUMN_NAME):
        timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
        timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with cell piling removed")
    df_rm_cell_piling = remove_annotated_timepoints_and_positions(df, remove_not_steady_state=False)
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_cell_piling[POSITION_COLUMN_NAME].unique(),
    )
    for position, df_pos in df_rm_cell_piling.groupby(POSITION_COLUMN_NAME):
        timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
        timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with non-steady state removed")
    df_rm_not_steady_state = remove_annotated_timepoints_and_positions(df, remove_cell_piling=False)
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_not_steady_state[POSITION_COLUMN_NAME].unique(),
    )
    for position, df_pos in df_rm_not_steady_state.groupby(POSITION_COLUMN_NAME):
        timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
        timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with both cell piling and non-steady state removed")
    df_rm_both = remove_annotated_timepoints_and_positions(df)
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_both[POSITION_COLUMN_NAME].unique(),
    )
    for position, df_pos in df_rm_both.groupby(POSITION_COLUMN_NAME):
        timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
        timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with only technical artifacts removed.")
    df_rm_neither = remove_annotated_timepoints_and_positions(
        df, remove_cell_piling=False, remove_not_steady_state=False
    )
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_neither[POSITION_COLUMN_NAME].unique(),
    )
    for position, df_pos in df_rm_neither.groupby(POSITION_COLUMN_NAME):
        timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
        timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
