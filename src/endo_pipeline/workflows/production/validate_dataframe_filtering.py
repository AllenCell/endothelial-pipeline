TAGS = ["validation"]


def main(
    model_manifest_name: str = "diffae_baseline_include_cell_piling",
    run_name: str = "20250918_no_log_norm",
    dataset_name: str = "20250618_20X",
) -> None:
    """Validate filtering of annotated timepoints and positions from a feature dataframe."""
    import logging

    from endo_pipeline.configs import TimepointAnnotation, get_subset_of_timepoint_annotations
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.diffae_dataframe import filter_dataframe_by_annotations
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import ColumnName

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
        "Dataframe contains features for positions [ %s ]", df[ColumnName.POSITION].unique()
    )
    for position, df_pos in df.groupby(ColumnName.POSITION):
        timepoint_min = df_pos[ColumnName.TIMEPOINT].min()
        timepoint_max = df_pos[ColumnName.TIMEPOINT].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info(
        "Validating dataframe with all annotations but [ %s ] removed",
        TimepointAnnotation.NOT_STEADY_STATE,
    )
    timepoint_annotations = get_subset_of_timepoint_annotations(
        annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
    )
    df_keep_non_steady_state = filter_dataframe_by_annotations(
        df, timepoint_annotations=timepoint_annotations
    )
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_keep_non_steady_state[ColumnName.POSITION].unique(),
    )
    for position, df_pos in df_keep_non_steady_state.groupby(ColumnName.POSITION):
        timepoint_min = df_pos[ColumnName.TIMEPOINT].min()
        timepoint_max = df_pos[ColumnName.TIMEPOINT].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info(
        "Validating dataframe with all annotations but [ %s ] removed",
        TimepointAnnotation.CELL_PILING,
    )
    timepoint_annotations = get_subset_of_timepoint_annotations(
        annotations_to_ignore=[TimepointAnnotation.CELL_PILING]
    )
    df_keep_cell_piling = filter_dataframe_by_annotations(
        df, timepoint_annotations=timepoint_annotations
    )
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_keep_cell_piling[ColumnName.POSITION].unique(),
    )
    for position, df_pos in df_keep_cell_piling.groupby(ColumnName.POSITION):
        timepoint_min = df_pos[ColumnName.TIMEPOINT].min()
        timepoint_max = df_pos[ColumnName.TIMEPOINT].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with all annotations removed.")
    df_rm_all = filter_dataframe_by_annotations(df)
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_all[ColumnName.POSITION].unique(),
    )
    for position, df_pos in df_rm_all.groupby(ColumnName.POSITION):
        timepoint_min = df_pos[ColumnName.TIMEPOINT].min()
        timepoint_max = df_pos[ColumnName.TIMEPOINT].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )

    logger.info("Validating dataframe with only technical artifacts removed.")
    timepoint_annotations = get_subset_of_timepoint_annotations(
        annotations_to_ignore=[
            TimepointAnnotation.CELL_PILING,
            TimepointAnnotation.NOT_STEADY_STATE,
        ]
    )
    df_rm_neither = filter_dataframe_by_annotations(df, timepoint_annotations=timepoint_annotations)
    logger.info(
        "Dataframe contains features for positions [ %s ]",
        df_rm_neither[ColumnName.POSITION].unique(),
    )
    for position, df_pos in df_rm_neither.groupby(ColumnName.POSITION):
        timepoint_min = df_pos[ColumnName.TIMEPOINT].min()
        timepoint_max = df_pos[ColumnName.TIMEPOINT].max()
        logger.info(
            "Position [ %s ] has timepoints from [ %s ] to [ %s ]",
            position,
            timepoint_min,
            timepoint_max,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
