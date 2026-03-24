from typing import Literal

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.manifests.model_manifest_utils import get_feature_dataframe_manifest_name
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

TAGS = ["diffae_features", "visualization", "pc_interpretation"]


def main(
    datasets_to_plot: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    dataset_info_columns: list[str] = DATASET_INFO_COLUMNS,
    segmentation_feature_group: str = "default",
    pc_group: str = "default",
    timepoint_annotations: list[TimepointAnnotation] | Literal["default"] | None = "default",
    aggregate_only: bool = True,
    skip_multi_feature_scatterplots: bool = True,
    plot_migration_coherence_correlations: bool = True,
) -> None:
    """
    Visualize correlation heatmaps and clustermaps for DiffAE features, PCs,
    and measured quantitites.

    Parameters
    ----------
    dataset_collection_to_plot
        The name of the dataset collection to analyze.
    dataset_collection_name_for_pca
        The name of the dataset collection to use for PCA fitting.
    model_manifest_name
        The name of the model manifest to use for DiffAE features.
    run_name
        The name of the run to use from the model manifest.
    seg_feature_manifest_name
        The name of the segmentation feature manifest to use for measured features.
    dataset_info_columns
        List of dataset metadata column names.
    segmentation_feature_group
        Preset name for selecting segmentation feature columns.
        If None, uses the default preset.
        Presets are defined in SEGMENTATION_FEATURE_COLUMNS.
    num_pcs
        Number of principal components to include. If None, uses NUM_PCS_TO_ANALYZE.
    timepoint_annotations
        List of timepoint annotations to exclude from the analysis. If "default",
        excludes NOT_STEADY_STATE and CELL_PILING timepoints. If None, includes all timepoints.
    aggregate_only
        If True, only uses the aggregated dataset in the analysis.
    skip_multi_feature_scatterplots
        If True, skips generating multi-feature scatterplots.

    NOTE
    ----
    This workflow may take several minutes to run, depending on the number of datasets
    and features being analyzed.
    Currently the datasets used to fit the PCA model (`diffae_training`) are also used
    to generate the correlation visualizations.
    Future versions may allow specifying separate dataset collections for PCA fitting
    and visualization.
    """

    import logging

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.configs.dataset_config_utils import get_subset_of_timepoint_annotations
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_pc_column_names,
    )
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_df_for_feature_correlation_viz,
        visualize_correlation_heatmaps,
    )
    from endo_pipeline.manifests import load_model_manifest

    logger = logging.getLogger(__name__)

    logger.info("Running correlation heatmap workflow...")

    dataset_name_list = use_default_collection(
        datasets_to_plot, DEFAULT_PCA_DATASET_COLLECTION_NAME
    )

    model_manifest = load_model_manifest(model_manifest_name)

    pc_columns = get_pc_column_names(pc_group)

    if timepoint_annotations == "default":
        annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
        timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)

    if isinstance(segmentation_feature_group, str):
        if segmentation_feature_group not in SEGMENTATION_FEATURE_COLUMNS:
            raise ValueError(
                f"Segmentation feature columns preset '{segmentation_feature_group}' "
                f"not found. Available presets: "
                f"{list(SEGMENTATION_FEATURE_COLUMNS.keys())}"
            )
        segmentation_feature_columns = SEGMENTATION_FEATURE_COLUMNS[segmentation_feature_group]
        segmentation_feature_columns.remove(Column.SegData.NODE_FLUOR_MEAN)
    else:
        raise TypeError(
            "segmentation_feature_group must be a string preset name or None.\n"
            "Refer to SEGMENTATION_FEATURE_COLUMNS for available presets."
        )

    # Long operation: takes several minutes
    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=dataset_info_columns,
        segmentation_feature_columns=segmentation_feature_columns,
        pc_columns=pc_columns,
        timepoint_annotations=timepoint_annotations,
    )

    label_column_tuples = [
        ("Measurement", [get_label_for_column(col) for col in segmentation_feature_columns]),
        ("PC", [get_label_for_column(col) for col in pc_columns]),
    ]
    out_dir = get_output_path(__file__)

    if plot_migration_coherence_correlations:
        # get the crop pattern for the migration coherence data (this is the grid crop pattern)
        crop_pattern = MIGRATION_COHERENCE_CROP_PATTERN
        model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
        feature_dataframe_manifest_name = get_feature_dataframe_manifest_name(
            model_manifest, run_name, crop_pattern=crop_pattern
        )
        feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

        # get fit PCA object to apply PCA transformation to diffae features before
        # plotting against optical flow features.
        pca = fit_pca(num_pcs=3)

        # load the grid-based DiffAE features upon which the migration coherence measurements
        # were based and add the optical flow features to this dataframe
        df_grid = pd.DataFrame()
        for dataset_name in dataset_name_list:
            if df_grid.empty:
                df_grid = get_dataframe_for_dynamics_workflows(
                    dataset_name,
                    feature_dataframe_manifest,
                    columns_to_keep=pc_columns,
                    pca=pca,
                    include_cell_piling=False,
                    include_not_steady_state=False,
                    crop_pattern=crop_pattern,
                )
            else:
                df_grid_new = get_dataframe_for_dynamics_workflows(
                    dataset_name,
                    feature_dataframe_manifest,
                    columns_to_keep=pc_columns,
                    pca=pca,
                    include_cell_piling=False,
                    include_not_steady_state=False,
                    crop_pattern=crop_pattern,
                )
                df_grid = pd.concat([df_grid, df_grid_new], ignore_index=True)
            df_grid = df_grid.reset_index()

        # add the optical flow features to the grid-based dataframe
        df_grid = add_optical_flow_features(
            df_grid,
            datasets=dataset_name_list,
        )

        optical_flow_features = [
            Column.OpticalFlow.UNIT_VECTOR_MEAN,
            Column.OpticalFlow.SPEED_MEAN,
            Column.OpticalFlow.ANGLE_MEAN,
        ]

        df_grid.rename(columns=get_label_for_column, inplace=True)

        label_column_tuples_grid = [
            ("Migration Coherence", [get_label_for_column(col) for col in optical_flow_features]),
            ("PC", [get_label_for_column(col) for col in pc_columns]),
        ]

    if aggregate_only:
        dataset_name_list = ["aggregate"]
    else:
        dataset_name_list = [*dataset_name_list, "aggregate"]

    for dataset_name in tqdm(dataset_name_list):
        # if the dataset name is "aggregate", use the full DataFrame
        # otherwise, filter the DataFrame for the specific dataset
        if dataset_name == "aggregate":
            df_dataset = df
        else:
            df_dataset = df[df[Column.DATASET] == dataset_name].copy()

        out_dir = get_output_path(__file__, dataset_name)

        visualize_correlation_heatmaps(
            dataset_name=dataset_name,
            df_dataset=df_dataset,
            label_column_tuples=label_column_tuples,
            out_dir=out_dir,
            skip_multi_feature_scatterplots=skip_multi_feature_scatterplots,
        )

        if plot_migration_coherence_correlations:
            if dataset_name == "aggregate":
                df_grid_dataset = df_grid
            else:
                df_grid_dataset = df_grid[df_grid[Column.DATASET] == dataset_name].copy()

            # out_dir = get_output_path(__file__, dataset_name)

            visualize_correlation_heatmaps(
                dataset_name=dataset_name,
                df_dataset=df_grid_dataset,
                label_column_tuples=label_column_tuples_grid,
                out_dir=out_dir,
                skip_multi_feature_scatterplots=skip_multi_feature_scatterplots,
            )

    logger.info(
        "Correlation heatmap workflow complete. Figures saved to [ %s ]",
        out_dir,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
