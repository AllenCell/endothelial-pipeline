from typing import Literal

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import TimepointAnnotation
from endo_pipeline.settings.diffae_feature_dataframes import MAX_PCS_TO_COMPUTE
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
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    dataset_info_columns: list[str] = DATASET_INFO_COLUMNS,
    segmentation_feature_group: str = "default",
    num_pcs: int | None = None,
    timepoint_annotations: list[TimepointAnnotation] | Literal["default"] | None = "default",
    aggregate_only: bool = True,
    skip_multi_feature_scatterplots: bool = True,
    compare_with_diffae_features: bool = False,
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
        The name of the run to use from the model manifest. If None, uses the most
        recent run.
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

    import itertools
    import logging

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.configs.dataset_config_utils import get_subset_of_timepoint_annotations
    from endo_pipeline.configs.model_config_utils import get_latent_dim_from_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.io.input import get_config_dict_from_mlflow
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        get_latent_feature_column_names,
        get_pc_column_names,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_df_for_feature_correlation_viz,
        plot_and_save_clustermap,
        plot_multi_feature_correlations,
    )
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.manifests.model_manifest_utils import (
        get_model_location_for_run,
        get_most_recent_run_name,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

    logger = logging.getLogger(__name__)

    logger.info("Running correlation heatmap workflow...")

    dataset_name_list = use_default_collection(
        datasets_to_plot, DEFAULT_PCA_DATASET_COLLECTION_NAME
    )

    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model_location = get_model_location_for_run(model_manifest, run_name_)
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)  # type: ignore
    num_features = get_latent_dim_from_config(model_config)
    num_pcs = num_pcs if num_pcs is not None else min(MAX_PCS_TO_COMPUTE, num_features)

    pc_columns = get_pc_column_names(num_pcs)
    # use the first 3 PCs and PC18 for correlation (PCs are 1-index)
    pc_columns = pc_columns[:2] + pc_columns[17:18]
    diffae_feature_columns = get_latent_feature_column_names(num_features)
    polar_pc_columns = [ColumnName.POLAR_RADIUS, ColumnName.POLAR_ANGLE]

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
        diffae_feature_columns=diffae_feature_columns,
        polar_pc_columns=polar_pc_columns,
        timepoint_annotations=timepoint_annotations,
    )

    pc_and_polar_group = [*pc_columns[:3], *polar_pc_columns]

    label_column_tuples = [
        ("Measurement", [get_label_for_column(col) for col in segmentation_feature_columns]),
        ("PC", [get_label_for_column(col) for col in pc_columns]),
        ("PC with polar transform", [get_label_for_column(col) for col in pc_and_polar_group]),
    ]
    if compare_with_diffae_features:
        label_column_tuples.append(
            ("DiffAE Feature", [get_label_for_column(col) for col in diffae_feature_columns])
        )

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
            df_dataset = df.query("dataset_name==@dataset_name").copy()

        # Pre-compute full correlation matrix once per dataset
        all_feature_columns = []
        for _, cols in label_column_tuples:
            all_feature_columns.extend(cols)
        # Remove duplicates while preserving order
        unique_feature_columns = []
        seen = set()
        for col in all_feature_columns:
            if col not in seen:
                unique_feature_columns.append(col)
                seen.add(col)

        logger.info("Computing full correlation matrix for dataset %s", dataset_name)
        values_for_corr = df_dataset[unique_feature_columns].dropna().to_numpy()
        # Use numpy to compute correlation matrix faster
        corr_matrix = np.corrcoef(values_for_corr, rowvar=False)
        corr_df = pd.DataFrame(
            corr_matrix,
            index=unique_feature_columns,
            columns=unique_feature_columns,
        )

        # Pre-compute dataset color mapping once per dataset
        dataset_color_mapping = {
            ds_nm: get_dataset_color(ds_nm) for ds_nm in df_dataset["dataset_name"].unique()
        }
        colors = df_dataset["dataset_name"].map(dataset_color_mapping).to_list()

        for (x_axis_label, x_cols), (
            y_axis_label,
            y_cols,
        ) in itertools.combinations_with_replacement(label_column_tuples, 2):
            logger.debug(
                "Processing correlation between %s and %s for dataset %s",
                x_axis_label,
                y_axis_label,
                dataset_name,
            )

            # Ensure the figure is in landscape orientation
            if len(y_cols) > len(x_cols):
                x_cols, y_cols = y_cols, x_cols
                x_axis_label, y_axis_label = y_axis_label, x_axis_label

            x_filename = x_axis_label.replace(" ", "_").lower()
            y_filename = y_axis_label.replace(" ", "_").lower()
            base_filename = f"correlation_{x_filename}_vs_{y_filename}"

            out_subdir = get_output_path(
                __file__,
                dataset_name,
                segmentation_feature_group,
                f"{num_features}_features_x_{num_pcs}_pcs",
                f"{x_filename}_vs_{y_filename}",
                include_timestamp=True,
            )

            # Extract correlation submatrix from pre-computed correlation matrix
            correlation_df = corr_df.loc[y_cols, x_cols].copy()
            correlation_df.columns.name = x_axis_label  # columns go on the x axis
            correlation_df.index.name = y_axis_label  # index goes on the y axis
            correlation_df.to_csv(out_subdir / f"{base_filename}_correlation_matrix.csv")

            # make correlation clustermap
            plot_and_save_clustermap(
                df=correlation_df,
                output_folder=out_subdir,
                filename=base_filename,
                metric="cosine",
                data_type="correlation",
            )

            if skip_multi_feature_scatterplots:
                continue

            if len(x_cols) > 16 or len(y_cols) > 16:
                logger.info(
                    "Skipping scatter plot for %s vs %s for dataset %s "
                    "due to large number of features (%s x %s).",
                    x_axis_label,
                    y_axis_label,
                    dataset_name,
                    len(x_cols),
                    len(y_cols),
                )
                continue

            column_list = []
            for col in x_cols + y_cols:
                if col not in column_list:
                    column_list.append(col)  # this preserves column order

            plot_multi_feature_correlations(
                df=df_dataset[column_list],
                output_folder=out_subdir,
                filename=f"{base_filename}_scatter",
                color=colors,
            )

    logger.info(
        "Correlation heatmap workflow complete. Figures saved to [ %s ]",
        out_subdir.parents[2],
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
