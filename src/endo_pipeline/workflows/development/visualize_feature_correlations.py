TAGS = ["diffae_features", "visualization", "pc_interpretation"]

CLASSICAL_FEATURE_COLUMNS = [
    "alignment_deg_rel_to_flow",
    "aspect_ratio",
    "cell_fluorescence_mean (a.u.)",
    "num_nuclei_in_crop",
    "area (um**2)",
    "nuc_pos_rel_cell_angle_deg",
    "number_of_neighbors",
]
DATASET_INFO_COLUMNS = [
    "dataset_name",
    "position",
    "image_index",
    "frame_number",
    "track_id",
    "crop_index",
    "label",
]
DIFFAE_FEATURE_COLUMNS = [f"feat_{i}" for i in range(8)]
PC_COLUMNS = [f"pc{i}" for i in range(1, 4)]


def main(
    dataset_collection_name: str = "pca_reference",
    dataset_info_columns: list[str] = DATASET_INFO_COLUMNS,
    classical_feature_columns: list[str] = CLASSICAL_FEATURE_COLUMNS,
    pc_columns: list[str] = PC_COLUMNS,
    diffae_feature_columns: list[str] = DIFFAE_FEATURE_COLUMNS,
    aggregate: bool = True,
) -> None:
    """
    Visualize correlation heatmaps and clustermaps for DiffAE features, PCs,
    and measured quantitites.

    Parameters
    ----------
    dataset_collection_name
        The name of the dataset collection to use.
    dataset_info_columns
        List of dataset metadata column names.
    classical_feature_columns
        List of classical feature column names.
    pc_columns
        List of PCA component column names.
    diffae_feature_columns
        List of DiffAE feature column names.
    aggregate
        If True, include an aggregated dataset in the analysis.
    """

    import itertools
    import logging

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_correlation_matrix_df,
        get_df_for_feature_correlation_viz,
        plot_and_save_clustermap,
        plot_and_save_heatmap,
        plot_multi_feature_correlations,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("Running correlation heatmap workflow...")

    dataset_name_list = get_datasets_in_collection(dataset_collection_name)

    df_all_timepoints, df_ss = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=dataset_info_columns,
        classical_feature_columns=classical_feature_columns,
        pc_columns=pc_columns,
        diffae_feature_columns=diffae_feature_columns,
    )

    label_column_tuples = [
        ("Measurement", [get_label_for_column(col) for col in classical_feature_columns]),
        ("PC", [get_label_for_column(col) for col in pc_columns]),
        ("DiffAE Feature", [get_label_for_column(col) for col in diffae_feature_columns]),
    ]

    if aggregate:
        dataset_name_list = [*dataset_name_list, "aggregate"]

    for dataset_name in dataset_name_list:
        # if the dataset name is "aggregate", use the full DataFrame
        # otherwise, filter the DataFrame for the specific dataset
        if dataset_name == "aggregate":
            df_dataset = df_all_timepoints
            df_dataset_ss = df_ss
        else:
            df_dataset = df_all_timepoints.query("dataset_name==@dataset_name").copy()
            df_dataset_ss = df_ss.query("dataset_name==@dataset_name").copy()

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
            x_filename = x_axis_label.replace(" ", "_").lower()
            y_filename = y_axis_label.replace(" ", "_").lower()
            base_filename = f"correlation_{x_filename}_vs_{y_filename}"

            if x_axis_label == y_axis_label:
                x_axis_label = f"{x_axis_label} 1"
                y_axis_label = f"{y_axis_label} 2"

            for df, timepoint_label in zip(
                (df_dataset_ss, df_dataset),
                ("steady_state", "all_timepoints"),
                strict=True,
            ):
                out_subdir = get_output_path(
                    __file__,
                    dataset_name,
                    timepoint_label,
                    f"{x_filename}_vs_{y_filename}",
                    include_timestamp=False,
                )

                # create the correlation DataFrame
                correlation_df = get_correlation_matrix_df(
                    features_df=df,
                    column_names_for_x_axis=x_cols,
                    column_names_for_y_axis=y_cols,
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    df_format="wide-corrcoeff",
                )

                # make correlation heatmap
                plot_and_save_heatmap(
                    df=correlation_df,
                    output_folder=out_subdir,
                    filename=f"{base_filename}_{timepoint_label}_heatmap",
                )

                # make correlation clustermap
                plot_and_save_clustermap(
                    df=correlation_df,
                    output_folder=out_subdir,
                    filename=f"{base_filename}_{timepoint_label}_clustermap",
                )

                # make scatter plot
                colors = df["dataset_name"].apply(lambda x: get_dataset_color(x)).tolist()
                column_list = []
                for col in x_cols + y_cols:
                    if col not in column_list:
                        column_list.append(col)  # this preserves column order

                plot_multi_feature_correlations(
                    df=df[column_list],
                    output_folder=out_subdir,
                    filename=f"{base_filename}_{timepoint_label}_scatter",
                    color=colors,
                    title=f"{dataset_name} {timepoint_label} {x_axis_label} vs {y_axis_label}",
                )

    logger.info(
        "Correlation heatmap workflow complete. Figures saved to [ %s ]",
        get_output_path(__file__, include_timestamp=False),
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
