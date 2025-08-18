TAGS = ["diffae_features", "visualization", "pc_interpretation"]


def main() -> None:
    """
    Visualize correlation heatmaps and clustermaps for DiffAE features, PCs,
    and measured quantitites.
    """
    import itertools
    import logging

    import numpy as np
    import pandas as pd

    from src.endo_pipeline.configs import get_datasets_in_collection
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.analyze.diffae_manifest import get_valid_subset
    from src.endo_pipeline.library.analyze.integration.track_integration import (
        get_preprocessed_manifests_and_km_bounds,
    )
    from src.endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_num_nuclei_in_crop_column,
    )
    from src.endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from src.endo_pipeline.library.visualize.multi_feature_visualization import (
        get_correlation_matrix_df,
        plot_and_save_clustermap,
        plot_and_save_heatmap,
        plot_multi_feature_correlations,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

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

    logger.info("Running correlation heatmap workflow...")

    def get_merged_feature_df(
        dataset_name_list: list[str],
        dataset_info_columns: list[str] = DATASET_INFO_COLUMNS,
        classical_feature_columns: list[str] = CLASSICAL_FEATURE_COLUMNS,
        pc_columns: list[str] = PC_COLUMNS,
        diffae_feature_columns: list[str] = DIFFAE_FEATURE_COLUMNS,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the manifests for the given dataset names,
        and return a DataFrame containing the merged features
        from all datasets at all timepoints, and a DataFrame
        containing the steady state timepoints only.

        Parameters
        ----------
        dataset_name_list
            List of dataset names to process.
        dataset_info_columns
            List of columns containing dataset information.
        classical_features
            List of classical feature column names.
        pc_columns
            List of PCA component column names.
        diffae_feature_columns
            List of DiffAE feature column names.

        Returns
        -------
        :
            A tuple containing two DataFrames:
            - The first DataFrame contains all timepoints for the given datasets.
            - The second DataFrame contains only the steady state timepoints.
        """
        df_list_all_tps: list = []
        df_list_ss: list = []
        for dataset_name in dataset_name_list:
            # load and preprocess the different diffae manifests and PCA pipeline
            # NOTE: this takes a little over a minute to load
            merged_feats_df, _, _ = get_preprocessed_manifests_and_km_bounds(
                dataset_name, datasets_for_bounds=dataset_name_list
            )

            # add the number of nuclei columns
            merged_feats_df = add_num_nuclei_in_crop_column(merged_feats_df, use_precomputed=True)

            # check that the chosen measurement column names
            # are actually in the DataFrame
            columns_to_check = classical_feature_columns + dataset_info_columns
            if not all(np.isin(columns_to_check, merged_feats_df.columns)):
                missing_columns = set(columns_to_check) - set(merged_feats_df.columns)
                raise ValueError(
                    f"Not all columns names are in merged_feats_df. Missing:\n{missing_columns}"
                )

            # filter data table to only include the steady state timepoints that are
            # used when projecting the DiffAE features onto PCA axes
            # in the segmentation-free dynamics workflow
            merged_feats_df_ss = get_valid_subset(
                df=merged_feats_df,
                dataset_name=dataset_name,
                verbose=False,
            )

            # keep only the columns that will be used
            cols_to_keep = (
                dataset_info_columns
                + classical_feature_columns
                + diffae_feature_columns
                + pc_columns
            )

            for df, df_list in zip(
                (merged_feats_df, merged_feats_df_ss),
                (df_list_all_tps, df_list_ss),
            ):
                df = df[cols_to_keep].copy()
                df.rename(columns=get_label_for_column, inplace=True)
                df_list.append(df)
        # merge the DataFrames from all datasets
        df_all_timepoints = pd.concat(df_list_all_tps, ignore_index=True)
        df_ss = pd.concat(df_list_ss, ignore_index=True)

        return df_all_timepoints, df_ss

    def run_correlation_heatmap_workflow(
        dataset_collection_name: str = "pca_reference",
        classical_feature_columns: list[str] = CLASSICAL_FEATURE_COLUMNS,
        pc_columns: list[str] = PC_COLUMNS,
        diffae_feature_columns: list[str] = DIFFAE_FEATURE_COLUMNS,
        aggregate: bool = False,
    ) -> None:
        """
        Run the workflow to generate correlation heatmaps between DiffAE features, PCA components,
        and measured properties for each dataset in the PCA reference collection.

        Parameters
        ----------
        dataset_collection_name
            The name of the dataset collection to use.
            Uses the pca reference collection by default.
        classical_feature_columns
            List of classical feature column names.
            Defaults to CLASSICAL_FEATURE_COLUMNS.
        pc_columns
            List of PCA component column names.
            Defaults to PC_COLUMNS.
        diffae_feature_columns
            List of DiffAE feature column names.
            Defaults to DIFFAE_FEATURE_COLUMNS.
        aggregate
            If True, include an aggregated dataset in the analysis.
            Defaults to False.
        """
        dataset_name_list = get_datasets_in_collection(dataset_collection_name)

        df_all_timepoints, df_ss = get_merged_feature_df(
            dataset_name_list,
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
            dataset_name_list = dataset_name_list + ["aggregate"]

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
                # loop over all combinations
                x_filename = x_axis_label.replace(" ", "_").lower()
                y_filename = y_axis_label.replace(" ", "_").lower()
                base_filename = f"correlation_{x_filename}_vs_{y_filename}"

                if x_axis_label == y_axis_label:
                    x_axis_label = f"{x_axis_label} 1"
                    y_axis_label = f"{y_axis_label} 2"

                for df, timepoint_label in zip(
                    (df_dataset_ss, df_dataset),
                    ("steady_state", "all_timepoints"),
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

    run_correlation_heatmap_workflow(
        dataset_collection_name="pca_reference",
        classical_feature_columns=CLASSICAL_FEATURE_COLUMNS,
        pc_columns=PC_COLUMNS,
        diffae_feature_columns=DIFFAE_FEATURE_COLUMNS,
        aggregate=True,
    )
    logger.info(
        "Correlation heatmap workflow complete. Figures saved to [ %s ]",
        get_output_path(__file__, include_timestamp=False),
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
