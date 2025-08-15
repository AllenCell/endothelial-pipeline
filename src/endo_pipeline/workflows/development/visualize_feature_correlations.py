TAGS = ["diffae_features", "visualization", "pc_interpretation"]


def main():
    """
    Visualize correlation heatmaps and clustermaps for DiffAE features, PCs,
    and measured quantitites.
    """
    import itertools
    import logging
    from pathlib import Path
    from typing import Literal

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import linkage
    from scipy.stats import pearsonr

    from src.endo_pipeline.configs import load_dataset_collection_config
    from src.endo_pipeline.io import get_output_path, save_plot_to_path
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

    def get_correlation_matrix_df(
        features_df: pd.DataFrame,
        column_names_for_x_axis: list[str],
        column_names_for_y_axis: list[str],
        x_axis_label: str,
        y_axis_label: str,
        df_format: Literal["long", "wide-corrcoeff", "wide-pval"] = "long",
        sort_by_correlation: bool = False,
    ) -> pd.DataFrame:
        """
        Get the Pearson correlations between each column in `column_names_for_x_axis`
        compared with each column in `column_names_for_y_axis`.
        This is used to compare the diffae features and the measured features,
        and then used again to compare the PCs and the measured features.
        If `df_format` is one of the "wide" options then the outputted dataframe
        of this function can be passed directly to `seaborn.heatmap` or
        `seaborn.clustermap` for visualization.

        Parameters
        ----------
        features_df
            The DataFrame containing the features to correlate.
        column_names_for_x_axis
            The names of the columns to use for the x-axis.
        column_names_for_y_axis
            The names of the columns to use for the y-axis.
        name_of_x_axis
            The name of the x-axis.
        name_of_y_axis
            The name of the y-axis.
        df_format
            The format of the output DataFrame. If "long", the output DataFrame will have columns:
            - name_of_y_axis
            - name_of_x_axis
            - pearsonr
            - pval
            If "wide-corrcoeff", the output DataFrame will have a column for each column in
            column_names_for_x_axis and the index will be the column names in
            column_names_for_y_axis, with the values in the DataFrame corresponding to the
            correlation coefficients from the "long" version of the table.
            "wide-pval" is similar to "wide-corrcoeff" but the values correspond to the p-values.
            Defaults to "long".
        sort_by_correlation
            If True, the output DataFrame will be sorted by the correlation coefficients

        Returns
        -------
        :
            A DataFrame containing the Pearson correlation coefficients and p-values between
            the specified columns in `features_df`. The format of the DataFrame depends on
            the `df_format` parameter.

        Notes
        -----
        Rows with non-finite values in the features_df DataFrame will be dropped
        For the specific comparison where the non-finite value would show up
        (but not for the other comparisons).
        """
        records = []
        for col_for_y in column_names_for_y_axis:
            for col_for_x in column_names_for_x_axis:
                valid_records = np.isfinite(features_df[[col_for_y, col_for_x]]).all(axis=1)
                corr, pval = pearsonr(
                    features_df[col_for_y][valid_records],
                    features_df[col_for_x][valid_records],
                )
                records.append(
                    {
                        y_axis_label: col_for_y,
                        x_axis_label: col_for_x,
                        "pearsonr": corr,
                        "pval": pval,
                    }
                )
        correlation_df = pd.DataFrame(records)

        if df_format in ("wide-corrcoeff", "wide-pval"):
            if df_format == "wide-corrcoeff":
                value_col = "pearsonr"
            elif df_format == "wide-pval":
                value_col = "pval"
            correlation_df = correlation_df.pivot(
                index=y_axis_label,
                columns=x_axis_label,
                values=value_col,
            )
            correlation_df = correlation_df[column_names_for_x_axis]  # sort the columns
            correlation_df = correlation_df.reindex(index=column_names_for_y_axis)  # sort the index
            if sort_by_correlation:
                correlation_df = correlation_df.T.loc[
                    correlation_df.T[column_names_for_y_axis]
                    .abs()
                    .sort_values(by=column_names_for_y_axis, axis=0, ascending=False)
                    .index
                ].T

        elif df_format == "long":
            pass
        else:
            raise ValueError(
                f"Unsupported df_format: {df_format}. "
                f"Supported: 'long', 'wide-corrcoeff', 'wide-pval'."
            )
        return correlation_df

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
        df_list_all_tps = []
        df_list_ss = []
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
            assert all(np.isin(columns_to_check, merged_feats_df.columns)), (
                f"Not all columns names are in merged_feats_df. Missing:\n"
                f"{set(columns_to_check) - set(merged_feats_df.columns)}"
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

    def plot_and_save_heatmap(
        df: pd.DataFrame,
        output_folder: Path,
        filename: str = "correlation_heatmap",
    ) -> None:
        """
        Plot and save a heatmap of the correlation matrix from the given DataFrame.

        Parameters
        ----------
        df
            The DataFrame containing the correlation matrix.
        output_folder
            The folder where the heatmap will be saved.
        filename
            The name of the file to save the heatmap as.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1, ax=ax)
        ax.tick_params(axis="y", rotation=0)
        save_plot_to_path(
            figure=fig,
            output_path=output_folder,
            figure_name=filename,
        )

    def plot_and_save_clustermap(
        df: pd.DataFrame,
        output_folder: Path,
        filename: str = "correlation_clustermap",
    ):
        """
        Plot and save a clustermap of the correlation matrix from the given DataFrame.
        Clustering is performed on absolute values of the correlation coefficients.

        Parameters
        ----------
        df
            The DataFrame containing the correlation matrix.
        output_folder
            The folder where the clustermap will be saved.
        filename
            The name of the file to save the clustermap as.
        """
        abs_data = df.abs().T
        col_linkage = linkage(abs_data)
        cluster_grid = sns.clustermap(
            df,
            annot=True,
            cmap="RdBu",
            center=0,
            vmin=-1,
            vmax=1,
            figsize=(10, 10),
            row_cluster=False,
            col_cluster=True,
            col_linkage=col_linkage,
        )
        save_plot_to_path(
            figure=cluster_grid.figure,
            output_path=output_folder,
            figure_name=filename,
        )

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
        dataset_name_list = load_dataset_collection_config(dataset_collection_name).datasets

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
                    f"Processing correlation between {x_axis_label} and {y_axis_label} "
                    f"for dataset {dataset_name}"
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
    # from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    # ipython_cli_flexecute(main)
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
