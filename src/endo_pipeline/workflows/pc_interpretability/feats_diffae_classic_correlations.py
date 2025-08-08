import re
from typing import Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from src.endo_pipeline.configs import (
    CytoDLModelConfig,
    load_dataset_collection_config,
    load_model_config,
)
from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_pca_loadings_as_df,
    get_valid_subset,
)
from src.endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_num_nuclei_in_crop_column,
)


def get_correlation_matrix_df(
    features_df: pd.DataFrame,
    column_names_for_x_axis: list[str],
    column_names_for_y_axis: list[str],
    x_axis_label: str,
    y_axis_label: str,
    df_format: Literal["long", "wide-corrcoeff", "wide-pval"] = "long",
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
    features_df : pd.DataFrame
        The DataFrame containing the features to correlate.
    column_names_for_x_axis : list[str]
        The names of the columns to use for the x-axis.
    column_names_for_y_axis : list[str]
        The names of the columns to use for the y-axis.
    name_of_x_axis : str
        The name of the x-axis.
    name_of_y_axis : str
        The name of the y-axis.
    df_format : Literal["long", "wide"], optional
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

    Returns
    -------
    pd.DataFrame
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
    elif df_format == "long":
        pass
    else:
        raise ValueError(
            f"Unsupported df_format: {df_format}. Supported formats are 'long', 'wide-corrcoeff', 'wide-pval'."
        )
    return correlation_df


def run_correlation_heatmap_workflow() -> None:
    """
    Run the workflow to generate correlation heatmaps between DiffAE features, PCA components,
    and measured properties for each dataset in the PCA reference collection.
    This function loads the dataset collection configuration, preprocesses the manifests,
    computes the correlations, and saves the heatmaps to the projects results directory
    under a folder with the same name as this script.

    Notes
    -----
    This workflow takes approximately 45 minutes to run with an 8 core CPU if
    nuclei centroids are not pre-computed.
    If they are pre-computed, it takes about 7 minutes.
    """

    # dataset_name_list = load_dataset_collection_config("pca_reference").datasets

    model_name = "diffae_04_10"
    dataset_name_list = []
    for manifest in cast(CytoDLModelConfig, load_model_config(model_name)).manifest_fmsids:
        if (
            manifest.dataset_name
            in load_dataset_collection_config(
                "live_20X_objective_3i_microscope_timelapses"
            ).datasets
        ):
            dataset_name_list.append(manifest.dataset_name)

    # instantiate an empty list to hold the DataFrames for later analysis
    df_list = []

    for dataset_name in dataset_name_list:

        out_subdir = get_output_path(__file__, dataset_name, include_timestamp=False)

        # load and preprocess the different diffae manifests and PCA pipeline
        # NOTE: this takes a little over a minute to load
        merged_feats_df, _, _ = get_preprocessed_manifests_and_km_bounds(
            dataset_name, datasets_for_bounds=dataset_name_list
        )

        # add the number of nuclei columns
        merged_feats_df = add_num_nuclei_in_crop_column(merged_feats_df, use_precomputed=True)

        # get the names of the columns that you are interested in comparing
        pc_col_names = []
        diffae_feature_col_names = []
        for col_nm in merged_feats_df.columns:
            # get any pc column names
            if re.match("pc[0-9]", col_nm):
                pc_col_names.append(col_nm)
            # get any diffae feature column names
            elif re.match("feat_[0-9]+", col_nm):
                diffae_feature_col_names.append(col_nm)
            else:
                continue

        # get select measurement column names
        measured_col_names = [
            "alignment_deg_rel_to_flow",
            "nematic_order",  # note: this is calculated from alignment_deg_rel_to_flow
            "eccentricity",
            "aspect_ratio",
            "cell_fluorescence_mean (a.u.)",
            "num_nuclei_in_crop",
            "perimeter",
            "area",
            "cell_solidity",
            "number_of_neighbors",
            "nuc_pos_rel_cell_angle_deg",
        ]
        dataset_info_cols = [
            "dataset_name",
            "position",
            "image_index",
            "frame_number",
            "track_id",
            "crop_index",
            "label",
        ]
        # double-check that the chosen measurement column names
        # are actually in the DataFrame
        assert all(np.isin(measured_col_names + dataset_info_cols, merged_feats_df.columns)), (
            f"Not all columns names are in merged_feats_df. "
            f"Missing: {set(measured_col_names + dataset_info_cols) - set(merged_feats_df.columns)}"
        )

        # keep only the columns that will be used to conserve memory
        cols_to_keep = (
            dataset_info_cols + measured_col_names + diffae_feature_col_names + pc_col_names
        )
        merged_feats_df = merged_feats_df[cols_to_keep]

        # add the dataframe to a list with all of them so we can
        # do a correlation analysis across all listed datasets
        df_list.append(merged_feats_df)

        # create heatmaps of the correlations between measured properties
        # and the diffae features or PCs:
        comparisons_to_make = [
            (  # heatmap args for measurements vs. DiffAE features
                "Measurement",
                measured_col_names,
                "DiffAE Feature",
                diffae_feature_col_names,
                f"{dataset_name}_correlation_feats_vs_measured",
            ),
            (  # heatmap args for measurements vs. PCs
                "Measurement",
                measured_col_names,
                "PC",
                pc_col_names,
                f"{dataset_name}_correlation_pcs_vs_measured",
            ),
            (  # heatmaps for the measurements vs. themselves to see check for co-occurrence
                "Measurement 1",
                measured_col_names,
                "Measurement 2",
                measured_col_names,
                f"{dataset_name}_correlation_measured_vs_measured",
            ),
        ]
        for x_axis_label, x_cols, y_axis_label, y_cols, filename in comparisons_to_make:
            # create the correlation DataFrame
            correlation_df = get_correlation_matrix_df(
                features_df=merged_feats_df,
                column_names_for_x_axis=x_cols,
                column_names_for_y_axis=y_cols,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                df_format="wide-corrcoeff",
            )
            # make the heatmap
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1, ax=ax)
            ax.tick_params(axis="y", rotation=0)  # rotate y-axis labels for better readability
            save_plot_to_path(fig, output_path=out_subdir, figure_name=filename)

        # repeat the above correlations but filter data table
        # to only include the steady state timepoints that are
        # used when projecting the DiffAE features onto PCA axes
        # in the segmentation-free dynamics workflow
        merged_feats_df = get_valid_subset(
            merged_feats_df,
            dataset_name=dataset_name,
            verbose=False,
        )

        for x_axis_label, x_cols, y_axis_label, y_cols, filename in comparisons_to_make:
            # create the correlation DataFrame
            correlation_df = get_correlation_matrix_df(
                features_df=merged_feats_df,
                column_names_for_x_axis=x_cols,
                column_names_for_y_axis=y_cols,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                df_format="wide-corrcoeff",
            )
            # make the heatmap
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1, ax=ax)
            ax.tick_params(axis="y", rotation=0)  # rotate y-axis labels for better readability
            save_plot_to_path(
                figure=fig,
                output_path=out_subdir,
                figure_name=f"{filename}_steady_state",
            )

    # get the PCA loadings DataFrame
    pca = fit_pca()
    pca_loadings_df = get_pca_loadings_as_df(pca, scaled=True, magnitude=False, df_format="wide")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        pca_loadings_df,
        annot=True,
        cmap="RdBu",
        center=0,
        ax=ax,
        cbar_kws={"label": "Scaled Loading Value"},
    )
    ax.set_xlabel("PC")
    ax.set_ylabel("Latent Feature")
    save_plot_to_path(
        figure=fig,
        output_path=get_output_path(__file__, include_timestamp=False),
        figure_name="pca_loadings_scaled_heatmap",
    )

    # produce correlation heatmaps across all datasets
    all_feats_df = pd.concat(df_list, ignore_index=True)

    comparisons_to_make = [
        (  # heatmap args for measurements vs. DiffAE features
            "Measurement",
            measured_col_names,
            "DiffAE Feature",
            diffae_feature_col_names,
            f"multi_dataset_correlation_feats_vs_measured",
        ),
        (  # heatmap args for measurements vs. PCs
            "Measurement",
            measured_col_names,
            "PC",
            pc_col_names,
            f"multi_dataset_correlation_pcs_vs_measured",
        ),
        (  # heatmaps for the measurements vs. themselves to see check for co-occurrence
            "Measurement 1",
            measured_col_names,
            "Measurement 2",
            measured_col_names,
            f"multi_dataset_correlation_measured_vs_measured",
        ),
    ]

    out_subdir = get_output_path(__file__, "aggregated_dataset_analysis", include_timestamp=False)

    for x_axis_label, x_cols, y_axis_label, y_cols, filename in comparisons_to_make:
        # create the correlation DataFrame
        correlation_df = get_correlation_matrix_df(
            features_df=all_feats_df,
            column_names_for_x_axis=x_cols,
            column_names_for_y_axis=y_cols,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            df_format="wide-corrcoeff",
        )
        # make the heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1, ax=ax)
        ax.tick_params(axis="y", rotation=0)  # rotate y-axis labels for better readability
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=filename,
        )


if __name__ == "__main__":
    from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    # NOTE: ipython_cli_flexecute calls `workflow_cli` internally
    #  if run from the command line
    ipython_cli_flexecute(run_correlation_heatmap_workflow)
