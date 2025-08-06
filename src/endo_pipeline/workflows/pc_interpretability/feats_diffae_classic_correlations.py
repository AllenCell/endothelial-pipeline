import re
from typing import Literal

import numpy as np
import pandas as pd

# for data exploration; remove later
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from src.endo_pipeline.configs import load_dataset_collection_config
from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest.diffae_manifest_utils import get_valid_subset
from src.endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_num_nuclei_in_crop_column,
)


def adjust_crop_bounds_to_0th_bin_level(
    merged_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adjust the crop bounds to the 0th level of resolution for the imaging data.
    Context: the start_y, end_y, start_x, end_x columns that are used to define
    a crop in the merged_feats_df DataFrame are recorded for the image at bin
    level 1 since that is what the DiffAE model was learning at the time.

    Parameters
    ----------
    merged_feats_df : pd.DataFrame
        The DataFrame containing crop bound columns in the form of
        (starty, end_y, start_x, end_x) and a column "resolution_level"
    Returns
    -------
    pd.DataFrame
        The DataFrame with the crop bounds adjusted to the 0th level of resolution.
    """
    # adjust the crop bounds to the 0th level of resolution
    merged_feats_df["start_y"] = (
        merged_feats_df["start_y"] * (merged_feats_df["resolution_level"] + 1)
    ).astype(int)
    merged_feats_df["end_y"] = (
        merged_feats_df["end_y"] * (merged_feats_df["resolution_level"] + 1)
    ).astype(int)
    merged_feats_df["start_x"] = (
        merged_feats_df["start_x"] * (merged_feats_df["resolution_level"] + 1)
    ).astype(int)
    merged_feats_df["end_x"] = (
        merged_feats_df["end_x"] * (merged_feats_df["resolution_level"] + 1)
    ).astype(int)
    return merged_feats_df


def get_correlation_matrix_df(
    features_df: pd.DataFrame,
    column_names_for_x_axis: list[str],
    column_names_for_y_axis: list[str],
    name_of_x_axis: str,
    name_of_y_axis: str,
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
                    name_of_y_axis: col_for_y,
                    name_of_x_axis: col_for_x,
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
            index=name_of_y_axis,
            columns=name_of_x_axis,
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


if __name__ == "__main__":
    dataset_name_list = load_dataset_collection_config("pca_reference").datasets
    # dataset_name = dataset_name_list[0]  # for testing purposes
    for dataset_name in dataset_name_list:

        # load and preprocess the different diffae manifests and PCA pipeline
        # NOTE: this takes a little over a minute to load
        merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
            dataset_name, datasets_for_bounds=dataset_name_list
        )

        # adjust the crop coordinates back to the native resolution since
        # the label-free nuclei predictions are saved at that resolution
        merged_feats_df = adjust_crop_bounds_to_0th_bin_level(merged_feats_df)

        # add the number of nuclei columns
        merged_feats_df = add_num_nuclei_in_crop_column(merged_feats_df)

        out_subdir = get_output_path(__file__, dataset_name, include_timestamp=False)

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
            "area",
            "perimeter",
            "eccentricity",
            "aspect_ratio",
            "cell_fluorescence_mean (a.u.)",
            "num_nuclei_in_crop",
            "cell_solidity",
            "number_of_neighbors",
            "nuc_pos_rel_cell_angle_deg",
        ]
        # double-check that the chosen measurement column names
        # are actually in the DataFrame
        assert all(np.isin(measured_col_names, merged_feats_df.columns)), (
            f"Not all measured_col_names are in merged_feats_df. "
            f"Missing: {set(measured_col_names) - set(merged_feats_df.columns)}"
        )

        # 1. create heatmaps of the correlations between measured properties
        # and the diffae features or PCs:
        # heatmap for measurements vs. DiffAE features
        correlation_df_feats = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=diffae_feature_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="feature",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_feats, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_feats_vs_measured",
        )

        # heatmap for measurements vs. PCs
        correlation_df_pcs = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_pcs, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_pcs_vs_measured",
        )

        # heatmaps for the measurements vs. themselves
        # to see if any measures tend to co-occur
        correlation_df = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=measured_col_names,
            name_of_x_axis="measure1",
            name_of_y_axis="measure2",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        ax.set_xlabel(""), ax.set_ylabel("")
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_measured_vs_measured",
        )

        # repeat the above correlations but filter data table
        # to only include the steady state timepoints that are
        # used when projecting the DiffAE features onto PCA axes
        # in the segmentation-free dynamics workflow
        merged_feats_df = get_valid_subset(
            merged_feats_df,
            dataset_name=dataset_name,
            verbose=False,
        )

        # heatmap for measurements vs. DiffAE features
        correlation_df_feats = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=diffae_feature_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="feature",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_feats, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_feats_vs_measured_steady_state",
        )

        # heatmap for measurements vs. PCs
        correlation_df_pcs = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_pcs, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_pcs_vs_measured_steady_state",
        )

        # heatmaps for the measurements vs. themselves
        # to see if any measures tend to co-occur
        correlation_df = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=measured_col_names,
            name_of_x_axis="measure1",
            name_of_y_axis="measure2",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        ax.set_xlabel(""), ax.set_ylabel("")
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_measured_vs_measured_steady_state",
        )

        # 2. plot the PC loadings
        # from src.endo_pipeline.workflows.development.visualize_pca_attributes import (
        #     plot_component_loadings,
        # )

        # 3. correlations between diffae features and PCs
        # (NOTE THAT GETTING THE PC LOADINGS IS THE CORRECT APPROACH)
        correlation_df = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=diffae_feature_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="feature",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_pcs_vs_diffae_feats",
        )
