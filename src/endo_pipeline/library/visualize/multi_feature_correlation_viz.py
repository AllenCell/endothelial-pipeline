"""
Methods to load data and visualize multi-feature correlations.

Creates an n_features X n_features grid of plots with:

1) Scatter plots of features on the lower triangle
2) Feature histograms on the diagonal
3) Correlation values on the upper triangle
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy import stats as spstats
from scipy.cluster.hierarchy import linkage

from endo_pipeline.configs import TimepointAnnotation, load_dataset_config
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import ModelManifest


def add_feature_scatter_plot(
    ax: Axes,
    feat1_id: int,
    feat2_id: int,
    feat1: np.ndarray,
    feat2: np.ndarray,
    num_features: int,
    color: str | list | np.ndarray = "black",
    alpha: float = 0.1,
) -> tuple[float, float]:
    """
    Add scatter plots to the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat1_id
        Index of feature to be plotted in x axis
    feat2_id
        Index of feature to be plotted in y axis
    feat1
        Feature to be plotted in x axis
    feat2
        Feature to be plotted in y axis
    num_features
        Total number of features shown in the grid
    color
        Color of points. Default is "black".
    alpha
        Opacity of points. Default is 0.01.

    Returns
    -------
    :
        The minimum and maximum y values for the scatter plot.
    """
    x, y = feat1, feat2
    ymin = y.min()
    ymax = y.max()
    ax.scatter(x, y, s=0.05, c=color, alpha=alpha)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    if feat2_id:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", which="both", length=0.0)
    if feat1_id < num_features - 1:
        ax.tick_params(axis="x", which="both", length=0.0)
    return (ymin, ymax)


def add_correlation_values(
    ax: Axes,
    feat1: np.ndarray,
    feat2: np.ndarray,
) -> None:
    """
    Add annotated correlation values to the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat1
        Feature to be plotted in x axis
    feat2
        Feature to be plotted in y axis
    """
    x, y = feat1, feat2
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0.0)
    ax.tick_params(axis="y", which="both", length=0.0)
    spearman, _ = spstats.spearmanr(x, y)

    rdbu_cmap = plt.colormaps["RdBu"]
    normalized_corr = (spearman + 1) / 2  # type: ignore
    bg_color = rdbu_cmap(normalized_corr)
    ax.set_facecolor(bg_color)
    ax.text(
        0.25,
        0.45,
        f"{spearman:.2f}",
        size=20,
        ha="left",
        transform=ax.transAxes,
    )


def add_feature_histogram(ax: Axes, feat: np.ndarray) -> None:
    """
    Add histogram plot to the diagonal of the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat
        Feature values to be plotted
    """
    ax.set_frame_on(False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.hist(
        feat,
        bins=16,
        density=True,
        histtype="stepfilled",
        color="white",
        edgecolor="black",
    )


def plot_multi_feature_correlations(
    df: pd.DataFrame,
    alpha: float = 0.7,
    cutoff_percent: float = 0,
    dpi: int = 150,
    title: str | None = None,
    output_folder: Path | None = None,
    color: str | list | np.ndarray = "black",
    filename: str = "multi_feature_correlations",
) -> None:
    """
    Create a scatter plot of all the columns in the dataframe.

    Parameters
    ----------
    df
        The dataframe to be plotted
    alpha
        The transparency of the points
    cutoff_percent
        The percentage of the data to be removed from the edges
    dpi
        The resolution of the plot
    title
        The title of the plot
    output_folder
        The folder where the plot will be saved
    color
        The color of the points in the scatter plot.
        Can be provided as a list of colors or a single color.
        Default is "black".
    filename
        The name of the file to save the plot as
    """
    num_features = len(df.columns)
    assert num_features >= 2
    npts = df.shape[0]
    prange = []
    for f in df.columns:
        prange.append(np.nanpercentile(df[f].to_numpy(), [cutoff_percent, 100 - cutoff_percent]))

    # Create a grid of num_featuresxnum_features
    fig, axs = plt.subplots(
        num_features,
        num_features,
        figsize=(2.1 * num_features, 2 * num_features),
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        constrained_layout=True,
    )

    for f1id, f1 in enumerate(df.columns):
        yrange = []
        for f2id, f2 in enumerate(df.columns):
            ax = axs[f1id, f2id]
            y = df[f1].to_numpy()
            x = df[f2].to_numpy()
            valids = np.where(
                (y >= prange[f1id][0])
                & (y <= prange[f1id][1])
                & (x >= prange[f2id][0])
                & (x <= prange[f2id][1])
                & ~np.isnan(y)
                & ~np.isnan(x)
                & ~np.isinf(y)
                & ~np.isinf(x)
            )[0]
            x = x[valids]
            y = y[valids]
            if isinstance(color, str):
                plot_color = [color] * len(x)
            elif isinstance(color, list | np.ndarray):
                plot_color = np.array(color)
                plot_color = plot_color[valids]

            # Make plots
            if f2id < f1id:
                data_range = add_feature_scatter_plot(
                    ax=ax,
                    feat1_id=f1id,
                    feat2_id=f2id,
                    feat1=x,
                    feat2=y,
                    alpha=alpha,
                    color=plot_color,
                    num_features=num_features,
                )
                yrange.append(data_range)
            elif f2id > f1id:
                add_correlation_values(ax=ax, feat1=x, feat2=y)
            else:
                add_feature_histogram(ax=ax, feat=x)

            if f1id == num_features - 1:
                ax.set_xlabel(f2, fontsize=12)
            if not f2id and f1id:
                ax.set_ylabel(f1, fontsize=12)
        if yrange:
            ymin = np.min([ymin for (ymin, _) in yrange])
            ymax = np.max([ymax for (_, ymax) in yrange])
            for f2id in range(len(df.columns)):
                ax = axs[f1id, f2id]
                if f2id < f1id:
                    ax.set_ylim(ymin, ymax)

    rdbu_cmap = plt.colormaps["RdBu"]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=rdbu_cmap), ax=axs, shrink=0.8, pad=0.02
    )
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    if title is not None:
        fig.suptitle(title, fontsize=24)
    else:
        fig.suptitle(f"Total number of points: {npts}", fontsize=24)

    if output_folder is None:
        plt.show()
        return

    save_plot_to_path(
        figure=fig,
        output_path=output_folder,
        figure_name=filename,
        dpi=dpi,
    )


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
) -> None:
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
    x_axis_label
        The name of the x-axis.
    y_axis_label
        The name of the y-axis.
    df_format
        The format of the output DataFrame. If "long", the output DataFrame will have columns:
        - y_axis_label
        - x_axis_label
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

    if df_format not in ("long", "wide-corrcoeff", "wide-pval"):
        raise ValueError(
            f"Unsupported df_format: {df_format}. "
            f"Supported: 'long', 'wide-corrcoeff', 'wide-pval'."
        )

    records = []
    for col_for_y in column_names_for_y_axis:
        for col_for_x in column_names_for_x_axis:
            valid_records = np.isfinite(features_df[[col_for_y, col_for_x]]).all(axis=1)
            corr, pval = spstats.pearsonr(
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

    else:
        # The table is already in the "long" format by default so no changes are necessary.
        pass
    return correlation_df


def get_df_for_feature_correlation_viz(
    dataset_name_list: list[str],
    dataset_info_columns: list[str],
    classical_feature_columns: list[str],
    pc_columns: list[str],
    diffae_feature_columns: list[str],
    model_manifest: ModelManifest,
    run_name: str | None = None,
    seg_feature_manifest_name: str = "live_merged_seg_features",
    timepoint_annotations: list[TimepointAnnotation] | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess the manifests for the given dataset names,
    and return a DataFrame containing the merged features.
    The returned DataFrame may be optionally filtered based on timepoint annotations.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to process.
    dataset_info_columns
        List of columns containing dataset information.
    classical_feature_columns
        List of classical feature column names.
    pc_columns
        List of PCA component column names.
    diffae_feature_columns
        List of DiffAE feature column names.
    model_manifest
        The model manifest containing information about the DiffAE model.
    run_name
        The name of the run to use for loading the manifests.
        If None, the latest run will be used.
    seg_feature_manifest_name
        The name of the segmentation feature manifest to use.
        Default is "live_merged_seg_features".
    timepoint_annotations
        List of timepoint annotations used to filter the DataFrame.
        If None, no filtering will be applied.

    Returns
    -------
    :
        A DataFrame containing the merged features from the specified datasets,
        filtered based on the provided timepoint annotations.
    """
    df_list: list = []
    for dataset_name in dataset_name_list:
        # load and preprocess the different diffae manifests and PCA pipeline
        # NOTE: this takes a little over a minute to load
        merged_feats_df, _, _ = get_preprocessed_manifests_and_km_bounds(
            dataset_name=dataset_name,
            model_manifest=model_manifest,
            run_name=run_name,
            seg_feature_manifest_name=seg_feature_manifest_name,
            datasets_for_bounds=dataset_name_list,
        )

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
        # only if timepoint annotations are provided
        dataset_config = load_dataset_config(dataset_name)
        merged_feats_df = filter_dataframe_by_annotations(
            dataframe=merged_feats_df,
            dataset_config=dataset_config,
            timepoint_annotations=timepoint_annotations,
        )

        # keep only the columns that will be used
        cols_to_keep = (
            dataset_info_columns + classical_feature_columns + diffae_feature_columns + pc_columns
        )

        merged_feats_df = merged_feats_df[cols_to_keep].copy()
        merged_feats_df.rename(columns=get_label_for_column, inplace=True)
        df_list.append(merged_feats_df)
    # merge the DataFrames from all datasets
    df = pd.concat(df_list, ignore_index=True)

    return df
