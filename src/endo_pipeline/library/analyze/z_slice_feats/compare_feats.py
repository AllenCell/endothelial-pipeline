from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from src.endo_pipeline.library.analyze.diffae_manifest import get_pc_column_names
from src.endo_pipeline.library.analyze.immunofluorescence.plot import bootstrap_confidence_cov
from src.endo_pipeline.library.visualize import viz_base


def calc_stats(df: pd.DataFrame, feature: str) -> tuple:
    """
    Calculate statistical metrics for a given feature in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the data.
    feature : str
        The feature/column name for which to calculate statistics.

    Returns
    -------
    tuple
        A tuple containing:
        - mean : The mean of the feature.
        - cov : The coefficient of variation (COV) of the feature.
        - low : The lower bound of the bootstrap confidence interval for COV.
        - high : The upper bound of the bootstrap confidence interval for COV.
    """
    mean = np.mean(df[feature])
    cov = np.std(df[feature]) / mean
    low, high = bootstrap_confidence_cov(df, feature)
    return mean, cov, low, high


def feature_density(
    df_all: pd.DataFrame,
    feature: str,
    xlim: np.ndarray,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Plot the probability density distribution of a feature for each position in the DataFrame.

    Parameters
    ----------
    df_all
        The input DataFrame containing the data.
    feature
        The feature/column name for which to plot the density.
    xlim
        The x-axis limits for the plot.
    title
        Optional; the title of the plot.

    Returns
    -------
    :
        The figure object of the plot.
    :
        The axis object of the plot.
    """

    fig = plt.figure(figsize=(15, 6))

    for position, df_position in df_all.groupby("position"):
        mean, cov, low, high = calc_stats(df_position, feature)
        label = (
            f"Pos={position}, "
            f"N={len(df_position)}, Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
        )
        ax = sb.kdeplot(df_position[feature], label=label, alpha=0.85)

        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.set_xlim(xlim[0], xlim[1])

        if title is not None:
            ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_scatter_by_position_and_frame(
    df: pd.DataFrame,
    target_frame: int,
    bounds: list,
    info: str | None = None,
    dataset_name: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot scatter plots of principal components (PC1 vs PC2 and PC1 vs PC3)
    for a specific frame and grouped by position.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the data.
    target_frame : int
        The frame number to filter the data by.
    bounds : list
        A list of bounds for the x and y axes:
        - bounds[0] : x-axis limits for both plots.
        - bounds[1] : y-axis limits for PC1 vs PC2.
        - bounds[2] : y-axis limits for PC1 vs PC3.
    info : str, optional
        Additional information to include in the plot title. Defaults to None.
    dataset_name : str, optional
        The name of the dataset to include in the plot title. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing:
        - fig : The matplotlib figure object.
        - ax : An array of matplotlib axes objects for the subplots.
    """

    fig, ax = viz_base.init_subplots(figsize=(15, 5))
    pc_column_names = get_pc_column_names(df, [0, 1, 2])

    for position, df_pos in df.groupby("position"):
        df_ = df_pos[df_pos["frame_number"] == target_frame]

        ax[0].scatter(df_[pc_column_names[0]], df_[pc_column_names[1]], s=20)
        ax[1].scatter(df_[pc_column_names[0]], df_[pc_column_names[2]], s=20, label=position)

    ax[0].set_xlim(bounds[0])
    ax[0].set_ylim(bounds[1])
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    ax[1].set_xlim(bounds[0])
    ax[1].set_ylim(bounds[2])
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC3")

    ax[1].legend(loc=(1.05, 0.75))
    fig.suptitle(f"{dataset_name}, {info}, T={target_frame} (frames)")

    return fig, ax
