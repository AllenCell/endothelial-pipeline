from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy import stats as spstats


def add_plot_to_lower_triangle(
    ax: Axes,
    feat1_id: int,
    feat2_id: int,
    feat1: np.ndarray,
    feat2: np.ndarray,
    valids: np.ndarray,
    include_data_points: bool,
    alpha: float,
    num_features: int,
) -> list:
    """
    Add scatter plots to the lower triangle of the num_features X num_features grid.

    Parameters
    ----------
    ax : plt axis
        Matplotlib axis to be used
    feat1_id : int
        Index if feature to be ploted in x axis
    feat2_id : int
        Index of feature to be ploted in y axis
    feat1 : array
        Feature to be ploted in x axis
    feat2 : array
        Feature to be ploted in y axis
    valids : array of bool
        True for points to be ploted
    include_data_points : bool
        If true, data points will be included in the plot
    alpha : float
        Opacity of points
    num_features : int
        Total number of features shown in the grid
    Returns:
    ---------
    :
        Range of the y axis
    """
    x, y = feat1, feat2
    ymin = y[valids].min()
    ymax = y[valids].max()
    if include_data_points:
        ax.plot(x[valids], y[valids], ".", markersize=2, color="black", alpha=alpha)
    if feat2_id:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", which="both", length=0.0)
    if feat1_id < num_features - 1:
        ax.tick_params(axis="x", which="both", length=0.0)
    return [ymin, ymax]


def add_plot_to_upper_triangle(
    ax: Axes,
    feat1: np.ndarray,
    feat2: np.ndarray,
    threshold: float = 0.3,
    measure: Literal["pearson", "spearman"] = "spearman",
) -> None:
    """
    Add plots to the upper triangle of the num_features X num_features grid.
    Parameters
    ----------
    ax : plt axis
        Matplotlib axis to be used
    feat1 : array
        Feature to be ploted in x axis
    feat2 : array
        Feature to be ploted in y axis
    threshold : float
        Threshold for the correlation to be highlighted
    """
    x, y = feat1, feat2
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0.0)
    ax.tick_params(axis="y", which="both", length=0.0)
    pearson, _ = spstats.pearsonr(x, y)
    spearman, _ = spstats.spearmanr(x, y)
    corr_dict = {
        "pearson": pearson,
        "spearman": spearman,
    }
    if corr_dict[measure] > threshold:
        for spine in ax.spines.values():
            spine.set_color("red")
    ax.text(0.05, 0.6, f"Pearson: {pearson:.2f}", size=10, ha="left", transform=ax.transAxes)
    # ax.text(0.05, 0.6, f"P-value: {p_pvalue:.1E}", size=10, ha="left", transform=ax.transAxes)
    ax.text(0.05, 0.4, f"Spearman: {spearman:.2f}", size=10, ha="left", transform=ax.transAxes)
    # ax.text(0.05, 0.2, f"P-value: {s_pvalue:.1E}", size=10, ha="left", transform=ax.transAxes)


def add_plot_to_diagonal(ax: Axes, feat: np.ndarray, valids: np.ndarray, cmap: Colormap) -> None:
    """
    Add histogram plots to the diagonal of the num_features X num_features grid.
    Parameters
    ----------
    ax : plt axis
        Matplotlib axis to be used
    feat : array
        Feature to be ploted
     valids : array of bool
        True for points to be ploted
    cmap : Matplotlib colormap
        To be used in the histogram
    """
    x = feat
    ax.set_frame_on(False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.hist(
        x[valids],
        bins=16,
        density=True,
        histtype="stepfilled",
        color="white",
        edgecolor="black",
        label="Complete",
    )
    ax.hist(
        x[valids],
        bins=16,
        density=True,
        histtype="stepfilled",
        color=cmap(0),
        alpha=0.2,
        label="Incomplete",
    )


def plot_multi_feature_correlations(
    df: pd.DataFrame,
    alpha: float = 0.01,
    off: float = 0,
    include_data_points: bool = True,
    include_outlines: bool = True,
    save: Path | None = None,
    dpi: int = 150,
    title: str | None = None,
) -> None:
    """
    Create a scatter plot of all the columns in the dataframe
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be plotted
    alpha : float
        The transparency of the points
    off : float
        The percentage of the data to be removed from the edges
    include_data_points : bool
        If True, the data points will be included in the scatter plot
    include_outlines : bool
        If True, the axes, histograms and text will be included
    save : str
        The path to save the scatter plot
    dpi : int
        The resolution of the scatter plot
    """
    num_features = len(df.columns)
    assert num_features >= 2
    npts = df.shape[0]
    cmap = plt.colormaps["tab10"]
    prange = []
    for f in df.columns:
        prange.append(np.nanpercentile(df[f].to_numpy(), [off, 100 - off]))

    # Create a grid of num_featuresxnum_features
    fig, axs = plt.subplots(
        num_features,
        num_features,
        figsize=(2 * num_features, 2 * num_features),
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    for f1id, f1 in enumerate(df.columns):
        yrange = []
        # _, f1_label, f1_unit, _ = get_plot_labels_for_metric(f1)
        for f2id, f2 in enumerate(df.columns):
            # _, f2_label, f2_unit, _ = get_plot_labels_for_metric(f2)
            ax = axs[f1id, f2id]
            if not include_outlines:
                ax.axis("off")
            y = df[f1].to_numpy()
            x = df[f2].to_numpy()
            valids = np.where(
                (
                    (y > prange[f1id][0])
                    & (y < prange[f1id][1])
                    & (x > prange[f2id][0])
                    & (x < prange[f2id][1])
                )
            )[0]

            # Make plots
            if f2id < f1id:
                data_range = add_plot_to_lower_triangle(
                    ax=ax,
                    feat1_id=f1id,
                    feat2_id=f2id,
                    feat1=x,
                    feat2=y,
                    valids=valids,
                    include_data_points=include_data_points,
                    alpha=alpha,
                    num_features=num_features,
                )
                yrange.append(data_range)
            elif (f2id > f1id) and include_outlines:
                add_plot_to_upper_triangle(ax=ax, feat1=x, feat2=y)
            elif include_outlines:
                add_plot_to_diagonal(ax=ax, feat=x, valids=valids, cmap=cmap)

            if f1id == num_features - 1:
                # ax.set_xlabel(f"{f2_label} {f2_unit}", fontsize=7)
                ax.set_xlabel(f2, fontsize=12)
            if not f2id and f1id:
                # ax.set_ylabel(f"{f1_label} {f1_unit}", fontsize=7)
                ax.set_ylabel(f1, fontsize=12)
        if yrange:
            ymin = np.min([ymin for (ymin, _) in yrange])
            ymax = np.max([ymax for (_, ymax) in yrange])
            for f2id, f2 in enumerate(df.columns):
                ax = axs[f1id, f2id]
                if f2id < f1id:
                    ax.set_ylim(ymin, ymax)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    if include_outlines:
        if title is not None:
            plt.title(title, fontsize=24)
        else:
            plt.title(f"Total number of points: {npts}", fontsize=24)
    if save is None:
        plt.show()
        return
    plt.savefig(save, dpi=dpi)
