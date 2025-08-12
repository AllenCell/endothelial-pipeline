"""Methods to visualize multi-feature correlations"""

# Creates an n_features X n_features grid of plots with:
# 1) Scatter plots of features on the lower triangle
# 2) Feature histograms on the diagonal
# 3) Correlation values on the upper triangle

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy import stats as spstats

from src.endo_pipeline.io.output import save_plot_to_path


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
    ax.scatter(x, y, s=0.01, c=color, alpha=alpha)
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
    alpha: float = 0.01,
    cutoff_percent: float = 0,
    dpi: int = 150,
    title: str | None = None,
    output_folder: Path | None = None,
    color: str | list | np.ndarray = "black",
    filename: str = "multi_feature_correlations",
) -> None:
    """
    Create a scatter plot of all the columns in the dataframe

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
                (
                    (y >= prange[f1id][0])
                    & (y <= prange[f1id][1])
                    & (x >= prange[f2id][0])
                    & (x <= prange[f2id][1])
                    & ~np.isnan(y)
                    & ~np.isnan(x)
                    & ~np.isinf(y)
                    & ~np.isinf(x)
                )
            )[0]
            x = x[valids]
            y = y[valids]
            if isinstance(color, str):
                plot_color = [color] * len(x)
            elif isinstance(color, (list, np.ndarray)):
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
            for f2id, f2 in enumerate(df.columns):
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
