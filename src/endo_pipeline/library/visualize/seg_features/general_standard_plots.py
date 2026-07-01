import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from endo_pipeline.settings.column_metadata import ColumnMetadata, ColumnType
from endo_pipeline.settings.column_names import ColumnName as Column

# set the plot shape to the golden ratio
AX_WIDTH = 4.5
AX_HEIGHT = AX_WIDTH * 2 / 3


def adjust_axes_ticks(
    ax: Axes,
    x_data: pd.Series,
    y_data: pd.Series,
    x_feature_metadata: ColumnMetadata,
    y_feature_metadata: ColumnMetadata,
    x_minor_ticks: bool = False,
    y_minor_ticks: bool = False,
) -> None:
    """
    Adjust axis ticks based on given feature data and metadata.

    Parameters
    ----------
    ax
        The axes instance.
    x_data
        Data for the x-axis.
    y_data
        Data for the y-axis.
    x_feature_metadata
        Feature metadata for the x-axis data.
    y_feature_metadata
        Feature metadata for the y-axis data.
    x_minor_ticks
        True to include minor ticks on the x-axis, False otherwise.
    y_minor_ticks
        True to include minor ticks on the y-axis, False otherwise.
    """

    x_min = x_data.min() if x_feature_metadata.min == "min" else x_feature_metadata.min
    x_max = x_data.max() if x_feature_metadata.max == "max" else x_feature_metadata.max
    y_min = y_data.min() if y_feature_metadata.min == "min" else y_feature_metadata.min
    y_max = y_data.max() if y_feature_metadata.max == "max" else y_feature_metadata.max

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if x_minor_ticks:
        ax.xaxis.minorticks_on()

    if y_minor_ticks:
        ax.yaxis.minorticks_on()

    if x_feature_metadata.ticks is not None:
        ax.set_xticks(x_feature_metadata.ticks)

    if y_feature_metadata.ticks is not None:
        ax.set_yticks(y_feature_metadata.ticks)

    if x_feature_metadata.type == ColumnType.DISCRETE:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(MaxNLocator(integer=True))

    if y_feature_metadata.type == ColumnType.DISCRETE:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(MaxNLocator(integer=True))


def plot_line_of_features(
    df: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_feature_metadata: ColumnMetadata,
    y_feature_metadata: ColumnMetadata,
    x_minor_ticks: bool = False,
    y_minor_ticks: bool = False,
    kwargs: dict = {},
) -> tuple[Figure, Axes]:
    """
    Plot line plot of given features.

    Parameters
    ----------
    df
        Dataframe containing data to plot. Should contain only a single dataset.
    x_column_name
        The column name for the x-axis data.
    y_column_name
        The column name for the y-axis data.
    x_feature_metadata
        Feature metadata for the x-axis data.
    y_feature_metadata
        Feature metadata for the y-axis data.
    x_minor_ticks
        True to include minor ticks on the x-axis, False otherwise.
    y_minor_ticks
        True to include minor ticks on the y-axis, False otherwise.
    kwargs
        Additional keyword arguments to pass to the seaborn lineplot function.
        This can include parameters like `color`, `style`, `markers`, etc.
    """

    unique_datasets = df[Column.DATASET].unique()
    if len(unique_datasets) != 1:
        raise ValueError(f"Only a single dataset can be plotted. Given: '{unique_datasets}'")
    dataset_name = unique_datasets[0]

    unique_positions = df[Column.POSITION].unique()
    if len(unique_positions) != 1:
        fig_title = f"{dataset_name} P{unique_positions[0]}"
    else:
        fig_title = f"{dataset_name}"

    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT), layout="constrained")
    sns.lineplot(data=df, x=x_column_name, y=y_column_name, ax=ax, **kwargs)

    # adjust the axes limits and tick behavior
    adjust_axes_ticks(
        ax=ax,
        x_data=df[x_column_name],
        y_data=df[y_column_name],
        x_feature_metadata=x_feature_metadata,
        y_feature_metadata=y_feature_metadata,
        x_minor_ticks=x_minor_ticks,
        y_minor_ticks=y_minor_ticks,
    )

    # set figure and axes titles
    ax.set_title(fig_title)
    ax.set_xlabel(x_feature_metadata.label_with_unit or x_column_name)
    ax.set_ylabel(y_feature_metadata.label_with_unit or y_column_name)

    return fig, ax


def plot_histogram_of_features(
    df: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_feature_metadata: ColumnMetadata,
    y_feature_metadata: ColumnMetadata,
    x_minor_ticks: bool = False,
    y_minor_ticks: bool = False,
    figsize: tuple[float, float] | None = None,
    colormap_name: str = "viridis",
) -> tuple[Figure, Axes]:
    """
    Plot 2D histogram of given features.

    df
        Dataframe containing data to plot. Should contain only a single dataset.
    x_column_name
        The column name for the x-axis data.
    y_column_name
        The column name for the y-axis data.
    x_feature_metadata
        Feature metadata for the x-axis data.
    y_feature_metadata
        Feature metadata for the y-axis data.
    x_minor_ticks
        True to include minor ticks on the x-axis, False otherwise.
    y_minor_ticks
        True to include minor ticks on the y-axis, False otherwise.
    figsize
        Set the figure size using a (width, height) tuple.
        If None, the default figure size will be used.
    colormap_name
        Name of colormap to use for the histogram.

    Returns
    -------
    :
        The figure and axes objects.
    """

    unique_datasets = df[Column.DATASET].unique()
    if len(unique_datasets) != 1:
        raise ValueError(f"Only a single dataset can be plotted. Given: '{unique_datasets}'")
    dataset_name = unique_datasets[0]

    unique_positions = df[Column.POSITION].unique()
    if len(unique_positions) != 1:
        fig_title = f"{dataset_name} P{unique_positions[0]}"
    else:
        fig_title = f"{dataset_name}"

    fig, ax = plt.subplots(figsize=figsize or (AX_WIDTH, AX_HEIGHT))

    if x_feature_metadata.bin_width and y_feature_metadata.bin_width:
        binwidth = (x_feature_metadata.bin_width, y_feature_metadata.bin_width)
    else:
        binwidth = None

    sns.histplot(
        data=df,
        x=x_column_name,
        y=y_column_name,
        binwidth=binwidth,
        cmap=colormap_name,
        ax=ax,
    )
    ax.set_box_aspect(1)

    # change the background color to grey
    ax.set_facecolor("grey")

    # adjust the axes limits and tick behavior
    adjust_axes_ticks(
        ax=ax,
        x_data=df[x_column_name],
        y_data=df[y_column_name],
        x_feature_metadata=x_feature_metadata,
        y_feature_metadata=y_feature_metadata,
        x_minor_ticks=x_minor_ticks,
        y_minor_ticks=y_minor_ticks,
    )

    # set figure and axes titles
    ax.set_title(fig_title)
    ax.set_xlabel(x_feature_metadata.label_with_unit or x_column_name)
    ax.set_ylabel(y_feature_metadata.label_with_unit or y_column_name)

    return fig, ax


def mark_parallel(ax: Axes, color: str = "black") -> None:
    """
    Draws horizontal lines at -180, 0, and 180 degrees to mark parallel angles.

    Parameters
    ----------
    Ax
        The axes object to mark the angles on.
    """

    parallel_angles = [-180, 0, 180]
    for ang in parallel_angles:
        ax.axhline(ang, color=color, linestyle="--", linewidth=1)


def mark_perpendicular(ax: Axes, color: str = "black") -> None:
    """
    Draws horizontal lines at -90 and 90 degrees to mark perpendicular angles.

    Parameters
    ----------
    ax
        The axes object to mark the angles on.
    """

    perpendicular_angles = [-90, 90]
    for ang in perpendicular_angles:
        ax.axhline(ang, color=color, linestyle=":", linewidth=1)
