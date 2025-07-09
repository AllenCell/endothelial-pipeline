from pathlib import Path
from typing import Literal

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# set the plot shape to the golden ratio
AX_WIDTH = 4.5
AX_HEIGHT = AX_WIDTH * 2 / 3


def lineplot_per_position(
    df_group: pd.DataFrame,
    x_key: str,
    y_key: str,
    filepath_out: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple = (None, None),
    y_lims: tuple = (None, None),
    show_plot: bool = False,
) -> None:
    """
    This function will save a standardized lineplot from the dataframe df_group.
    x_key and y_key are the column names that you want to plot along the x-axis
    and y-axis, respectively.
    df_group is expected to contain a single dataset and a single position.

    Parameters
    ----------
    df_group : pd.DataFrame
        The dataframe containing the data to plot.
        Should contain only a single dataset and a single position.
    x_key : str
        The column name for the x-axis data.
    y_key : str
        The column name for the y-axis data.
    filepath_out : str | Path
        The file path where the plot will be saved.
    x_label : str | None, optional
        The label for the x-axis, will use x_key as the default.
    y_label : str | None, optional
        The label for the y-axis, will use y_key as the default.
    x_lims : tuple, optional
        The limits for the x-axis as a (min, max) tuple.
    y_lims : tuple, optional
        The limits for the y-axis as a (min, max) tuple.
    show_plot : bool, optional
        Whether to show the plot after saving it. Default is False.
    """

    num_positions = df_group["position"].nunique()
    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]
    assert (
        num_positions == 1
    ), f'Only a single position allowed in df_group, positions found: {df_group["position"].unique()}'
    position = df_group["position"].unique()[0]

    fig, ax = plt.subplots(nrows=num_positions, figsize=(AX_WIDTH, AX_HEIGHT))
    ax.set_title(f"{dataset_name} P{position}")
    sns.lineplot(data=df_group, x=x_key, y=y_key, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    plt.tight_layout()
    fig.savefig(filepath_out, bbox_inches="tight")

    if not show_plot:
        plt.close(fig)
    return


def hist_2D_per_position(
    df_group: pd.DataFrame,
    x_key: str,
    y_key: str,
    filepath_out: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple | Literal["tight"] = (None, None),
    y_lims: tuple | Literal["tight"] = (None, None),
    bin_width: tuple[int, int] | None = None,
    kwargs: dict | None = None,
    show_plot: bool = False,
) -> None:
    """
    x_lims: tuple | Literal["tight"]
        Set the limits for the x-axis.
        If "tight", the limits will be set to the data min and max.
    bin_width: tuple[int, int] | None
        Set the bin width for the histogram using a (width_x, width_y)
        tuple. If None, the default bin width will be used.
    """

    num_positions = df_group["position"].nunique()
    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]
    assert (
        num_positions == 1
    ), f'Only a single position allowed in df_group, positions found: {df_group["position"].unique()}'
    position = df_group["position"].unique()[0]

    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=df_group,
        x=x_key,
        y=y_key,
        binwidth=bin_width,
        ax=ax,
    )
    ax.set_yticks(range(0, 91, 15))
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Alignment (deg)")
    plt.tight_layout()
    fig.savefig(filepath_out, bbox_inches="tight")
    plt.close(fig)
