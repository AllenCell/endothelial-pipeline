from typing import Any, Literal

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# set the plot shape to the golden ratio
AX_WIDTH = 4.5
AX_HEIGHT = AX_WIDTH * 2 / 3


def lineplot_per_dataset(
    df_group: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    set_xticks: range | None = None,
    set_yticks: range | None = None,
    hlines: list[float] | None = None,
    vlines: list[float] | None = None,
    kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    This function will save a standardized lineplot from the dataframe df_group.
    x_key and y_key are the column names that you want to plot along the x-axis
    and y-axis, respectively.
    df_group is expected to contain a single dataset and a single position.

    Parameters
    ----------
    df_group : pd.DataFrame
        The dataframe containing the data to plot.
        Should contain only a single dataset.
    x_column_name : str
        The column name for the x-axis data.
    y_column_name : str
        The column name for the y-axis data.
    x_label : str | None, optional
        The label for the x-axis, will use x_key as the default.
    y_label : str | None, optional
        The label for the y-axis, will use y_key as the default.
    x_lims: tuple | Literal["tight"]
        Set the limits for the x-axis using a tuple of form (x_min, x_max).
        If "tight", the limits will be set to the data min and max.
    y_lims: tuple | Literal["tight"]
        Set the limits for the y-axis using a tuple of form (y_min, y_max).
        If "tight", the limits will be set to the data min and max.
    """

    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]

    positions = tuple(pos for pos in df_group.position.unique())
    if len(positions) == 1:
        positions = positions[0]
    fig_title = f"{dataset_name} P{positions}"

    x_min = df_group[x_column_name].min() if x_lims[0] == "min" else x_lims[0]
    x_max = df_group[x_column_name].max() if x_lims[1] == "max" else x_lims[1]
    y_min = df_group[y_column_name].min() if y_lims[0] == "min" else y_lims[0]
    y_max = df_group[y_column_name].max() if y_lims[1] == "max" else y_lims[1]

    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    ax.set_title(fig_title)
    sns.lineplot(data=df_group, x=x_column_name, y=y_column_name, ax=ax, kwargs=kwargs)
    if set_xticks:
        ax.set_xticks(set_xticks)
    if set_yticks:
        ax.set_yticks(set_yticks)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label or x_column_name)
    ax.set_ylabel(y_label or y_column_name)
    plt.tight_layout()

    return fig, ax


def hist_2D_per_dataset(
    df_group: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    set_xticks: range | None = None,
    set_yticks: range | None = None,
    bin_width: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    df_group : pd.DataFrame
        The dataframe containing the data to plot.
        Should contain only a single dataset.
    x_column_name : str
        The column name for the x-axis data.
    y_column_name : str
        The column name for the y-axis data.
    x_label : str | None, optional
        The label for the x-axis, will use x_key as the default.
    y_label : str | None, optional
        The label for the y-axis, will use y_key as the default.
    x_lims: tuple | Literal["tight"]
        Set the limits for the x-axis using a tuple of form (x_min, x_max).
        If "tight", the limits will be set to the data min and max.
    y_lims: tuple | Literal["tight"]
        Set the limits for the y-axis using a tuple of form (y_min, y_max).
        If "tight", the limits will be set to the data min and max.
    bin_width: tuple[int, int] | None
        Set the bin width for the histogram using a (width_x, width_y)
        tuple. If None, the default bin width will be used.
    """

    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]

    positions = tuple(pos for pos in df_group.position.unique())
    if len(positions) == 1:
        positions = positions[0]
    fig_title = f"{dataset_name} P{positions}"

    x_min = df_group[x_column_name].min() if x_lims[0] == "min" else x_lims[0]
    x_max = df_group[x_column_name].max() if x_lims[1] == "max" else x_lims[1]
    y_min = df_group[y_column_name].min() if y_lims[0] == "min" else y_lims[0]
    y_max = df_group[y_column_name].max() if y_lims[1] == "max" else y_lims[1]

    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    ax.set_title(fig_title)
    sns.histplot(
        data=df_group,
        x=x_column_name,
        y=y_column_name,
        binwidth=bin_width,
        ax=ax,
    )
    if set_xticks:
        ax.set_xticks(set_xticks)
    if set_yticks:
        ax.set_yticks(set_yticks)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label or x_column_name)
    ax.set_ylabel(y_label or y_column_name)
    plt.tight_layout()

    return fig, ax


def mark_parallel(ax: plt.Axes) -> plt.Axes:
    parallel_angles = [-180, 0, 180]
    for ang in parallel_angles:
        ax.axhline(ang, color="black", linestyle="--", linewidth=1)
    return ax


def mark_perpendicular(ax: plt.Axes) -> plt.Axes:
    perpendicular_angles = [-90, 90]
    for ang in perpendicular_angles:
        ax.axhline(ang, color="black", linestyle=":", linewidth=1)
    return ax


def get_seg_feat_plot_args() -> dict[str, dict[str, Any]]:
    feat_args: dict[str, dict[str, Any]] = {
        "time_hrs": {
            "column_name": "time_hours",
            "label": "Time (h)",
            "lims": (0, "max"),
            "bin_width": 0.5,
            "ticks": None,
        },
        "alignment_deg": {
            "column_name": "alignment_deg_rel_to_flow",
            "label": "Alignment (deg)",
            "lims": (0, 90),
            "bin_width": 1,
            "ticks": range(0, 91, 15),
        },
        "orientation_deg": {
            "column_name": "orientation_deg",
            "label": "Orientation (deg)",
            "lims": (-180, 180),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
        },
        "nematic_order": {
            "column_name": "nematic_order",
            "label": "Nematic Order",
            "lims": (0, 1),
            "bin_width": None,
            "ticks": None,
        },
        "eccentricity": {
            "column_name": "eccentricity",
            "label": "Eccentricity",
            "lims": (0, 1),
            "bin_width": None,
            "ticks": None,
        },
        "aspect_ratio": {
            "column_name": "aspect_ratio",
            "label": "Aspect Ratio",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
        },
        "area_um2": {
            "column_name": "area (um**2)",
            "label": "Area (μm²)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
        },
        "num_neighbors": {
            "column_name": "number_of_neighbors",
            "label": "Number of Neighbors",
            "lims": (0, None),
            "bin_width": 1,
            "ticks": None,
        },
        "centroid_velocity_magnitude": {
            "column_name": "centroid_velocity_magnitude",
            "label": "Centroid Velocity Magnitude (μm/min)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
        },
        "centroid_velocity_orientation_deg": {
            "column_name": "centroid_velocity_angle_deg",
            "label": "Centroid Velocity Orientation (deg)",
            "lims": (-180, 181),
            "bin_width": 5,
            "ticks": range(0, 181, 90),
        },
        "num_nuclei": {
            "column_name": "number_of_nuclei",
            "label": "Number of Nuclei",
            "lims": (0, None),
        },
        "cell_nuc_orientation_deg": {
            "column_name": "nuclei_orientation_rel_cell_deg",
            "label": "Nuclei Orientation Relative to Cell (deg)",
            "lims": (-180, 180),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
        },
        "cell_nuc_dist": {
            "column_name": "nuclei_rel_cell_centroid_magnitude",
            "label": "Nuclei Relative to Cell Centroid (px)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
        },
        "num_nuclei": {
            "column_name": "total_nuclei_count_at_T",
            "label": "Total Nuclei Count at T",
            "lims": (0, None),
            "bin_width": None,
            "ticks": None,
        },
        "num_tracks": {
            "column_name": "num_tracks_at_T",
            "label": "Number of Tracks at T",
            "lims": (0, None),
            "bin_width": None,
            "ticks": None,
        },
    }

    return feat_args
