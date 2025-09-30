from typing import Any, Literal

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# set the plot shape to the golden ratio
AX_WIDTH = 4.5
AX_HEIGHT = AX_WIDTH * 2 / 3


def lineplot_of_feats(
    df_group: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    set_xticks: range | None = None,
    set_yticks: range | None = None,
    discrete_xticks: bool = False,
    discrete_yticks: bool = False,
    minor_ticks: Literal["x", "y", "xy"] | None = None,
    kwargs: dict = dict(),
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
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]]
        Set the limits for the x-axis using a tuple of form (x_min, x_max).
        If "min" or "max", the limits will be set to the data min or max.
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]]
        Set the limits for the y-axis using a tuple of form (y_min, y_max).
        If "min" or "max", the limits will be set to the data min or max.
    set_xticks: range | None
        Set the x-ticks for the plot. If None, the default ticks will be used.
    set_yticks: range | None
        Set the y-ticks for the plot. If None, the default ticks will be used.
    discrete_xticks: bool
        If True, the x-ticks will be set to discrete values (integers).
    discrete_yticks: bool
        If True, the y-ticks will be set to discrete values (integers).
    minor_ticks: Literal["x", "y", "xy"] | None
        If "x", "y", or "xy", minor ticks will be added to the respective axes.
    kwargs: dict
        Additional keyword arguments to pass to the seaborn lineplot function.
        This can include parameters like `color`, `style`, `markers`, etc.
    """

    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]

    positions = tuple(pos for pos in df_group.position.unique())
    if len(positions) == 1:
        positions = positions[0]
    fig_title = f"{dataset_name} P{positions}"

    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.lineplot(data=df_group, x=x_column_name, y=y_column_name, ax=ax, **kwargs)

    # adjust the axes limits and tick behavior
    x_min = df_group[x_column_name].min() if x_lims[0] == "min" else x_lims[0]
    x_max = df_group[x_column_name].max() if x_lims[1] == "max" else x_lims[1]
    y_min = df_group[y_column_name].min() if y_lims[0] == "min" else y_lims[0]
    y_max = df_group[y_column_name].max() if y_lims[1] == "max" else y_lims[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if minor_ticks:
        if "x" in minor_ticks:
            ax.xaxis.minorticks_on()
        if "y" in minor_ticks:
            ax.yaxis.minorticks_on()
    if set_xticks:
        ax.set_xticks(set_xticks)
    if set_yticks:
        ax.set_yticks(set_yticks)
    if discrete_xticks:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True))
    if discrete_yticks:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(plt.MaxNLocator(integer=True))

    # set figure and axes titles
    ax.set_title(fig_title)
    ax.set_xlabel(x_label or x_column_name)
    ax.set_ylabel(y_label or y_column_name)
    plt.tight_layout()

    return fig, ax


def hist_2D_of_feats(
    df_group: pd.DataFrame,
    x_column_name: str,
    y_column_name: str,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]] = (None, None),
    set_xticks: range | None = None,
    set_yticks: range | None = None,
    discrete_xticks: bool = False,
    discrete_yticks: bool = False,
    minor_ticks: Literal["x", "y", "xy"] | None = None,
    bin_width: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    tight_layout: bool = True,
    cmap: str = "viridis",
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
    x_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]]
        Set the limits for the x-axis using a tuple of form (x_min, x_max).
        If "min" or "max", the limits will be set to the data min or max.
    y_lims: tuple[float | None | Literal["min"], float | None | Literal["max"]]
        Set the limits for the y-axis using a tuple of form (y_min, y_max).
        If "min" or "max", the limits will be set to the data min or max.
    set_xticks: range | None
        Set the x-ticks for the plot. If None, the default ticks will be used.
    set_yticks: range | None
        Set the y-ticks for the plot. If None, the default ticks will be used.
    discrete_xticks: bool
        If True, the x-ticks will be set to discrete values (integers).
    discrete_yticks: bool
        If True, the y-ticks will be set to discrete values (integers).
    minor_ticks: Literal["x", "y", "xy"] | None
        If "x", "y", or "xy", minor ticks will be added to the respective axes.
    bin_width: tuple[int, int] | None
        Set the bin width for the histogram using a (width_x, width_y)
        tuple. If None, the default bin width will be used.
    figsize: tuple[float, float] | None
        Set the figure size using a (width, height) tuple.
        If None, the default figure size will be used.
    tight_layout: bool
        If True, plt.tight_layout() will be called to adjust the figure layout.
    colormap: str
        The colormap to use for the histogram.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
    """

    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]

    positions = tuple(pos for pos in df_group.position.unique())
    if len(positions) == 1:
        positions = positions[0]
    fig_title = f"{dataset_name} P{positions}"

    if figsize is None:
        figsize = (AX_WIDTH, AX_HEIGHT)
    else:
        figsize = figsize
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df_group,
        x=x_column_name,
        y=y_column_name,
        binwidth=bin_width,
        cmap=cmap,
        ax=ax,
    )

    # adjust the axes limits and tick behavior
    x_min = df_group[x_column_name].min() if x_lims[0] == "min" else x_lims[0]
    x_max = df_group[x_column_name].max() if x_lims[1] == "max" else x_lims[1]
    y_min = df_group[y_column_name].min() if y_lims[0] == "min" else y_lims[0]
    y_max = df_group[y_column_name].max() if y_lims[1] == "max" else y_lims[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if minor_ticks:
        if "x" in minor_ticks:
            ax.xaxis.minorticks_on()
        if "y" in minor_ticks:
            ax.yaxis.minorticks_on()
    if set_xticks:
        ax.set_xticks(set_xticks)
    if set_yticks:
        ax.set_yticks(set_yticks)
    if discrete_xticks:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(plt.MaxNLocator(integer=True))
    if discrete_yticks:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(plt.MaxNLocator(integer=True))

    # set figure and axes titles
    ax.set_title(fig_title)
    ax.set_xlabel(x_label or x_column_name)
    ax.set_ylabel(y_label or y_column_name)
    if tight_layout:
        plt.tight_layout()

    return fig, ax


def mark_parallel(ax: plt.Axes, color: str = "black") -> plt.Axes:
    """
    Draws horizontal lines at -180, 0, and 180 degrees
    to mark the parallel angles.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to mark the angles on.

    Returns
    -------
    plt.Axes
        The axes object with the marked angles.
    """
    parallel_angles = [-180, 0, 180]
    for ang in parallel_angles:
        ax.axhline(ang, color=color, linestyle="--", linewidth=1)
    return ax


def mark_perpendicular(ax: plt.Axes, color: str = "black") -> plt.Axes:
    """
    Draws horizontal lines at -90 and 90 degrees to mark
    the perpendicular angles.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to mark the angles on.

    Returns
    -------
    plt.Axes
        The axes object with the marked angles.
    """
    perpendicular_angles = [-90, 90]
    for ang in perpendicular_angles:
        ax.axhline(ang, color=color, linestyle=":", linewidth=1)
    return ax


def get_seg_feat_plot_args() -> dict[str, dict[str, Any]]:
    """
    Returns a dictionary of dictionaries representing the arguments
    for plotting segmentation features.
    The first level keys are a short name for the feature, and the
    second level keys are the arguments for plotting.
    Arguments include:
        - column_name: The name of the column in the DataFrame.
        - label: The label to use for the feature in the plot.
        - lims: A tuple of (min, max) values for the axis limits.
        - bin_width: The width of the bins for the histogram.
        - ticks: A range of ticks to use for the axis.
        - discrete_ticks: Whether the ticks should be discrete (True) or continuous (False).

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary containing the plotting arguments for each feature.
    """
    feat_args: dict[str, dict[str, Any]] = {
        "time_hrs": {
            "column_name": "time_hours",
            "label": "Time (h)",
            "lims": (0, "max"),
            "bin_width": 0.5,
            "ticks": range(0, 49, 12),
            "discrete_ticks": False,
        },
        "alignment_deg": {
            "column_name": "alignment_deg_rel_to_flow",
            "label": "Alignment (deg)",
            "lims": (0, 90),
            "bin_width": 1,
            "ticks": range(0, 91, 15),
            "discrete_ticks": False,
        },
        "orientation_deg": {
            "column_name": "orientation_deg",
            "label": "Orientation (deg)",
            "lims": (-180, 180),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
            "discrete_ticks": False,
        },
        "nematic_order": {
            "column_name": "nematic_order",
            "label": "Nematic Order",
            "lims": (-1, 1),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "eccentricity": {
            "column_name": "eccentricity",
            "label": "Eccentricity",
            "lims": (0, 1),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "aspect_ratio": {
            "column_name": "aspect_ratio",
            "label": "Aspect Ratio",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "area_um2": {
            "column_name": "area (um**2)",
            "label": "Area (μm²)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "num_neighbors": {
            "column_name": "number_of_neighbors",
            "label": "Number of Neighbors",
            "lims": (0, "max"),
            "bin_width": 1,
            "ticks": None,
            "discrete_ticks": True,
        },
        "centroid_velocity_magnitude": {
            "column_name": "centroid_velocity_magnitude",
            "label": "Centroid Velocity Magnitude (μm/min)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "centroid_velocity_orientation_deg": {
            "column_name": "centroid_velocity_angle_deg",
            "label": "Centroid Velocity\nOrientation (deg)",
            "lims": (-180, 181),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
            "discrete_ticks": False,
        },
        "cell_nuc_orientation_deg": {
            "column_name": "nuc_pos_rel_cell_angle_deg",
            "label": "Nuclei Orientation\nRel. to Flow (deg)",
            "lims": (-180, 180),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
            "discrete_ticks": False,
        },
        "cell_nuc_dist": {
            "column_name": "nuc_pos_rel_cell_magnitude",
            "label": "Nuclei-Cell Centroid Distance (px)",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "num_nuclei": {
            "column_name": "total_nuclei_count_at_T",
            "label": "Number of Nuclei",
            "lims": (0, None),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": True,
        },
        "num_tracks": {
            "column_name": "num_tracks_at_T",
            "label": "Number of Tracks",
            "lims": (0, None),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": True,
        },
        "num_nuclei_in_crop": {
            "column_name": "num_nuclei_in_crop",
            "label": "Number of Nuclei in Crop",
            "lims": (0, None),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": True,
        },
        "cell_fluorescence_mean": {
            "column_name": "cell_fluorescence_mean (a.u.)",
            "label": "Mean Cell Fluorescence",
            "lims": (0, "max"),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "cell_solidity": {
            "column_name": "cell_solidity",
            "label": "Cell Solidity",
            "lims": (0, 1),
            "bin_width": None,
            "ticks": None,
            "discrete_ticks": False,
        },
        "nuc_orientation_deg_rel_migration": {
            "column_name": "cell_nuc_orientation_deg_rel_to_migration",
            "label": "Nuclei Orientation\nRel. to Migration (deg)",
            "lims": (-180, 180),
            "bin_width": 5,
            "ticks": range(-180, 181, 90),
            "discrete_ticks": False,
        },
    }

    return feat_args
