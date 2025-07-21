from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from numpy.typing import ArrayLike

from src.endo_pipeline.io import save_plot_to_path


def get_common_plot_range(
    fixed_features: pd.DataFrame,
    live_features: pd.DataFrame,
    lagged_live_features: pd.DataFrame,
    truncated_live_features: pd.DataFrame,
    pc: int,
) -> tuple[float, float]:
    """
    Get common plot ranges for each PC.

    Parameters
    ----------
    fixed_features : pd.DataFrame
        Dataframe containing PCs for fixed data
    live_features : pd.DataFrame
        Dataframe containing PCs for live data
    lagged_live_features : pd.DataFrame
        Dataframe containing time-lagged PC values for live data
    truncated_live_features : pd.DataFrame
        Dataframe containing original live data PC values truncated to remove the rows that were shifted out by the lag
    pc : int
        PC to analyze

    Returns
    -------
    x_min, x_max : tuple[float, float]
        Common plot ranges for fixed and live data for the specified PC
    """
    x_min = min(
        fixed_features[f"pc{pc}"].min(),
        live_features[f"pc{pc}"].min(),
        lagged_live_features[f"pc{pc}"].min(),
        truncated_live_features[f"pc{pc}"].min(),
    )
    x_max = max(
        fixed_features[f"pc{pc}"].max(),
        live_features[f"pc{pc}"].max(),
        lagged_live_features[f"pc{pc}"].max(),
        truncated_live_features[f"pc{pc}"].max(),
    )
    return x_min, x_max


def plot_paired_fixed_live_validation_features(
    save_path: Path,
    pc: int,
    raw_data: tuple[ArrayLike, ArrayLike],
    paired_validation_features: tuple[Any, Any, Any, Any, Any, Ellipse],
    color_list: list[str] = ["#5F9ED1", "#FF800E", "#C85200"],
    lagged_live_validation: bool = False,
    axmin: float | None = None,
    axmax: float | None = None,
) -> None:
    """
    Plot the raw fixed and live data for a given PC along with a unity line for reference
    and the validation features, including the 2-sigma confidence ellipse, linear
    model mapping between fixed and live data and the error bar for the given PC.

    Parameters
    ----------
    save_path : Path
        Local path to parent directory where results are saved
    pc : int
        Number PC (1-8) to analyze
    raw_data : tuple
        Live (first element) and fixed (second element) PC data
    paired_validation_features : tuple
        Set of all validation needed for plotting
    color_list : list
        List of hex codes for three colors used in plots
    lagged_live_validation : bool
        Flag to plot time-lagged live validation features in place of fixed feautures
    axmin: float | None
        Minimum value for x and y axes. If None, it is calculated from the raw data.
    axmax: float | None
        Maximum value for x and y axes. If None, it is calculated from the raw data.
    """

    # Get raw fixed (y) and live (x) PC data and its lower and upper limits
    x, y = raw_data

    # Get all validation features
    center, height, angle, slope, intercept, ellipse = paired_validation_features

    # Create scatter plot of PC data from two experiments
    plt.clf()
    ax = plt.gca()

    # Plot unity line
    plt.plot([axmin, axmax], [axmin, axmax], c="gray", linestyle="--", label="Unity line")

    # Plot raw data
    ax.scatter(x, y, s=0.5, c="black", alpha=0.1)

    # Plot confidence ellipse
    ax.add_patch(ellipse)

    # Plot linear model along major axis of ellipse
    y_model_min = slope * axmin + intercept
    y_model_max = slope * axmax + intercept
    plt.plot(
        [axmin, axmax],
        [y_model_min, y_model_max],
        color=color_list[0],
        linewidth=2,
        label=f"y={slope:.2f}x+{intercept:.2f}",
    )

    # Plot line along minor axis of ellipse
    minor_axis_length = height / 2
    minor_axis_x1 = center[0] + (minor_axis_length * np.cos(np.radians(angle + 90)))
    minor_axis_y1 = center[1] + (minor_axis_length * np.sin(np.radians(angle + 90)))
    minor_axis_x2 = center[0] + (minor_axis_length * np.cos(np.radians(angle - 90)))
    minor_axis_y2 = center[1] + (minor_axis_length * np.sin(np.radians(angle - 90)))
    plt.plot(
        [minor_axis_x1, minor_axis_x2],
        [minor_axis_y1, minor_axis_y2],
        color=color_list[1],
        label="Minor axis",
    )

    # Plot error bar as y-projection of minor axis
    y_error_bar_length = np.abs(minor_axis_y1 - minor_axis_y2)
    plt.plot(
        [center[0], center[0]],
        [minor_axis_y2, minor_axis_y1],
        color=color_list[2],
        linewidth=3,
        label=f"Error bar = {y_error_bar_length:.2f}",
    )

    # Add labels
    plt.legend(loc="upper left")

    if lagged_live_validation:
        plt.xlabel(f"PC{pc} reference data")
        plt.ylabel(f"PC{pc} reference data lagged 15 min")
    else:
        plt.xlabel(f"PC{pc} live data")
        plt.ylabel(f"PC{pc} fixed data")
    plt.title(f"PC{pc}")

    # Format axes
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(axmin, axmax)
    plt.ylim(axmin, axmax)
    plt.tight_layout()

    # Save figure
    filename = f"paired_features_pc{pc}"
    if lagged_live_validation:
        filename += "_lagged_live_validation"
    save_plot_to_path(plt.gcf(), save_path / f"{filename}.png", dpi=300)

    plt.close()
