from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.typing import ArrayLike


def plot_paired_fixed_live_validation_features(
    save_path: Path,
    pc: int,
    raw_data: tuple[ArrayLike, ArrayLike],
    paired_validation_features: tuple[Any, Any, Any, Any, Any, Ellipse],
    color_list: list[str] = ["#5F9ED1", "#FF800E", "#C85200"],
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
    """

    # Get raw fixed (y) and live (x) PC data and its lower and upper limits
    x, y = raw_data
    min_ = min(x.min(), y.min())
    max_ = max(x.max(), y.max())

    # Get all validation features
    center, height, angle, slope, intercept, ellipse = paired_validation_features

    # Create scatter plot of PC data from two experiments
    plt.clf()
    ax = plt.gca()

    # Plot unity line
    plt.plot([min_, max_], [min_, max_], c="gray", linestyle="--", label="Unity line")

    # Plot raw data
    ax.scatter(x, y, s=0.5, c="black", alpha=0.1)

    # Plot confidence ellipse
    ax.add_patch(ellipse)

    # Plot linear model along major axis of ellipse
    y_model_min = slope * min_ + intercept
    y_model_max = slope * max_ + intercept
    plt.plot(
        [min_, max_],
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
    plt.xlabel(f"PC{pc} live data")
    plt.ylabel(f"PC{pc} fixed data")
    plt.title(f"PC{pc}")

    # Format axes
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.tight_layout()

    # Save figure
    filename = f"paired_features_pc{pc}"
    plt.savefig(save_path / f"{filename}.png", dpi=300)
    print(f"Fig saved to directory {save_path}.")
    plt.close()
