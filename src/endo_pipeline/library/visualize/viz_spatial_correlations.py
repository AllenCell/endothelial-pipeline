import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

logger = logging.getLogger(__name__)


def plot_points_with_angles(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    ax: Axes | None = None,
    size_x: float | None = None,
    size_y: float | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Visualize points with directional arrows showing their orientations.

    Parameters
    ----------
    x
        X-coordinates of points to plot
    y
        Y-coordinates of points to plot
    theta
        Orientation angles in radians
    ax
        Axes for plotting. If None, creates new figure with 6x6 inch size at 300 dpi
    size_x
        Width of the plotting area in the x-direction
    size_y
        Height of the plotting area in the y-direction

    Returns
    -------
    :
        Figure and axes containing the quiver plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    else:
        fig = ax.figure

    common_kwargs = {
        "linewidth": 0.4,
        "alpha": 0.7,
        "color": "tab:blue",
        "scale": 0.02,
        "headwidth": 1,
        "headlength": 1,
        "headaxislength": 1,
        "angles": "xy",
        "scale_units": "xy",
    }

    ax.quiver(
        x,
        y,
        np.cos(theta),
        np.sin(theta),
        **common_kwargs,
    )
    ax.quiver(
        x,
        y,
        np.cos(theta + np.pi),
        np.sin(theta + np.pi),
        **common_kwargs,
    )
    if size_x is not None:
        ax.set_xlim(0, size_x)
    if size_y is not None:
        ax.set_ylim(0, size_y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    return fig, ax


def plot_orientational_correlation(
    r: np.ndarray,
    C_r: np.ndarray,
    xi_orient: float | None = None,
    fitted_c_r: np.ndarray | None = None,
    ax: Axes | None = None,
    plot_scatter: bool = True,
    title: str | None = None,
    **kwargs,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot orientational correlation function versus distance.

    Parameters
    ----------
    r
        Distance bin centers
    C_r
        Correlation values at each distance
    xi_orient
        Correlation length to mark with vertical line
    fitted_c_r
        Exponential decay fit to overlay on plot
    ax
        Axes for plotting. If None, creates new figure with 6x6 inch size at 300 dpi
    plot_scatter
        If True, plot data as scatter points. Default is True
    title
        Optional plot title to display
    **kwargs
        Additional keyword arguments passed to the fitted curve plot

    Returns
    -------
    :
        Figure and axes containing the correlation plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    else:
        fig = ax.figure

    if plot_scatter:
        ax.scatter(r, C_r, s=10)

    if xi_orient is not None:
        ax.axvline(x=xi_orient, color="red", linestyle="--", label=f"ξ = {xi_orient:.2f}")

    if fitted_c_r is not None:
        ax.plot(r, fitted_c_r, **kwargs)

    ax.set_xlabel("Distance r")
    ax.set_ylabel("C(r)")
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(-0.1, 1.05)

    return fig, ax


def plot_topological_defects(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    defect_positions: np.ndarray,
    defect_numbers: np.ndarray,
    ax: Axes | None = None,
    size_x: float | None = None,
    size_y: float | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Visualize points with directional arrows showing their orientations and
    overlay topological defects.

    Parameters
    ----------
    x
        X-coordinates of points to plot
    y
        Y-coordinates of points to plot
    theta
        Orientation angles in radians
    defect_positions
        Array of shape (N, 2) containing X and Y coordinates of detected topological defects
    defect_numbers
        Array of length N containing the topological charge (+/- 0.5) of each defect
    ax
        Axes for plotting. If None, creates new figure with 6x6 inch size at 300 dpi
    size_x
        Width of the plotting area in the x-direction
    size_y
        Height of the plotting area in the y-direction

    Returns
    -------
    :
        Figure and axes containing the quiver plot with defects marked
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    else:
        fig = ax.figure

    fig, ax = plot_points_with_angles(x, y, theta, ax=ax, size_x=size_x, size_y=size_y)

    positive_defects = defect_numbers > 0
    negative_defects = defect_numbers < 0

    for positions, kwargs in [
        (defect_positions[positive_defects], {"color": "red", "marker": "x"}),
        (defect_positions[negative_defects], {"color": "green", "marker": "o"}),
    ]:
        if positions.size > 0:
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=50,
                marker=kwargs["marker"],
                facecolors=kwargs["color"],
                edgecolors=kwargs["color"],
                label=f"{'+' if kwargs['color']=='red' else '-'}1/2 Defects",
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return fig, ax
