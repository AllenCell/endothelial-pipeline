from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# set default plot parameters
global plt_params

plt_params = {
    "legend.fontsize": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 20,
    "figure.max_open_warning": 0,  # suppress warning for too many open figures
}
plt.rcParams.update(plt_params)


def init_plot(figsize: tuple = (7, 6)) -> tuple[plt.Figure, plt.Axes]:
    """
    Initialize a matplotlib figure and axes with default settings.

    Input:
    - figsize: tuple (default=(7,6)), size of the figure

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def init_subplots(
    nrows: int = 1, ncols: int = 2, figsize: tuple = (14, 6)
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Initialize matplotlib figure with multiple subplots.

    Input:
    - nrows: int (default=1) number of rows of subplots
    - ncols: int (default=2) number of columns of subplots
    - figsize: tuple (default=(14,6)) size of the figure

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax


def save_plot(fig: plt.Figure, filename: str, format: str = ".png", dpi: int = 450) -> None:
    """
    Save a matplotlib figure to a file with the specified filename.

    Input:
    - fig: plt.Figure, the figure to save
    - filename: str, the filepath to save the figure
        (includes file name, but NOT the file extension)
    - format: str (default='.png'), the file format to save the figure
    - dpi: int (default=450), the resolution of the figure in
        dots per inch (dpi) if format is .png

    Output:
    - None, saves the figure to the specified file
    """
    if format == ".png":
        fig.savefig(filename + format, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(filename + format, bbox_inches="tight")
