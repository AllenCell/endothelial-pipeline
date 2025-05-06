from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellsmap.analyses.utils.numerics import data_driven_3D_flow_field as ddff
from cellsmap.analyses.utils.viz import viz_base as vb


def set_slice_plot_bounds_and_labels(
    axs: Tuple[plt.Axes], bounds: list[Tuple[float]]
) -> plt.Axes:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]

    for ax, (qmin, qmax) in zip(axs, [(ymin, ymax), (zmin, zmax)]):
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("PC1", fontsize=18)
        ax.set_ylabel("PC2" if ax == axs[0] else "PC3", fontsize=18)
        ax.set_ylim(qmin, qmax)
        ax.set_aspect("equal")
        # set number of x ticks = number of y ticks = 5
        ax.set_xticks(np.linspace(xmin + 0.05, xmax - 0.05, 5))
        ax.set_yticks(np.linspace(qmin + 0.05, qmax - 0.05, 5))
        # set aspect
        ax.set_aspect("auto", adjustable="box")
    return axs


def get_slice_indexes(
    sliced_variable_grid: np.ndarray, sliced_variable_val: float
) -> np.ndarray:

    # get slice closest to the prescribed value
    # first, get the absolute distance to the prescribed value
    dist_to_point = np.abs(sliced_variable_grid - sliced_variable_val)
    # get indexes of points where this distance is minimized
    slice_indexes = np.where(dist_to_point.ravel() == dist_to_point.min())[0]
    # unravel the grid to get the indices of the points in the grid
    # that are within the z-range of interest
    slice_indexes = np.unravel_index(slice_indexes, sliced_variable_grid.shape)
    return slice_indexes


def plot_one_slice_quiver(
    velocities: Tuple,
    grid: Tuple,
    slice_indexes: np.ndarray,
    color: str = "mediumturquoise",
    norm: bool = True,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one slice of the flow field (quiver plot) for a given slice of the grid.
    """
    if ax is None:
        _, ax = vb.init_subplots()

    # slice the grid to get the points in the slice
    x1_grid = grid[0][slice_indexes]
    x2_grid = grid[1][slice_indexes]
    dx1 = velocities[0][slice_indexes]
    dx2 = velocities[1][slice_indexes]
    if norm:  # norm in 2D
        dx1_ = dx1 / np.sqrt(dx1**2 + dx2**2)
        dx2_ = dx2 / np.sqrt(dx1**2 + dx2**2)
    else:
        dx1_ = dx1.copy()
        dx2_ = dx2.copy()

    ax.quiver(x1_grid, x2_grid, dx1_, dx2_, color=color, scale=50)

    return ax


def plot_quiver_slices(
    flow_field_dict: dict,
    slice_indexes: Tuple[np.ndarray],
    color: str = "mediumturquoise",
    norm: bool = True,
    fig_ax: Tuple | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # get flow field
    dU, dV, dQ = flow_field_dict["vectors"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # plot quiver plots for the specified slices
    if fig_ax is None:
        fig, ax = vb.init_subplots(figsize=(14, 5))
    else:
        fig, ax = fig_ax
    ax[0] = plot_one_slice_quiver(
        (dU, dV), (xgrid, ygrid), slice_indexes[0], color=color, ax=ax[0], norm=norm
    )
    ax[1] = plot_one_slice_quiver(
        (dU, dQ), (xgrid, zgrid), slice_indexes[1], color=color, ax=ax[1], norm=norm
    )

    return fig, ax


def plot_one_slice_streamplot(
    velocities: Tuple,
    grid: Tuple,
    slice_indexes: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = vb.init_subplots()

    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]
    # slice the grid to get the points in the slice, reshape for plotting
    x1_grid = grid[0][slice_indexes].reshape(my_shape)
    x2_grid = grid[1][slice_indexes].reshape(my_shape)
    # flatten down to 2D depending on which axis has shape == 1
    which_idx = np.where(np.array(my_shape) == 1)[0][0]
    # get xi_grid[... 0 ...] where 0 is taken from the axis with shape == 1
    x1_grid = np.take(x1_grid, 0, axis=which_idx)
    x2_grid = np.take(x2_grid, 0, axis=which_idx)

    # get the velocities at these points (again, the correct slices)
    dx1 = velocities[0][slice_indexes].reshape(x1_grid.shape)
    dx2 = velocities[1][slice_indexes].reshape(x2_grid.shape)

    # transpose the grid and velocities for streamplot (meshgrid generated via indexing ij)
    ax.streamplot(
        x1_grid.T, x2_grid.T, dx1.T, dx2.T, color="black", linewidth=1, density=2
    )
    return ax


def plot_streamplot_slices(
    flow_field_dict: dict, slice_indexes: Tuple[np.ndarray]
) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
    # get flow field
    dU, dV, dQ = flow_field_dict["vectors"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # plot streamplot for the specified slices
    fig, ax = vb.init_subplots(figsize=(14, 5))
    ax[0] = plot_one_slice_streamplot(
        (dU, dV), (xgrid, ygrid), slice_indexes[0], ax=ax[0]
    )
    ax[1] = plot_one_slice_streamplot(
        (dU, dQ), (xgrid, zgrid), slice_indexes[1], ax=ax[1]
    )

    return fig, ax


def plot_flow_field_slices(
    flow_field_dict: dict,
    df_cond: pd.DataFrame | None,
    fig_savedir: str | None,
    PC_vals: Tuple[float] | None = None,
    color: str = "mediumturquoise",
    norm: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # get bounds of the grid
    xmin, xmax = xgrid[0, 0, 0], xgrid[-1, 0, 0]
    ymin, ymax = ygrid[0, 0, 0], ygrid[0, -1, 0]
    zmin, zmax = zgrid[0, 0, 0], zgrid[0, 0, -1]
    bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

    # for plotting in 2D, we need to slice the data in PC3 and PC2
    # to get PC1 v. PC2 and PC1 v. PC3 plots, respectively

    if PC_vals is None:  # if not specified, use mean of data at last time point
        # get mean at all time points over crops
        mean_over_crops = df_cond.groupby("T").mean(numeric_only=True)
        mean_over_crops = mean_over_crops.iloc[-1]  # get last time point
        PC2_val = mean_over_crops["PC2"].mean()
    else:  # if specified, use these values
        PC3_val = PC_vals[0]
        PC2_val = PC_vals[1]

    zvalids = get_slice_indexes(zgrid, PC3_val)  # get z-slice closest to PC3 = PC3_val
    yvalids = get_slice_indexes(ygrid, PC2_val)  # get y-slice closest to PC2 = PC2_val

    # plot quiver plots of these PC2 and PC3 slices overlaid on scatter plot of data
    fig, ax = vb.init_subplots(figsize=(14, 5))
    if df_cond is not None:
        # plot scatter of data overlaid on quiver plot
        ax[0].scatter(df_cond.PC1, df_cond.PC2, s=0.25, color="black", alpha=0.1)
        ax[1].scatter(df_cond.PC1, df_cond.PC3, s=0.25, color="black", alpha=0.1)
    fig, ax = plot_quiver_slices(
        flow_field_dict, (zvalids, yvalids), color=color, norm=norm, fig_ax=(fig, ax)
    )

    # set the axis limits and labels
    ax = set_slice_plot_bounds_and_labels(ax, bounds)
    ax[0].set_title(f"PC3 = {PC3_val:.2f}")  # title for PC3 slice
    ax[1].set_title(f"PC2 = {PC2_val:.2f}")  # title for PC2 slice
    plt.tight_layout()
    plt.show()

    # plot streamplot of these PC2 and PC3 slices
    fig_, ax_ = plot_streamplot_slices(flow_field_dict, (zvalids, yvalids))
    # set the axis limits and labels
    ax_ = set_slice_plot_bounds_and_labels(ax_, bounds)
    ax_[0].set_title(f"PC3 = {PC3_val:.2f}")  # title for PC3 slice
    ax_[1].set_title(f"PC2 = {PC2_val:.2f}")  # title for PC2 slice
    plt.tight_layout()
    plt.show()

    if fig_savedir is not None:
        condition = df_cond.description.unique()[
            0
        ]  # get the condition name for saving the plot
        vb.save_plot(
            fig, filename=fig_savedir + f"flow_field_{condition}", dpi=300
        )  # save the figure
        vb.save_plot(
            fig_, filename=fig_savedir + f"flow_field_streamplot_{condition}", dpi=300
        )  # save the figure

    return fig, ax


def flow_field_viz_main(
    flow_field_dict: dict, df_cond: pd.DataFrame, traj: np.ndarray, fig_savedir: str
) -> None:

    # dataset flow condition for saving the figures
    condition = df_cond["description"].values[0]

    # plot 2D slices at PC2 and PC3 values given by the last point of the trajectory
    PC_vals = (traj[-1, 2], traj[-1, 1])  # get last point of trajectory

    # baseline visualization: plot flow field slices (quiver plot with scatter of data, streamplot)
    plot_flow_field_slices(flow_field_dict, df_cond, fig_savedir, PC_vals=PC_vals)

    ###### additional plots for visualization of flow field #######
    # 1) last point of trajectory over flow field
    # 2) entire trajectory over flow field
    # 3) trajectory with equally spaced interpolated points

    # get z-slice and y-slice closest to PC2 and PC3 values
    zvalids = get_slice_indexes(
        flow_field_dict["grid"][-1], PC_vals[0]
    )  # get z-slice closest to PC3 = PC3_val
    yvalids = get_slice_indexes(
        flow_field_dict["grid"][-2], PC_vals[1]
    )  # get y-slice closest to PC2 = PC2_val

    # get bounds of the grid
    xmin, xmax = (
        flow_field_dict["grid"][0][0, 0, 0],
        flow_field_dict["grid"][0][-1, 0, 0],
    )
    ymin, ymax = (
        flow_field_dict["grid"][1][0, 0, 0],
        flow_field_dict["grid"][1][0, -1, 0],
    )
    zmin, zmax = (
        flow_field_dict["grid"][2][0, 0, 0],
        flow_field_dict["grid"][2][0, 0, -1],
    )
    bounds_ = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

    # 1) plot last point of trajectory over flow field
    fig, ax = plot_quiver_slices(flow_field_dict, (zvalids, yvalids))
    # plot last point of trajectory
    for j, ax_ in enumerate(ax):  # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[-1, 0], traj[-1, j + 1], s=200, color="black")
    ax = set_slice_plot_bounds_and_labels(ax, bounds_)
    ax[0].set_title(f"PC3 = {PC_vals[0]:.2f}")  # title for PC3 slice
    ax[1].set_title(f"PC2 = {PC_vals[1]:.2f}")  # title for PC2 slice
    plt.tight_layout()
    plt.show()

    vb.save_plot(
        fig, fig_savedir + f"flow_field_{condition}_fp", dpi=300
    )  # save the figure

    # 2) plot entire trajectory over flow field
    fig, ax = plot_quiver_slices(flow_field_dict, (zvalids, yvalids))
    for j, ax_ in enumerate(ax):  # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[:, 0], traj[:, j + 1], s=30, color="navy")
    ax = set_slice_plot_bounds_and_labels(ax, bounds_)
    plt.tight_layout()
    plt.show()
    vb.save_plot(
        fig, fig_savedir + f"flow_field_{condition}_traj", dpi=300
    )  # save the figure

    # 3) trajectory with equally spaced interpolated points
    interpolated_points = ddff.interpolate_on_curve(traj)
    for j, ax_ in enumerate(ax):
        ax_.scatter(
            interpolated_points[:, 0],
            interpolated_points[:, j + 1],
            s=10,
            color="springgreen",
        )
    plt.tight_layout()
    plt.show()
    vb.save_plot(
        fig, fig_savedir + f"flow_field_{condition}_traj_interpolated", dpi=300
    )  # save the figure

    return
