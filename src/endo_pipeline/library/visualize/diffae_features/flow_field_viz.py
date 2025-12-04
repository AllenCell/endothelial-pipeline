from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import get_dataset_descriptions
from endo_pipeline.library.analyze.dynamics_utils import data_driven_flow_field
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.visualize.diffae_features import feature_viz
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES, ColumnName


def set_slice_plot_bounds_and_labels(
    axs: np.ndarray[plt.Axes, Any],
    bounds: list[np.ndarray] | list[tuple[float, float]],
    x_label: str = "PC1",
    y_label: str = "PC2",
) -> np.ndarray[plt.Axes, Any]:
    """
    Set the axis limits and labels for the plots
    of 2D slices of the 3D flow field.
    """
    xmin, xmax = bounds[0][0], bounds[0][1]

    for i, ax in enumerate(axs):
        qmin, qmax = bounds[i + 1][0], bounds[i + 1][1]
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label if i == 0 else "PC3", fontsize=18)
        ax.set_ylim(qmin, qmax)
        ax.set_aspect("equal")
        # set number of x ticks = number of y ticks = 5
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        # set aspect
        ax.set_aspect("auto", adjustable="box")
    return axs


def get_slice_indexes(
    sliced_variable_grid: np.ndarray, sliced_variable_val: float
) -> tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...]:
    """
    Get the slice indexes of the grid that are closest to the prescribed value.
    This is used to slice the grid in 2D for plotting.

    Inputs:
    - sliced_variable_grid: np.ndarray
        The grid of the variable to be sliced.
    - sliced_variable_val: float
        The value of the variable to be sliced.

    Outputs:
    - slice_indexes: np.ndarray
        The indexes of the grid that are closest to
        sliced_variable_val.
    """
    # get slice closest to the prescribed value
    # first, get the absolute distance to the prescribed value
    dist_to_point = np.abs(sliced_variable_grid - sliced_variable_val)
    # get indexes of points where this distance is minimized
    slice_indexes_ = np.where(dist_to_point.ravel() == dist_to_point.min())[0]
    # unravel the grid to get the indices of the points in the grid
    # that are within the z-range of interest
    slice_indexes = np.unravel_index(slice_indexes_, sliced_variable_grid.shape)
    return slice_indexes


def plot_one_slice_quiver(
    velocities: tuple,
    grid: tuple,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ax: plt.Axes,
    color: str | np.ndarray = "dimgrey",
    norm: bool = True,
    ds: int = 3,
    scale: int | float = 30,
) -> plt.Axes:
    """
    Plot one slice of the flow field (quiver plot)
    for a given slice of the grid.
    """

    # slice the grid to get the points in the slice
    # and reshape to 2d array
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    x1_grid = grid[0][slice_indexes].reshape(my_shape)
    x2_grid = grid[1][slice_indexes].reshape(my_shape)
    dx1 = velocities[0][slice_indexes].reshape(my_shape)
    dx2 = velocities[1][slice_indexes].reshape(my_shape)

    # flatten down to 2D depending on which axis has shape == 1
    which_idx = np.where(np.array(my_shape) == 1)[0][0]
    # get xi_grid[... 0 ...] where 0 is taken from the axis with shape == 1
    # and same for dx1 and dx2
    x1_grid = np.take(x1_grid, 0, axis=which_idx)
    x2_grid = np.take(x2_grid, 0, axis=which_idx)
    dx1 = np.take(dx1, 0, axis=which_idx)
    dx2 = np.take(dx2, 0, axis=which_idx)

    if norm:  # norm in 2D
        dx1_ = dx1 / np.sqrt(dx1**2 + dx2**2)
        dx2_ = dx2 / np.sqrt(dx1**2 + dx2**2)
    else:
        dx1_ = dx1.copy()
        dx2_ = dx2.copy()

    # downsample the grid: every 5th point
    x1_grid_ = x1_grid[::ds, ::ds]
    x2_grid_ = x2_grid[::ds, ::ds]
    dx1_ = dx1_[::ds, ::ds]
    dx2_ = dx2_[::ds, ::ds]

    # transpose the grid and velocities for quiver plot
    # (meshgrid generated via indexing ij)
    ax.quiver(x1_grid_.T, x2_grid_.T, dx1_.T, dx2_.T, color=color, scale=scale)

    return ax


def plot_quiver_slices(
    flow_field_dict: dict,
    slice_indexes: tuple[
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ],
    color: str = "dimgrey",
    norm: bool = True,
    fig_ax: tuple | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Plot quiver plots of the 3D flow field
    for the specified 2D slices.
    """
    # get flow field
    v1, v2, v3 = flow_field_dict["vectors"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # plot quiver plots for the specified slices
    if fig_ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax = fig_ax
    ax[0] = plot_one_slice_quiver(
        (v1, v2), (xgrid, ygrid), slice_indexes[0], ax=ax[0], color=color, norm=norm
    )
    ax[1] = plot_one_slice_quiver(
        (v1, v3), (xgrid, zgrid), slice_indexes[1], ax=ax[1], color=color, norm=norm
    )

    return fig, ax


def plot_one_slice_streamplot(
    velocities: tuple,
    grid: tuple,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ax: plt.Axes,
) -> plt.Axes:
    """
    Plot one slice of the flow field (streamplot)
    for a given slice of the grid.
    """
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

    # transpose the grid and velocities for streamplot
    # (meshgrid generated via indexing ij)
    ax.streamplot(x1_grid.T, x2_grid.T, dx1.T, dx2.T, color="black", linewidth=1, density=2)
    return ax


def plot_streamplot_slices(
    flow_field_dict: dict,
    slice_indexes: tuple[
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
        tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ],
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Plot streamplot of the 3D flow field
    for the specified 2D slices.
    """
    # get flow field
    v1, v2, v3 = flow_field_dict["vectors"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # plot streamplot for the specified slices
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0] = plot_one_slice_streamplot((v1, v2), (xgrid, ygrid), slice_indexes[0], ax=ax[0])
    ax[1] = plot_one_slice_streamplot((v1, v3), (xgrid, zgrid), slice_indexes[1], ax=ax[1])

    return fig, ax


def plot_flow_field_stack(
    flow_field_dict: dict,
    plot_axes_indicies: tuple[int, int],
    slice_axis_index: int,
    plot_bounds: list[np.ndarray],
    slice_steps: np.ndarray,
    dataset_name: str,
    fig_savedir: Path | None,
) -> None:
    """
    Plot flow field PC{i} vs PC{j} over a stack of slices in the 3rd variable.

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    plot_axes_indicies
        Tuple (i,j) of indices specifying which principal components to plot.
    slice_axis_index
        Index of the principal component to slice over.
    plot_bounds
        List of arrays specifying the plot bounds for the x and y axes of the 2D plots.
    slice_steps
        List of arrays specifying the slice steps for the slicing axis.
    dataset_name
        Name of the dataset for saving the figure.
    fig_savedir
        Directory to save the figures.
    color
        Color of the quiver arrows.
    norm
        Whether to normalize the velocity vectors.
    """
    # unpack plot axes
    i, j = plot_axes_indicies

    # get flow field
    v_i = flow_field_dict["vectors"][i]
    v_j = flow_field_dict["vectors"][j]
    v_k = flow_field_dict["vectors"][slice_axis_index]

    # color by magnitude of the flow field
    vector_magnitude = np.sqrt(v_i**2 + v_j**2 + v_k**2)
    # set to zero where magnitude is NaN
    vector_magnitude = np.nan_to_num(vector_magnitude, nan=0.0)
    colormap = get_cmap("inferno")
    norm_colors = Normalize().autoscale(vector_magnitude)
    color = colormap(norm_colors(vector_magnitude))

    # get grid and grid spacing
    x_i_grid = flow_field_dict["grid"][i]
    x_j_grid = flow_field_dict["grid"][j]
    x_k_grid = flow_field_dict["grid"][slice_axis_index]

    for n, slice_value in enumerate(slice_steps.tolist()):
        # set up figure
        fig, ax = plt.subplots(figsize=(7, 5))

        # get slice indexes for the current slice value
        x_k_valids = get_slice_indexes(x_k_grid, slice_value)

        # plot quiver plots for the specified slice
        ax = plot_one_slice_quiver(
            (v_i, v_j),
            (x_i_grid, x_j_grid),
            x_k_valids,
            ax=ax,
            color=color,
        )
        # set the axis limits and labels
        ax = set_slice_plot_bounds_and_labels(
            np.array([ax]),
            plot_bounds,
            x_label=f"PC{i+1}",
            y_label=f"PC{j+1}",
        )[0]
        ax.set_title(f"PC{slice_axis_index+1} = {slice_value:.4f}")
        save_plot_to_path(
            fig,
            fig_savedir,
            f"flow_field{dataset_name}_pc{slice_axis_index+1}_stack_{n}",
        )


def plot_flow_field_slices(
    flow_field_dict: dict,
    df_cond: pd.DataFrame | None,
    plot_bounds: list[np.ndarray],
    fig_savedir: Path | None,
    pc_vals: tuple[Any, Any] | None = None,
    color: str = "black",
    norm: bool = True,
) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """
    Plot 2D slices of the 3D flow field
    for the specified 2D slices.

    Plots both quiver plots and streamplots.

    Inputs:
    - flow_field_dict: dict
        Dictionary containing the flow field data.
        Has keys:
        - "vectors": tuple of 3D arrays (v1,v2,v3)
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)
    - df_cond: pd.DataFrame
        DataFrame containing the data to be plotted.
        If None, no data is plotted.
    - fig_savedir: Path
        Directory to save the figures.
        If None, no figures are saved.
    - pc_vals: tuple of floats
        Values of the 2nd and 3rd principal components
        (2nd and 3rd variables) to slice the data.
        If None, the mean of the data at the last
        time point is used.

    """
    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # for plotting in 2D, we need to slice
    # the data in PC3 and PC2 to get PC1 v. PC2
    # and PC1 v. PC3 plots, respectively

    # if not specified, use mean of data at last time point
    # if data are not provided, use pc2 = pc3 = 0
    if pc_vals is None:
        if df_cond is None:
            pc3_val = 0.0
            pc2_val = 0.0
        else:
            # get mean at all time points over crops
            mean_over_crops_ = df_cond.groupby(ColumnName.TIMEPOINT).mean(numeric_only=True)
            # get last time point
            mean_over_crops = mean_over_crops_.iloc[-1]
            pc3_val = mean_over_crops[DIFFAE_PC_COLUMN_NAMES[2]].mean()
            pc2_val = mean_over_crops[DIFFAE_PC_COLUMN_NAMES[1]].mean()
    # if specified, unpack
    else:
        pc3_val = pc_vals[0]
        pc2_val = pc_vals[1]

    # get z-slice closest to PC3 = pc3_val
    zvalids = get_slice_indexes(zgrid, pc3_val)
    # get y-slice closest to PC2 = pc2_val
    yvalids = get_slice_indexes(ygrid, pc2_val)

    # plot quiver plots of these PC2 and PC3 slices
    # overlaid on scatter plot of data
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    if df_cond is not None:
        # get the color for the scatter plot
        dataset_name = sequence_to_scalar(df_cond[ColumnName.DATASET])
        scatter_color = feature_viz.get_dataset_color(dataset_name)
        # plot scatter of data overlaid on quiver plot
        ax[0].scatter(df_cond.pc_1, df_cond.pc_2, s=0.25, color=scatter_color, alpha=0.15)
        ax[1].scatter(df_cond.pc_1, df_cond.pc_3, s=0.25, color=scatter_color, alpha=0.15)
    fig, ax = plot_quiver_slices(
        flow_field_dict, (zvalids, yvalids), color=color, norm=norm, fig_ax=(fig, ax)
    )

    # set the axis limits and labels
    ax = set_slice_plot_bounds_and_labels(ax, plot_bounds)
    # set titles with slice values
    ax[0].set_title(f"PC3 = {pc3_val:.2f}")
    ax[1].set_title(f"PC2 = {pc2_val:.2f}")
    plt.tight_layout()
    plt.show()

    # plot streamplot of these PC2 and PC3 slices
    fig_, ax_ = plot_streamplot_slices(flow_field_dict, (zvalids, yvalids))
    # set the axis limits and labels
    ax_ = set_slice_plot_bounds_and_labels(ax_, plot_bounds)
    # set titles with slice values
    ax_[0].set_title(f"PC3 = {pc3_val:.2f}")
    ax_[1].set_title(f"PC2 = {pc2_val:.2f}")
    plt.tight_layout()
    plt.show()

    if fig_savedir is not None:
        # if data provided, get
        # get the condition name
        # for saving the plot
        if df_cond is not None:
            name = df_cond[ColumnName.DATASET].unique()[0]
            condition = get_dataset_descriptions([name], simple=True)[name]
        else:
            condition = "from_data"
        fig.suptitle(condition, fontsize=16)
        save_plot_to_path(fig, fig_savedir, f"flow_field_{name}")  # save the figure
        fig_.suptitle(condition, fontsize=16)
        save_plot_to_path(fig_, fig_savedir, f"flow_field_streamplot_{name}")  # save the figure

    return fig, ax


def plot_stable_fixed_points_together(
    list_of_datasets: list[str], fig_savedir: Path, output_savedir: Path
) -> None:
    """
    Generate plot of fixed points of the low,
    high, and intermediate (12dyn) shear stress conditions
    on the same plot.
    """

    traj_dict = np.load(output_savedir / "traj_dict.npy", allow_pickle=True).item()

    conditions = get_dataset_descriptions(list_of_datasets, simple=True)

    # initialize plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # get bounds of the grid - load one of the flow field objects
    # saved in main function
    flow_field_dict = np.load(
        output_savedir / f"flow_field_dict_{list_of_datasets[0]}.npy", allow_pickle=True
    ).item()
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

    # loop through the datasets and plot the fixed points
    for name in list_of_datasets:
        condition = conditions[name]
        coords = traj_dict[condition]
        scatter_color = feature_viz.get_dataset_color(name)

        if type(coords) is np.ndarray:  # single attractor
            # get last point of trajectory
            fp = coords[-1, :]
            # plot fixed point
            # PC1 vs PC2, PC1 vs PC3
            ax[0].scatter(fp[0], fp[1], s=100, color=scatter_color, edgecolor="black")
            ax[1].scatter(fp[0], fp[2], s=100, color=scatter_color, edgecolor="black")
        elif type(coords) is list:  # multiple attractors
            for coord in coords:
                # get last point of trajectory
                fp = coord[-1, :]
                # plot fixed point
                # PC1 vs PC2, PC1 vs PC3
                ax[0].scatter(fp[0], fp[1], s=100, color=scatter_color, edgecolor="black")
                ax[1].scatter(fp[0], fp[2], s=100, color=scatter_color, edgecolor="black")

    # set the axis limits and labels
    ax = set_slice_plot_bounds_and_labels(ax, bounds_)
    # set titles with slice values
    plt.tight_layout()
    plt.show()

    # save the figure
    save_plot_to_path(fig, fig_savedir, "fixed_points_plot")


def flow_field_viz_main(
    flow_field_dict: dict,
    df_cond: pd.DataFrame,
    traj: np.ndarray,
    plot_bounds: list[np.ndarray],
    fig_savedir: Path,
) -> None:
    """
    Plot 2D summary plots for the computed 3D flow fields.

    **Input dictionary flow_field_dict:**

    The method input ``flow_field_dict`` should have the following key/value pairs:
        - "vectors": tuple of 3D arrays (v1,v2,v3)
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid)

    Parameters
    ----------
    flow_field_dict
        Dictionary containing the flow field data.
    df_cond
        DataFrame containing the data to be plotted.
    traj
        Trajectory of the system in the flow field.
    plot_bounds
        List of arrays specifying the plot bounds for each principal component.
    fig_savedir
        Directory to save the figures.
    """
    # dataset flow condition for saving the figures
    name = df_cond[ColumnName.DATASET].unique()[0]
    condition = get_dataset_descriptions([name], simple=True, include_shear_stress=True)[name]

    # plot 2D slices at PC2 and PC3 values given by
    # the last point of the input trajectory
    pc_vals = (traj[-1, 2], traj[-1, 1])

    # baseline visualization: plot flow field slices
    # (quiver plot with scatter of data, streamplot)
    plot_flow_field_slices(flow_field_dict, df_cond, plot_bounds, fig_savedir, pc_vals=pc_vals)

    ###### additional plots for visualization of flow field #######
    # 1) plot stacks of flow field slices
    # 2) last point of trajectory over flow field
    # 3) entire trajectory over flow field
    # 4) trajectory with equally spaced interpolated points

    # 1) plot stacks of flow field slices
    # get PC1, PC2, and PC3 slices from meshgrid (ijk indexing)
    pc_slices = [
        flow_field_dict["grid"][0][:, 0, 0][
            :: len(flow_field_dict["grid"][0][:, 0, 0]) // 10
        ],  # PC1
        flow_field_dict["grid"][1][0, :, 0][
            :: len(flow_field_dict["grid"][1][0, :, 0]) // 10
        ],  # PC2
        flow_field_dict["grid"][2][0, 0, :][
            :: len(flow_field_dict["grid"][2][0, 0, :]) // 10
        ],  # PC3
    ]
    plot_axes_indicies = [
        (0, 1),  # PC1 vs PC2 over PC3 slices
        (0, 2),  # PC1 vs PC3 over PC2 slices
        (1, 2),  # PC2 vs PC3 over PC1 slices
    ]
    slice_axis_indices = [2, 1, 0]  # PC3, PC2, PC1

    for plot_axes, slice_axis in zip(plot_axes_indicies, slice_axis_indices, strict=True):
        slice_steps = pc_slices[slice_axis]
        plot_bounds_2d = [
            plot_bounds[plot_axes[0]],
            plot_bounds[plot_axes[1]],
        ]
        plot_flow_field_stack(
            flow_field_dict,
            plot_axes_indicies=(0, 1),
            slice_axis_index=2,
            plot_bounds=plot_bounds_2d,
            slice_steps=slice_steps,
            fig_savedir=fig_savedir,
            dataset_name=name,
        )

    # get z-slice and y-slice closest to PC2 and PC3 values
    zvalids = get_slice_indexes(
        flow_field_dict["grid"][-1], pc_vals[0]
    )  # get z-slice closest to PC3 = PC3_val
    yvalids = get_slice_indexes(
        flow_field_dict["grid"][-2], pc_vals[1]
    )  # get y-slice closest to PC2 = PC2_val

    # 2) plot last point of trajectory over flow field
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # get the color for the scatter plot
    scatter_color = feature_viz.get_dataset_color(name)
    # plot scatter of data overlaid on quiver plot
    ax[0].scatter(df_cond.pc_1, df_cond.pc_2, s=0.25, color=scatter_color, alpha=0.05)
    ax[1].scatter(df_cond.pc_1, df_cond.pc_3, s=0.25, color=scatter_color, alpha=0.05)
    fig, ax = plot_quiver_slices(flow_field_dict, (zvalids, yvalids), fig_ax=(fig, ax))
    fig.suptitle(condition, fontsize=16)

    # plot last point of trajectory
    for j, ax_ in enumerate(ax):  # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[-1, 0], traj[-1, j + 1], s=100, color="black")

    # plot second stable point
    ax = set_slice_plot_bounds_and_labels(ax, plot_bounds)
    # set titles with slice values
    ax[0].set_title(f"PC3 = {pc_vals[0]:.2f}")
    ax[1].set_title(f"PC2 = {pc_vals[1]:.2f}")
    plt.tight_layout()
    plt.show()
    # save the figure
    save_plot_to_path(fig, fig_savedir, f"flow_field_{name}_fp")

    # 3) plot entire trajectory over flow field
    # PC1 v s PC2, PC1 vs PC3
    for j, ax_ in enumerate(ax):
        ax_.plot(traj[:, 0], traj[:, j + 1], linewidth=2.5, color="navy")
    plt.tight_layout()
    plt.show()
    # save the figure
    save_plot_to_path(fig, fig_savedir, f"flow_field_{name}_traj")

    # 4) trajectory with equally spaced interpolated points
    interpolated_points = data_driven_flow_field.interpolate_on_curve(traj)
    for j, ax_ in enumerate(ax):
        ax_.scatter(
            interpolated_points[:, 0],
            interpolated_points[:, j + 1],
            s=10,
            color="red",
        )
    plt.tight_layout()
    plt.show()
    # save the figure
    save_plot_to_path(fig, fig_savedir, f"flow_field_{name}_traj_interpolated")
