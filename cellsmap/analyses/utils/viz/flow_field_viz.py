import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from cellsmap.analyses.utils.viz import viz_base as vb

def set_slice_plot_bounds_and_labels(axs:Tuple[plt.Axes],bounds:list[Tuple[float]]) -> plt.Axes:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    
    for ax, (qmin, qmax) in zip(axs, [(ymin, ymax), (zmin, zmax)]):
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if ax==axs[0] else "PC3")
        ax.set_ylim(qmin, qmax)
        ax.set_aspect("equal")
    return axs

def get_slice_indexes(sliced_variable_grid:np.ndarray,
                        sliced_variable_val:float) -> np.ndarray:
    
    # get slice closest to the prescribed value
    # first, get the absolute distance to the prescribed value
    dist_to_point = np.abs(sliced_variable_grid - sliced_variable_val)
    # get indexes of points where this distance is minimized
    slice_indexes = np.where(dist_to_point.ravel()==dist_to_point.min())[0]
    # unravel the grid to get the indices of the points in the grid
    # that are within the z-range of interest
    slice_indexes = np.unravel_index(slice_indexes, sliced_variable_grid.shape)
    return slice_indexes

def plot_one_slice_quiver(velocities:Tuple, 
                   grid:Tuple, 
                   slice_indexes:np.ndarray, 
                   ax:plt.Axes|None=None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one slice of the flow field (quiver plot) for a given slice of the grid.
    """
    if ax is None:
        _, ax = vb.init_subplots()

    ax.quiver(grid[0][slice_indexes],grid[1][slice_indexes], 
              velocities[0][slice_indexes],velocities[1][slice_indexes], 
              color="red")
    
    return ax

def plot_one_slice_streamplot(velocities:Tuple,
                              grid:Tuple,
                              slice_indexes:np.ndarray,
                              ax:plt.Axes|None=None) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        _, ax = vb.init_subplots()

    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]
    # slice the grid to get the points in the slice, reshape for plotting
    x1_grid = grid[0][slice_indexes].reshape(my_shape)
    x2_grid = grid[1][slice_indexes].reshape(my_shape)
    # flatten down to 2D depending on which axis has shape == 1
    which_idx = np.where(np.array(my_shape)==1)[0][0]
    # get xi_grid[... 0 ...] where 0 is taken from the axis with shape == 1
    x1_grid = np.take(x1_grid, 0, axis=which_idx)
    x2_grid = np.take(x2_grid, 0, axis=which_idx)

    # get the velocities at these points (again, the correct slices)
    dx1 = velocities[0][slice_indexes].reshape(x1_grid.shape)
    dx2 = velocities[1][slice_indexes].reshape(x2_grid.shape)

    # transpose the grid and velocities for streamplot (meshgrid generated via indexing ij)
    ax.streamplot(x1_grid.T, x2_grid.T,
                    dx1.T, dx2.T,
                    color="black", linewidth=1, density=2)
    return ax

def plot_flow_field_slices(flow_field_dict:dict, 
                           df_cond:pd.DataFrame,
                           fig_savedir:str, 
                           verbose:bool=True,
                           norm:bool=True) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    
    # get flow field
    dU, dV, dQ = flow_field_dict["velocities"]
    # normalize the flow field (for visualization)
    if norm:
        dU = dU/np.sqrt(dU**2 + dV**2 + dQ**2)
        dV = dV/np.sqrt(dU**2 + dV**2 + dQ**2)
        dQ = dQ/np.sqrt(dU**2 + dV**2 + dQ**2)

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]
    grid_spacing = xgrid[1,0,0] - xgrid[0,0,0]

    # get bounds of the grid
    xmin, xmax = xgrid[0,0,0], xgrid[-1,0,0]
    ymin, ymax = ygrid[0,0,0], ygrid[0,-1,0]
    zmin, zmax = zgrid[0,0,0], zgrid[0,0,-1]
    bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

    # get mean at all time points over crops
    mean_over_crops = df_cond.groupby("T").mean(numeric_only=True)
    mean_over_crops = mean_over_crops.loc[mean_over_crops["T"] == df_cond["T"].max()] # get last time point
    # plotting 2D slices of the 3D flow field
    # get z-slice closest to PC3 = PC3_val
    # where PC3_val = mean of PC3 at last time point in the data
    PC3_val = mean_over_crops["PC3"].mean()
    zvalids = get_slice_indexes(zgrid, PC3_val)

    # get y-slice closest to PC2 = PC2_val
    # where PC2_val = mean of PC2 at last time point in the data
    PC2_val = mean_over_crops["PC2"].mean()
    yvalids = get_slice_indexes(ygrid, PC2_val)

    # plot quiver plots of these PC2 and PC3 slices overlaid on scatter plot of data
    fig, (ax1, ax2) = vb.init_subplots()
    ax1.scatter(df_cond.PC1, df_cond.PC2, s=0.25, color="black", alpha=0.1)
    ax1 = plot_one_slice_quiver((dU, dV), (xgrid, ygrid), zvalids, ax=ax1)
    ax1.set_title(f"PC3 = {PC3_val:.2f}")
    ax2.scatter(df_cond.PC1, df_cond.PC3, s=0.25, color="black", alpha=0.1)    
    ax2 = plot_one_slice_quiver((dU, dQ), (xgrid, zgrid), yvalids, ax=ax2)
    ax2.set_title(f"PC2 = {PC2_val:.2f}")
    
    (ax1,ax2) = set_slice_plot_bounds_and_labels((ax1,ax2), bounds)
    
    condition = df_cond.description.unique()[0] # get the condition name for saving the plot
    plt.tight_layout()
    plt.show()
    vb.save_plot(fig, filename=fig_savedir+f"flow_field_pc_{condition}", dpi=300) # save the figure

    # plot streamplot of these PC2 and PC3 slices
    fig, (ax1, ax2) = vb.init_subplots()
    ax1 = plot_one_slice_streamplot((dU, dV), (xgrid, ygrid), zvalids, ax=ax1)
    ax2 = plot_one_slice_streamplot((dU, dQ), (xgrid, zgrid), yvalids, ax=ax2)
    (ax1,ax2) = set_slice_plot_bounds_and_labels((ax1,ax2), bounds)
    ax1.set_title("PC3 = 0")
    ax2.set_title("PC2 = 0")

    return fig, ax1, ax2

def compare_mean_to_traj(data_mean_traj,traj,fig_ax:Tuple[plt.Figure, plt.Axes]|None=None):
    if fig_ax is None:
        fig, ax = vb.init_subplots()
    else:
        fig, ax = fig_ax

    
    ax[0].quiver
    ax[0].scatter(data_mean_traj[:,0], data_mean_traj[:,1], alpha=0.5)
    ax[0].scatter(traj[:,0], traj[:,1], c='k',s=8)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')


    ax[1].scatter(data_mean_traj[:,0], data_mean_traj[:,2], alpha=0.5)
    ax[1].scatter(traj[:,0], traj[:,2], c='k',s=8)
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC3')

    return fig, ax

