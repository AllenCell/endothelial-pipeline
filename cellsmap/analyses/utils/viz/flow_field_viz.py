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

def get_slice_indexes(grid_spacing:float,
                        sliced_variable_name:str,
                        sliced_variable_grid:np.ndarray,
                        sliced_variable_val:float=0.0,
                        verbose:bool=True) -> np.ndarray:
    
    slice_min = sliced_variable_val - 0.8*grid_spacing
    slice_max = sliced_variable_val + 1.2*grid_spacing

    # get indexes of points where the sliced variable is within 20% of grid_spacing of the prescribed value
    slice_indexes = np.where((sliced_variable_grid.ravel()>slice_min)&(sliced_variable_grid.ravel()<slice_max))[0]
    if verbose:
        print(f"Number of points found within ± {(0.2*grid_spacing):.3f} of {sliced_variable_name} = {sliced_variable_val}:")
        print(len(slice_indexes))
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
    
    ax.streamplot(grid[0][slice_indexes],grid[1][slice_indexes],
                    velocities[0][slice_indexes],velocities[1][slice_indexes],
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
        try:
            dU = dU/np.sqrt(dU**2 + dV**2 + dQ**2)
            dV = dV/np.sqrt(dU**2 + dV**2 + dQ**2)
            dQ = dQ/np.sqrt(dU**2 + dV**2 + dQ**2)
        except ZeroDivisionError:
            # set zero magnitude to epsilon (small float)
            zero_mag_mask = np.sqrt(dU**2 + dV**2 + dQ**2) == 0
            epsilon = 1e-10
            dU[zero_mag_mask] = epsilon
            dV[zero_mag_mask] = epsilon
            dQ[zero_mag_mask] = epsilon

            # now normalize
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

    # plotting 2D slices of the 3D flow field
    # get points within 20% of grid_spacing of PC3 = 0
    zvalids = get_slice_indexes(grid_spacing, "PC3", zgrid, verbose=verbose)

    # get points within 20% of self._grid_spacing of PC2 = 0
    yvalids = get_slice_indexes(grid_spacing, "PC2", ygrid, verbose=verbose)

    # plot quiver plots of these PC2 and PC3 slices overlaid on scatter plot of data
    fig, (ax1, ax2) = vb.init_subplots()
    ax1.scatter(df_cond.PC1, df_cond.PC2, s=0.25, color="black", alpha=0.1)
    ax1 = plot_one_slice_quiver((dU, dV), (xgrid, ygrid), zvalids, ax=ax1)
    ax1.set_title("PC3 = 0")
    ax2.scatter(df_cond.PC1, df_cond.PC3, s=0.25, color="black", alpha=0.1)    
    ax2 = plot_one_slice_quiver((dU, dQ), (xgrid, zgrid), yvalids, ax=ax2)
    ax2.set_title("PC2 = 0")
    
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

