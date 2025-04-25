import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate as spinterp
from typing import Tuple

from cellsmap.analyses.utils.viz import viz_base as vb


def set_3D_bounds_from_data(x: np.array, y: np.array, z: np.array, excluded_fraction: float=0.1) -> list[np.ndarray]:
    """
    Set the bounds for the 3D flow field based on the data.
    """
    bounds = []
    for var in [x, y, z]:
        bounds.append(np.percentile(var, [excluded_fraction, 100-excluded_fraction]))
    return bounds

def compute_extrapolated_flow_field(drift_kmcs:np.ndarray, 
                                    grid_centers:list[np.ndarray], 
                                    verbose:bool=True) -> dict:
    '''
    Get flow field dx/dt = f(x) via estimates of drift (first Kramers-Moyal coefficient)
    for a given condition.
    '''

    # generate a mesh grid of points in the state space
    xgrid, ygrid, zgrid = np.meshgrid(*grid_centers, indexing='ij') # make meshgrid
    
    if verbose:
        print("Shape of grid:")
        print(xgrid.shape, ygrid.shape, zgrid.shape)
    
    assert drift_kmcs.shape == (len(grid_centers[0]), len(grid_centers[1]), len(grid_centers[2]), 3), \
        f"Shape of flow field {drift_kmcs.shape} does not match shape of grid {xgrid.shape}."

    # flow field: dx/dt = f(x) (drift, first Kramers-Moyal coefficient)
    dU = drift_kmcs[...,0]
    dV = drift_kmcs[...,1]
    dQ = drift_kmcs[...,2]

    # where KMCs have been masked to nan, extrapolate 
    # via nearest neighbors.
    # use spinterp.interpn with method='nearest'
    # Find the indices of valid (non-NaN) points
    valid_mask = ~np.isnan(dU)
    valid_points = np.array(np.nonzero(valid_mask)).T  # Get the coordinates of valid points
    valid_values_U = dU[valid_mask]  # Get the corresponding values for dU
    valid_values_V = dV[valid_mask]  # Get the corresponding values for dV
    valid_values_Q = dQ[valid_mask]  # Get the corresponding values for dQ

    # Create nearest neighbor interpolators for dU, dV, and dQ
    interpolator_U = spinterp.NearestNDInterpolator(valid_points, valid_values_U)
    interpolator_V = spinterp.NearestNDInterpolator(valid_points, valid_values_V)
    interpolator_Q = spinterp.NearestNDInterpolator(valid_points, valid_values_Q)

    # Find the indices of all points (including NaN points)
    all_points = np.array(np.indices(dU.shape)).reshape(len(dU.shape), -1).T

    # Interpolate the NaN points
    dU = interpolator_U(all_points).reshape(dU.shape)
    dV = interpolator_V(all_points).reshape(dV.shape)
    dQ = interpolator_Q(all_points).reshape(dQ.shape)

    flow_field_dict = {"velocities": (dU, dV, dQ), "grid": (xgrid, ygrid, zgrid)}

    return flow_field_dict

def plot_flow_field_slice(flow_field_dict:dict, 
                           df_cond:pd.DataFrame,
                           fig_savedir:str, 
                           verbose:bool=True) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    
    # get flow field
    dU, dV, dQ = flow_field_dict["velocities"]

    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]
    grid_spacing = xgrid[1,0,0] - xgrid[0,0,0]

    # get bounds of the grid
    xmin, xmax = xgrid[0,0,0], xgrid[-1,0,0]
    ymin, ymax = ygrid[0,0,0], ygrid[0,-1,0]
    zmin, zmax = zgrid[0,0,0], zgrid[0,0,-1]

    # plotting 2D slices of the 3D flow field
    # get points within 20% of grid_spacing of PC3 = 0
    pc3val = 0.0
    zgridmin = pc3val-0.8*grid_spacing
    zgridmax = pc3val+1.2*grid_spacing
    zvalids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax))[0]
    if verbose:
        print(f"Number of points found within ± {(0.2*grid_spacing):.3f} of PC3 = {pc3val}:")
        print(len(zvalids))
    # unravel the grid to get the indices of the points in the grid
    # that are within the z-range of interest
    zvalids = np.unravel_index(zvalids, zgrid.shape)

    # get points within 20% of self._grid_spacing of PC2 = 0
    pc2val = 0.0
    ygridmin = pc2val-0.8*grid_spacing
    ygridmax = pc2val+1.2*grid_spacing
    yvalids = np.where((ygrid.ravel()>ygridmin)&(ygrid.ravel()<ygridmax))[0]
    if verbose:
        print(f"Number of points found within ± {(0.2*grid_spacing):.3f} of PC2 = {pc2val}:")
        print(len(yvalids))
    # unravel the grid to get the indices of the points in the grid
    # that are within the y-range of interest
    yvalids = np.unravel_index(yvalids, ygrid.shape)

    fig, (ax1, ax2) = vb.init_subplots()
    ax1.scatter(df_cond.PC1, df_cond.PC2, s=0.25, color="black", alpha=0.1)
    ax1.quiver(xgrid[zvalids], ygrid[zvalids], dU[zvalids], dV[zvalids], color="red")
    ax2.scatter(df_cond.PC1, df_cond.PC3, s=0.25, color="black", alpha=0.1)
    ax2.quiver(xgrid[yvalids], zgrid[yvalids], dU[yvalids], dQ[yvalids], color="red")
    
    for ax, (qmin, qmax) in zip((ax1, ax2), [(ymin, ymax), (zmin, zmax)]):
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if ax==ax1 else "PC3")
        ax.set_ylim(qmin, qmax)
        ax.set_aspect("equal")
    
    condition = df_cond.description.unique()[0] # get the condition name for saving the plot
    plt.tight_layout()
    plt.show()
    vb.save_plot(fig, filename=fig_savedir+f"flow_field_pc_{condition}", dpi=72)

    return fig, ax1, ax2
    

def get_random_early_points(self, condition:str, npoints:int, buffer:float=0.1, tmax:int=50) -> np.array:
    # Sample no flow at early timepoints points for setting initial
    # condition of the simulations
    df_initial = self._df.loc[
        (self._df.description==condition)&
        (self._df["T"]<tmax)&
        (self._df.PC1>(1-buffer)*self._bounds.xmin)&
        (self._df.PC1<(1-buffer)*self._bounds.xmax)&
        (self._df.PC2>(1-buffer)*self._bounds.ymin)&
        (self._df.PC2<(1-buffer)*self._bounds.ymax)&
        (self._df.PC3>(1-buffer)*self._bounds.zmin)&
        (self._df.PC3<(1-buffer)*self._bounds.zmax)
    ]
    if len(df_initial) < npoints:
        raise Exception(f"Number of points available in condition {condition} is {len(df_initial)}.")
    df_initial = df_initial.sample(npoints).copy()
    
    for var, origin in zip(self._ss_vars, [self._bounds.xmin, self._bounds.ymin, self._bounds.zmin]):
        df_initial[var] = self.convert_coordinates_from_pc_to_volume(xpc=df_initial[var], origin=origin)

    print("Bounds of state space variables:")
    coords = df_initial[self._ss_vars].values
    print(coords.min(axis=0))
    print(coords.max(axis=0))
    if self._verbose:
        print("Shape of sampled coordinated:")
        print(coords.shape)
    return coords

def convert_coordinates_from_pc_to_volume(self, xpc:np.array, origin:float) -> np.array:
    xvol = (xpc - origin) / self._grid_spacing
    return xvol

def convert_coordinates_from_volume_to_pc(self, xvol:np.array, origin:float) -> np.array:
    xpc = origin + xvol*self._grid_spacing
    return xpc

def get_random_points(self, npoints: int, offset:int=5) -> np.array:
    xmin, xmax = self._bounds.xmin, self._bounds.xmax
    ymin, ymax = self._bounds.ymin, self._bounds.ymax
    zmin, zmax = self._bounds.zmin, self._bounds.zmax
    coords = [
        [offset+(((vmax-vmin)/self._grid_spacing)-2*offset)*np.random.rand() for (vmin, vmax) in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    ] for _ in range(npoints)]
    coords = np.array(coords)
    if self._verbose:
        print("Shape of sampled coordinated:")
        print(coords.shape)
    return coords

def simulate_particles_in_flow_field(self, condition, filename_prefix:str=None, npoints=500, initial_coords=None, target_nframes=100, use_pc_units=False, clusters=0):
    # condition can be either a string or a list of string that serve
    # as a secheduler for changes between landspace.

    if isinstance(condition, str):
        condition = [condition] * target_nframes

    coords = initial_coords
    if coords is None:
        coords = self.get_random_early_points(condition=condition[0], npoints=npoints)

    start_condition = condition[0]

    assert start_condition in self._flow_field.keys(), f"Flow field for condition {start_condition} has not been yet computed."

    tp = 0
    if filename_prefix is None:
        filename_prefix = start_condition

    output_path = self.get_vtk_folder() + filename_prefix
    save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")

    sim_speed = self.calculate_simulation_speed(target_nframes=target_nframes)

    evolution = []
    
    eps = 2 # small displacement in volume
    for tp in range(1, target_nframes):
        
        coords_new = []
        for r in coords:
            x, y, z = r
            # euler method using mean velocity as magnitude for vectors
            vx = self._flow_field[condition[tp]]["velocities"][0][int(x), int(y), int(z)]
            vy = self._flow_field[condition[tp]]["velocities"][1][int(x), int(y), int(z)]
            vz = self._flow_field[condition[tp]]["velocities"][2][int(x), int(y), int(z)]
            x_new = x + sim_speed * vx
            y_new = y + sim_speed * vy
            z_new = z + sim_speed * vz

            # periodic - if trajectory goes out of bounds, wrap around
            x_new = (x_new + eps) % ((self._bounds.xmax-self._bounds.xmin)/self._grid_spacing) - eps
            y_new = (y_new + eps) % ((self._bounds.ymax-self._bounds.ymin)/self._grid_spacing) - eps
            z_new = (z_new + eps) % ((self._bounds.zmax-self._bounds.zmin)/self._grid_spacing) - eps
    
            coords_new.append([x_new, y_new, z_new])

        evolution.append(np.array(coords_new))
        coords = np.array(coords_new).copy()
        save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")

    # save out the mean trajectory
    mean_evolution = []
    for coords in evolution:
        xc, yc, zc = coords.mean(axis=0)
        if use_pc_units: # convert to pc units instead of volume units
            xc = self._bounds.xmin+self._grid_spacing*xc
            yc = self._bounds.ymin+self._grid_spacing*yc
            zc = self._bounds.zmin+self._grid_spacing*zc
        mean_evolution.append([xc, yc, zc])
    mean_evolution = np.array(mean_evolution)
    save_points_as_polydata(coordinates=mean_evolution, file_name=f"{output_path}_mean_trajectory.vtk")

    # we want to get evenly spaced points along the trajectory (use self._grid_spacing as step size)
    # this is for visualization purposes (reconstruction of crops along mean traj)
    # first compute distance between points
    distances = np.linalg.norm(np.diff(mean_evolution, axis=0), axis=1)

    # compute cumulative distance from the first point along the trajectory
    arc_length = np.cumsum(np.concatenate(([0],distances)))

    # interpolate to get evenly spaced points at self._grid_spacing
    n_points = int(np.ceil(arc_length[-1] / self._grid_spacing))
    arc_length_new = np.linspace(0, arc_length[-1], n_points) # arc length distance of evenly spaced points
    interpolated_points = np.zeros((n_points, 3))
    for i in range(3):
        interpolated_points[:, i] = np.interp(arc_length_new, arc_length, mean_evolution[:, i])

    save_points_as_polydata(coordinates=interpolated_points, file_name=f"{output_path}_interpolated_mean_trajectory.vtk")

    return mean_evolution


