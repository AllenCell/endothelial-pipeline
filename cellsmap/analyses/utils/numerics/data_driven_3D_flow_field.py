import numpy as np
from scipy import interpolate as spinterp
from typing import Callable

from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.utils.io import vtk_io


def set_3D_bounds_from_data(x: np.array, y: np.array, z: np.array, excluded_fraction: float=0.1) -> list[np.ndarray]:
    """
    Set the bounds for the 3D flow field based on the data (leaving out a fraction of the data 
    based on the excluded_fraction parameter).
    """
    bounds = []
    for var in [x, y, z]:
        bounds.append(np.percentile(var, [excluded_fraction, 100-excluded_fraction]))
    return bounds

def compute_extrapolated_flow_field(drift_kmcs:np.ndarray, 
                                    grid_centers:list[np.ndarray], 
                                    interpolator:str="nearest",
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

    # Create interpolators for dU, dV, and dQ
    if interpolator == "nearest": # nearest neighbor
        interpolator_U = spinterp.NearestNDInterpolator(valid_points, valid_values_U)
        interpolator_V = spinterp.NearestNDInterpolator(valid_points, valid_values_V)
        interpolator_Q = spinterp.NearestNDInterpolator(valid_points, valid_values_Q)
    elif interpolator == "linear": # linear interpolation
        interpolator_U = spinterp.LinearNDInterpolator(valid_points, valid_values_U)
        interpolator_V = spinterp.LinearNDInterpolator(valid_points, valid_values_V)
        interpolator_Q = spinterp.LinearNDInterpolator(valid_points, valid_values_Q)
    else:
        raise ValueError(f"Interpolator {interpolator} not recognized. Use 'nearest' or 'linear'.")

    # Find the indices of all points (including NaN points)
    all_points = np.array(np.indices(dU.shape)).reshape(len(dU.shape), -1).T

    # Interpolate the NaN points
    dU = interpolator_U(all_points).reshape(dU.shape)
    dV = interpolator_V(all_points).reshape(dV.shape)
    dQ = interpolator_Q(all_points).reshape(dQ.shape)

    flow_field_dict = {"velocities": (dU, dV, dQ), "grid": (xgrid, ygrid, zgrid)}

    return flow_field_dict

def get_callable_flow_field(flow_field_dict:dict) -> Callable:
    """
    Get a callable flow field via linear interpolation on computed values of f on the grid.
    """

    # get the interpolator for f_KM
    f_grid = np.stack(flow_field_dict["velocities"], axis=-1) # shape (num_bins_x, num_bins_y, num_bins_z, 3)
    X = np.moveaxis(np.array(flow_field_dict["grid"]),0,-1).reshape((-1,3))    
    f_interp = spinterp.LinearNDInterpolator(X, f_grid.reshape((-1, 3))) # interpolator for f_KM

    # define a callable function to pass into the ODE solver
    # for scipy.integrate.solve_ivp, need time as first argument
    # and x as second argument even if the system is time-independent
    def my_flow(t,x):
        # get interpolated value
        f_interp_val = f_interp(x)
        # return dx/dt = f(x)
        return f_interp_val
    
    return my_flow

def convert_coordinates_from_pc_to_volume(self, xpc:np.array, origin:float) -> np.array:
    xvol = (xpc - origin) / self._grid_spacing
    return xvol

def convert_coordinates_from_volume_to_pc(self, xvol:np.array, origin:float) -> np.array:
    xpc = origin + xvol*self._grid_spacing
    return xpc


