import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from itertools import product

# suppress RuntimeWarnings that come up - happens when taking the mean of an empty array
# occurs in KM_avg_ND function for bins with no data points
# probably a better way to handle this, but for now, suppress warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import cellsmap.analyses.utils.numerics.kramersmoyal.kmc as km

def get_km_powers(ndim:int) -> np.ndarray:
    '''
    Generate the powers for the Kramers-Moyal coefficients based on the dimensionality of the data.

    Inputs:
    - ndim: number of dimensions in the data

    Outputs:
    - powers: numpy array of powers for Kramers-Moyal coefficients

    For example, for 1D data, the powers are:
    [[0],  # normalization for kernel convolution (density)
     [1],  # f
     [2]]  # D

    For 2D data, the powers are:
    [[0,0],  # normalization for kernel convolution (density)
     [1,0],  # f_1
     [0,1],  # f_2
     [2,0],  # D_1
     [0,2]]  # D_2
    '''

    if ndim == 1: # straightforward case for 1D data
        powers = np.array([[0], [1], [2]])
        #                   /    f    D
        #          index:   0    1    2
    else: # if ndim > 1, utilize identity matrix to generate powers
        n_powers = 2*ndim + 1
        powers = np.zeros((n_powers, ndim), dtype=int) # row 0 is all zeros
        # drift powers: row 1 to ndim
        powers[1:ndim+1] = np.eye(ndim, dtype=int)
        # diffusion powers: row ndim+1 to end (no interaction terms)
        powers[ndim+1:] = 2*np.eye(ndim, dtype=int)
        # # For higher dimensions, generate all combinations of powers
        # powers = np.array(sorted(product(*(range(ndim+1),) * ndim),
        #                           key=lambda x: (max(x), x)))
    return powers

def get_km_kernel(X_list:list,dX_list:list,dT_list:list,bins:list,dt:float,kernel_params:dict) -> Tuple[np.ndarray,np.ndarray]:

    for i, dT in enumerate(dT_list):
        mask = np.where(dT==1)[0] # where outlier points were removed, time difference was greater than 1, mask out these points
        # mask_ for trajectories: should be mask but with additional point
        # include frame after last frame in mask
        mask_ = np.concatenate((mask, [mask[-1]+1]))
        X_list[i] = X_list[i][mask_]
        dX_list[i] = dX_list[i][mask]
        
    ndim = len(bins)
    powers = get_km_powers(ndim)

    kmc = km.km(X_list, grads=dX_list, bins=bins, 
                bw = kernel_params['bandwidth'],
                kernel = kernel_params['kernel'],
                powers = powers, multi_traj=True) / dt

    if ndim == 1: # just need to take the first two rows
        f_KM = kmc[1]
        D_KM = kmc[2]
    else: # if ndim > 1, need to make sure arrays are in the right shape
        axes_permute = [0] + list(reversed(range(1, ndim+1))) # permuted axes (0, ndim, ndim-1, ..., 1)
        kmc = np.transpose(kmc,axes_permute) # swap last ndim axes to get correct shape: n_powers x N[ndim] x N[ndim-1] x ... x N[1]
        f_KM = kmc[1:ndim+1].T # take drift terms, shape is N[1] x N[2] x ... x N[ndim] x ndim
        D_KM = kmc[ndim+1:].T # take diffusion terms, shape is N[1] x N[2] x ... x N[ndim] x ndim

    return f_KM, D_KM

def get_km_histogram(X_list:list,dX_list:list,dT_list:list,bins:list,dt:float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Kramers-Moyal average drift and diffusion estimates for trajectories in N-dimensional space.

    Inputs:
    - X_list: list of -+numpy arrays, each array is a single trajectory in feature space
    - dX_list: list of numpy arrays, each array is the displacement vectors along that trajectory
    - dT_list: list of numpy arrays, each array is the time differences along that trajectory
    - bins: list of numpy arrays, each array contains the bin edges for a dimension (used for computing conditional averages)
    - dt: time step between data points (used to compute Kramers-Moyal coefficients)

    Outputs:
    - f_KM_avg: numpy array, average drift estimate for each bin in feature space (average taken over all trajectories)
    - D_KM_avg: numpy array, average diffusion estimate for each bin in feature space (average taken over all trajectories)
    - f_err: numpy array, error estimate for drift (standard deviation of samples in each bin)
    - D_err: numpy array, error estimate for diffusion (standard deviation of samples in each bin)
    '''
    ndim = len(bins)
    n = len(X_list) # number of trajectories from which dX was computed
    my_list = [len(bins[i])-1 for i in range(ndim)]
    my_list = my_list + [ndim,n]
    f_KM = np.nan*np.ones(my_list)
    D_KM = np.nan*np.ones(f_KM.shape)
    for (j,X) in enumerate(X_list):
        dX = dX_list[j]
        dT = dT_list[j]
        mask = np.where(dT==1)[0] # where outlier points were removed, time difference was greater than 1, mask out these points
        X = X[mask]
        dXdt = dX[mask]/dt # displacement divided by time step to get velocity (for fitting drift)
        dX2dt = dX**2/dt # squared displacement divided by time step (for fitting diffusion)

        id_list = [np.digitize(X[:-1,i],bins[i]) for i in range(ndim)] # which bin each data point falls into (by each dimension)
        uids = list(set(zip(*id_list))) # unique bin ids (zipped tuple of bin ids by dimension)
        if any([len(bins[i]) in id_list[i] for i in range(ndim)]):
            raise ValueError('Data point outside of histogram bins. Please update bounds.')

        for uid in uids:
            my_cond = 1
            for i in range(ndim):
                my_cond = my_cond*(id_list[i]==uid[i])
            bin_mask = np.where(my_cond)[0]
            # At each histogram bin, find time series points where the state falls into this bin
            slices = [uid[i]-1 for i in range(ndim)]
            f_KM[tuple(slices)][:,j] = np.mean(dXdt[bin_mask],axis=0) # Conditional average  ~ drift
            D_KM[tuple(slices)][:,j] = 0.5*np.mean(dX2dt[bin_mask],axis=0) # Conditional variance  ~ diffusion

    # take average over all trajectories to get Kramers-Moyal drift and diffusion estimates
    f_KM = np.nanmean(f_KM,axis=-1)
    D_KM = np.nanmean(D_KM,axis=-1)

    return f_KM, D_KM

def plot_km_estimates(fig_ax, X_1,X_2,km_predictions):
    fig, ax_00, ax_01, ax_10, ax_11 = fig_ax
    Y_1_pred, Y_2_pred, V_1_pred, V_2_pred = km_predictions

    ax_00.contour3D(X_1, X_2, Y_1_pred.reshape(X_1.shape), 50, cmap='Blues')
    ax_01.contour3D(X_1, X_2, Y_2_pred.reshape(X_1.shape), 50, cmap='Blues')
    ax_10.contour3D(X_1, X_2, V_1_pred.reshape(X_1.shape), 50, cmap='Blues')
    ax_11.contour3D(X_1, X_2, V_2_pred.reshape(X_1.shape), 50, cmap='Blues')

    return fig, ax_00, ax_01, ax_10, ax_11