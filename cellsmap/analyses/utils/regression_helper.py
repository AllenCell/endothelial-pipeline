import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

import cellsmap.util.dataset_io as dio
import cellsmap.analyses.utils.manifest_io as mio

def get_bins(Nbins,data=None,bin_limits=None):
    '''Generate histogram bins for the data.'''
    if bin_limits is None: # Automatically determine bins based on data
        if data is None:
            raise ValueError('Please provide data or or upper and lower bounds for bins.')
        ndim = data[0].shape[1]
        bins = []
        centers = []
        for i in range(ndim):
            my_min = min([min(traj[:,i]) for traj in data])
            my_max = max([max(traj[:,i]) for traj in data])
            bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
            bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
            my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    else: # Use user-defined bins
        ndim = len(bin_limits)
        bins = []
        centers = []
        for i in range(ndim):
            my_bins = np.linspace(bin_limits[i][0], bin_limits[i][1], Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    return bins, centers

def get_X_by_flow(df_proj:pd.DataFrame,ds_name:str,verbose:bool=True) -> Tuple[list,list]:
    '''Get feature data for different flow conditions in dataset ds_name.
    Returns list of feature data (dataframes) and list of corresponding flow conditions.'''

    if 'outlier' in df_proj.columns:
        df_proj = df_proj[df_proj['outlier']==False] # remove outliers

    data_config = dio.get_dataset_info(ds_name)
    first_shear = float(data_config['flow'][0][-1])

    shear_list = [first_shear]
    if len(data_config['flow']) > 1:
        change_frame = mio.get_flow_change_frame(ds_name) # change from time in hours to frame number
        second_shear = float(data_config['flow'][1][-1])
        if verbose:
            print('Shear stress',first_shear,'dyn/cm^2 until frame',change_frame)
            print('Shear stress',second_shear,'dyn/cm^2 after frame',change_frame)
        data_flow1 = df_proj[df_proj['T']<change_frame].copy()
        data_flow2 = df_proj[df_proj['T']>=change_frame].copy()
        data_all = [data_flow1,data_flow2]
        shear_list.append(second_shear)
    else:
        if verbose:
            print('Constant shear stress')
        data_all = [df_proj.copy()]
    return data_all, shear_list

def get_X_dX_and_dT(X:pd.DataFrame,feat_cols:list) -> Tuple[list,list,list]:
    '''X is a pandas DataFrame with columns for each feature. Should have a column for time, a column
    for the crop index, and a column indicating an outlier point. Returns tuple of lists, 
    each a list of numpy arrays.'''
    if 'outlier' not in X.columns:
        raise ValueError('Data must have a column for outlier')
    if 'T' not in X.columns:
        raise ValueError('Data must have a column for time')
    if 'crop_index' not in X.columns:
        raise ValueError('Data must have a column for crop_index')
    
    X = X[X['outlier']==False] # remove outliers
    crop_list = X['crop_index'].unique()
    X_list = []
    dX_list = []
    dT_list = []
    for crop in crop_list:
        X_crop = X[X['crop_index']==crop].sort_values(by='T')
        num_T = X_crop['T'].nunique()
        assert X_crop[feat_cols].values.shape == (num_T,len(feat_cols))
        dX = np.diff(X_crop[feat_cols].values,axis=1)
        dT = np.diff(X_crop['T'].values)
        X_list.append(X_crop[feat_cols].values)
        dX_list.append(dX)
        dT_list.append(dT)
    return X_list, dX_list, dT_list


def KM_avg_ND(X_list,dX_list,dT_list,bins,dt=5):
    '''Kramers-Moyal average drift and diffusion estimates for N-dimensional data'''
    ndim = len(bins)
    n = len(X_list) # number of trajectories from which dX was computed
    my_list = [len(bins[i])-1 for i in range(ndim)]
    my_list = my_list + [ndim,n]
    f_KM = np.nan*np.ones(my_list)
    D_KM = np.nan*np.ones(f_KM.shape)
    f_err = np.nan*np.ones(f_KM.shape)
    a_err = np.nan*np.ones(f_KM.shape)
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

            # Estimate error by variance of samples in the bin
            if len(bin_mask) > 1: # if trajectory passes through bin more than once
                f_err[tuple(slices)][:,j] = np.nanstd(dXdt[bin_mask],axis=0)/np.sqrt(len(mask))
                a_err[tuple(slices)][:,j] = np.nanstd(dX2dt[bin_mask],axis=0)/np.sqrt(len(mask))

    # take average over all trajectories to get Kramers-Moyal drift and diffusion estimates
    f_KM_avg = np.nanmean(f_KM,axis=-1)
    D_KM_avg = np.nanmean(D_KM,axis=-1)
    # think about how to generalize standard deviation computation to short traj vs. long traj
    f_err_mean = np.nanmean(f_err,axis=-1)
    f_err_mean = np.nan_to_num(f_err_mean,nan=1e10)
    f_KM_std = np.nanstd(f_KM,axis=-1)
    f_KM_std = np.nan_to_num(f_KM_std,nan=1e10)
    f_err = f_err_mean + f_KM_std

    D_err_mean = np.nanmean(a_err,axis=-1)
    D_err_mean = np.nan_to_num(D_err_mean,nan=1e10)
    D_KM_std = np.nanstd(D_KM,axis=-1)
    D_KM_std = np.nan_to_num(D_KM_std,nan=1e10)
    D_err = D_err_mean + D_KM_std

    return f_KM_avg, D_KM_avg, f_err, D_err

def masked_vector_field(F,X):
    '''Mask out the vector field F defined over points X 
    where F(X) is NaN'''
    mask = np.where(np.isfinite(F))
    ndim = F.shape[-1]
    X_mask = X[mask].reshape((-1,ndim))
    F_mask = F[mask].reshape((-1,ndim))
    return F_mask, X_mask

def train_test_all(X,F,D,num_flow,train_frac=0.8,seed=47,concat=False):
    '''Split data for different flow conditions into training and testing sets (80/20 by default)'''
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    V_train = []
    V_test = []

    for j in range(num_flow):
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X[j], F[j], train_size=train_frac, random_state=seed+j)
        _, _, V_train_temp, V_test_temp = train_test_split(X[j], D[j], train_size=train_frac, random_state=seed+j) # same random seed to get same x points for train and test
        X_train.append(X_train_temp)
        X_test.append(X_test_temp)
        Y_train.append(Y_train_temp)
        Y_test.append(Y_test_temp)
        V_train.append(V_train_temp)
        V_test.append(V_test_temp)

    # concatenate lists into arrays (one train/test set for all flow conditions)
    if concat:
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        Y_train = np.concatenate(Y_train)
        Y_test = np.concatenate(Y_test)
        V_train = np.concatenate(V_train)
        V_test = np.concatenate(V_test)
    
    return X_train, X_test, Y_train, Y_test, V_train, V_test

def get_stationary_hist(data, bins,ndim=2,frame_index=-100):
    '''Get stationary histogram of data, using values 
    at time frame_index and on as the stationary data.'''

    if ndim == 2:
        p_hist, _, _ = np.histogram2d(np.concatenate([data[j][frame_index:,0] for j in range(len(data))]).flatten(),
                              np.concatenate([data[j][frame_index:,1] for j in range(len(data))]).flatten(), 
                              bins, density=True)
    elif ndim == 1:
        p_hist, _ = np.histogram(np.concatenate([data[j][frame_index:] for j in range(len(data))]).flatten(), 
                                 bins, density=True)
    
    else:
        p_hist, _ = np.histogramdd(np.concatenate([data[j][frame_index:] for j in range(len(data))],axis=0), 
                                   bins, density=True)
    return p_hist


