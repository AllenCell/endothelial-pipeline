import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

import cellsmap.util.dataset_io as dio
import cellsmap.analyses.utils.numerics.kramers_moyal as km

def get_bins(Nbins:list,data:pd.DataFrame|None=None,bin_limits:list|None=None) -> Tuple[list,list]:
    '''
    Generate histogram bins for computing Kramers-Moyal estimates from trajectories, either automatically based on data or user-defined.

    Inputs:
    - Nbins: list of number of bins in each dimension (list of length ndim, where ndim is the number of dimensions of the feature space)
    - data: list of numpy arrays, each array is the trajectory of a single crop in feature space (ndim = len(Nbins))
    - bin_limits: list of tuples, each tuple contains the lower and upper bounds for the bins in each dimension
    Either data or bin_limits must be provided. If bin_limits provided, data is ignored.

    Outputs:
    - bins: list of numpy arrays, each array contains the bin edges for a dimension
    - centers: list of numpy arrays, each array contains the center of each bin in a dimension

    If the dimension is 1, bins and centers are still lists (of length 1), containing the bin edges and centers for the single dimension.
    '''
    if bin_limits is None: # Automatically determine bins based on data
        if data is None:
            raise ValueError('Please provide data or or upper and lower bounds for bins.')
        ndim = data[0].shape[1]
        assert ndim == len(Nbins), 'Number of bins must match number of dimensions in data.'
        bins = []
        centers = []
        for i in range(ndim):
            traj_concat = np.stack([traj[:,i] for traj in data],axis=1) # vertically stack all trajectories
            bin_min, bin_max = traj_concat.min() - 0.1, traj_concat.max() + 0.1
            my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    else: # Use user-defined bins
        ndim = len(bin_limits)
        assert ndim == len(Nbins), 'Number of bins must match number of dimensions in data.'
        bins = []
        centers = []
        for i in range(ndim):
            my_bins = np.linspace(bin_limits[i][0], bin_limits[i][1], Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    return bins, centers

def get_X_by_flow(df_proj:pd.DataFrame,ds_name:str,verbose:bool=True) -> Tuple[list,list]:
    '''
    Get crop-based feature data (Diffusion AE output) for different flow conditions present in dataset ds_name.

    Inputs:
    - df_proj: pandas dataframe containing the dataset of interest, projected onto all principal component axes (change of basis, no dimensionality reduction)
    - ds_name: name of the dataset (used to split out data by flow condition, acessed via data_config.yaml)

    Outputs:
    - data_all: list of dataframes, each containing the feature data for one flow condition
    - shear_list: list of shear stress conditions for each flow condition
    
    If there is only one flow condition, data_all and shear_list are still lists (of length 1), respectively containing the original dataframe and single shear stress condition.
    '''

    if 'outlier' in df_proj.columns:
        df_proj = df_proj[df_proj['outlier']==False] # remove outliers (bubble detection)

    # load flow information from data_config.yaml
    flow_info = dio.get_flow_info(ds_name)

    # split out data by flow condition, starting with first flow condition
    first_shear = float(flow_info[0][-1])
    # initialize list of shear stress conditions
    shear_list = [first_shear]
    if len(flow_info) > 1: # if there is a change in flow condition
        # get frame number where flow condition changes (reported in hours in data_config.yaml)
        change_frame = dio.get_flow_change_frame(ds_name)
        # get second shear stress condition
        second_shear = float(flow_info[1][-1])
        shear_list.append(second_shear)
        if verbose: # option to print out shear stress conditions and frame number where flow condition changes
            print('Shear stress',first_shear,'dyn/cm^2 until frame',change_frame)
            print('Shear stress',second_shear,'dyn/cm^2 after frame',change_frame)
        # separate data into two dataframes based on frame number where flow condition changes
        data_flow1 = df_proj[df_proj['T']<change_frame].copy()
        data_flow2 = df_proj[df_proj['T']>=change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1,data_flow2]
    else: # else, there is only one flow condition
        if verbose:
            print('Constant shear stress at',first_shear,'dyn/cm^2')
        # list of dataframes for one flow condition = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list

def get_X_dX_and_dT(X:pd.DataFrame,feat_cols:list) -> Tuple[list,list,list]:
    '''
    Get list of per-crop trajectories, the corresponding displacement vectors, and time differences along the trajectory for each crop in the dataset.

    Inputs:
    - X: pandas DataFrame with columns for each feature. Should have a column for time, a column for the crop index, and a column indicating an outlier point.
        This data should be for one dataset and one flow condition.
    - feat_cols: list of feature column names (used to extract feature data from the dataframe X)

    Outputs:
    - X_list: list of numpy arrays, each array is the trajectory of a single crop in feature space
    - dX_list: list of numpy arrays, each array is the displacement vectors along that trajectory for a single crop in feature space
    - dT_list: list of numpy arrays, each array is the time differences along that trajectory for a single crop
    '''
    if 'outlier' not in X.columns:
        raise ValueError('Data must have a column for outlier')
    if 'T' not in X.columns:
        raise ValueError('Data must have a column for time')
    if 'crop_index' not in X.columns:
        raise ValueError('Data must have a column for crop_index')
    
    X = X[X['outlier']==False] # remove outliers

    # get list of unique crop indices
    crop_list = X['crop_index'].unique()

    # initialize lists for storing data
    X_list = []
    dX_list = []
    dT_list = []

    # loop over each crop in the dataset
    for crop in crop_list:
        # get data for each crop, sorted by time
        X_crop = X[X['crop_index']==crop].sort_values(by='T')

        num_T = X_crop['T'].nunique() # number of timepoints for this crop
        # check that the array of feature data has the correct shape (num_T x ndim)
        assert X_crop[feat_cols].values.shape == (num_T,len(feat_cols))

        # get displacement vectors and time differences for each crop
        dX = np.diff(X_crop[feat_cols].values,axis=0)
        dT = np.diff(X_crop['T'].values)

        # append data to lists: trajectory, displacement vectors, time differences
        X_list.append(X_crop[feat_cols].values)
        dX_list.append(dX)
        dT_list.append(dT)

    return X_list, dX_list, dT_list

def get_kramers_moyal(X_list:list[np.ndarray], dX_list:list[np.ndarray], dT_list:list[np.ndarray], 
                      bins:list[np.ndarray], dt:float, method:str='kernel') -> Tuple[np.ndarray,np.ndarray]:
    ''' 
    Wrapper function for Kramers-Moyal coefficients for drift and diffusion estimates.
    Calls either the kernel or histogram method for estimating Kramers-Moyal coefficients.
    These functions are defined in cellsmap.analyses.utils.numerics.kramers_moyal.py.

    Inputs:
    - X_list: list of numpy arrays, each array is a single trajectory in feature space
    - dX_list: list of numpy arrays, each array is the displacement vectors along that trajectory
    - dT_list: list of numpy arrays, each array is the time differences along that trajectory
    - bins: list of numpy arrays, each array contains the bin edges for a dimension (used for computing conditional averages)
    - dt: time step between data points (used to compute Kramers-Moyal coefficients)
    - method: method to use for estimating Kramers-Moyal coefficients (default is 'kernel')

    Outputs:
    - f_KM: numpy array, drift estimates for each bin in feature space
    - D_KM: numpy array, diffusion estimates for each bin in feature space
    '''
    if method == 'kernel':
        f_KM, D_KM = km.get_km_kernel(X_list, dX_list, dT_list, bins, dt)
    elif method == 'histogram':
        f_KM, D_KM = km.get_km_histogram(X_list, dX_list, dT_list, bins, dt)
    else:
        raise ValueError('Method must be either kernel or histogram.')
    return f_KM, D_KM

def masked_vector_field(F:np.ndarray, X:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    '''
    For the vector field F over grid X, mask out F at points X where F(X) is NaN.

    Inputs:
    - F: numpy array (n_1 x n_2 x ... x n_ndim x ndim), ndim-D vector field evaluated on a meshgrid
    - X: numpy array (n_1 x n_2 x ... x n_ndim x ndim), ndim-D meshgrid where vector field is evaluated

    Outputs:
    - F_mask: numpy array (n x ndim), masked vector field flattened to 2D array (n = number of non-NaN points)
    - X_mask: numpy array (n x ndim), masked meshgrid flattened to 2D array (n = number of non-NaN points)
    '''
    # mask out NaN values in F
    mask = np.where(np.isfinite(F))
    ndim = F.shape[-1]

    # mask and flatten F and X over grid
    X_mask = X[mask].reshape((-1,ndim))
    F_mask = F[mask].reshape((-1,ndim))

    return F_mask, X_mask

def train_test_all(X:list[np.ndarray], 
                   F:list[np.ndarray], 
                   D:list[np.ndarray], 
                   train_frac:float=0.8, 
                   seed:int=47) -> tuple:
    '''
    Split feature data from a given dataset into training and testing sets for each flow condition present in the dataset.

    Inputs:
    - X: list of numpy arrays, each array contains the points in feature space for a single flow condition
    - F: list of numpy arrays, each array contains the drift estimates for each point in feature space in X for a single flow condition
    - D: list of numpy arrays, each array contains the diffusion estimates for each point in feature space in X for a single flow condition
    - train_frac: fraction of data to use for training (default = 0.8)
    - seed: random seed for train/test split (default = 47)

    Outputs:
    - X_train: points in feature space corresponding to the drift and diffusion estimates in the training sets
    - X_test: points in feature space corresponding to the drift and diffusion estimates in the test sets
    - Y_train: training data for drift estimates
    - Y_test: test data for drift estimates
    - V_train: training data for diffusion estimates
    - V_test: test data for diffusion estimates

    If concat=True, X_train, X_test, Y_train, Y_test, V_train, and V_test are all numpy arrays. Else, they are lists of numpy arrays, one for each flow condition.
    '''
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    V_train = []
    V_test = []

    num_flow = len(X)

    # get train/test split for each flow condition
    for j in range(num_flow):
       
        X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X[j], F[j], train_size=train_frac, random_state=seed+j)
        _, _, V_train_, V_test_ = train_test_split(X[j], D[j], train_size=train_frac, random_state=seed+j) # same random seed to get same x points for train and test
        X_train.append(X_train_)
        X_test.append(X_test_)
        Y_train.append(Y_train_)
        Y_test.append(Y_test_)
        V_train.append(V_train_)
        V_test.append(V_test_)

    # concatenate all data into one array, one train/test for all flow conditions
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)
    V_train = np.concatenate(V_train)
    V_test = np.concatenate(V_test)
    
    return X_train, X_test, Y_train, Y_test, V_train, V_test

def get_stationary_hist(data:pd.DataFrame, feat_cols:list, bins:list, frame_index:int=-100) -> np.ndarray:
    '''
    Get stationary histogram of data.
    
    Inputs:
    - data: pandas DataFrame containing the dataset of interest
    - feat_cols: list of feature column names (used to extract feature data from the dataframe data)
    - bins: list of number of bins in each dimension (list of length ndim, where ndim is the number of dimensions of the feature space)
    - frame_index: index of the time point (frame number) to use as the reference point for reaching stationarity (default is 100 timepoints before the last frame)

    Outputs:
    - p_hist: numpy array, stationary histogram of the data in feature space
    '''
    ndim = len(feat_cols)
    T_max = data['T'].max()
    if frame_index < 0: # if negative, frame_index is relative to the last frame
        frame_index = T_max + frame_index

    # call 1D or 2D histogram function based on number of dimensions
    if ndim == 2:
        # data T > frame_index, all rows, columns feat_cols[0] and feat_cols[1]
        p_hist, _, _ = np.histogram2d(data[data['T']>frame_index][feat_cols[0]], 
                                      data[data['T']>frame_index][feat_cols[1]], bins, density=True)
    elif ndim == 1:
        p_hist, _ = np.histogram(data[data['T']>frame_index][feat_cols[0]], bins[0], density=True)
    else:
        raise ValueError('Only 1D or 2D data supported.')
    
    return p_hist


