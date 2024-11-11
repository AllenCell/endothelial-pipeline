import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_array(df:pd.DataFrame,metadata_col=None) -> np.ndarray:
    ''' Get numpy array from dataframe df.
    Optional argument metadata_col is list of strings of column names
    that contain various metadata for the features (index, file path, time, etc.)
    to be left out of the numpy array.'''

    if metadata_col is not None:
        X = df[df.columns[~df.columns.isin(metadata_col)]].values
    else:
        X = df.values
    return X

def get_PCA(X:np.ndarray, n_components=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Get PCA of feature array (scaled or not) X.
    Default is to return all components, unless n_components
    is specified. Returns singular values, explained variance
    ratio, and principal components.
    '''
    if n_components is not None:
        pca = PCA(n_components=n_components).fit(X)
    else:
        pca = PCA().fit(X)
    return pca.singular_values_, pca.explained_variance_ratio_, pca.components_

def project_trajectories(df:pd.DataFrame, components:np.ndarray, traj_id:str, metadata_col=None) -> np.ndarray:
    '''Project trajectories (features grouped by metadata traj_id, 
    e.g., inidividual crops or cells, hence using df to organize) onto 
    specified principal components from get_PCA. Optional argument
    metadata_col is list of strings of column names
    that contain various metadata for the features (index, file path, time, etc.)
    to be left out of the numpy array.
    
    Returns array of size (n_trajectories, n_timepoints, n_components).'''

    X_t = []
    for idx in df[traj_id].unique():
        traj_df = df[df[traj_id]==idx] # just get the dataframe for the current trajectory
        traj_df.head()
        if metadata_col is not None:
            traj = traj_df[traj_df.columns[~traj_df.columns.isin(metadata_col)]].values
        else:
            traj = traj_df.values
        X_t.append(traj@components.T) # project trajectory onto specified principal components, append to list
        
    return np.array(X_t)

    