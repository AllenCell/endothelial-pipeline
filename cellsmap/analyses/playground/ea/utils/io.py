import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
from pathlib import Path

def make_savedir(savedir:str,subfolders:bool=True) -> None:
    '''Create directory savedir if it does not exist and/or subfolders
    for various outputs of model fitting and analysis if those do not exist.'''
    if not os.path.exists(savedir):
        if not savedir.endswith('/'):
            savedir += '/'
        print("*** Creating directory to save results... \n")
        os.makedirs(savedir)
    if subfolders:
        os.makedirs(savedir+'data')
        os.makedirs(savedir+'outputs')
        os.makedirs(savedir+'figs')
        os.makedirs(savedir+'logs')

def load_array(file_path:str) -> pd.DataFrame:
    '''Load Pandas DataFrame from file_path.'''
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path,sep='\t')
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f'File extension not supported: {file_path}')

def get_PCA_reference(df:pd.DataFrame) -> pd.DataFrame:
    df['pca_ref'] = df.group.str.contains('20241120') | df.group.str.contains('20241203')
    # select no flow timepoints right after feeding
    df.loc[df.group.str.contains('20241210') & (df['T'] > 300) & (df['T'] < 450), 'pca_ref'] = True
    return df[df.pca_ref]

def add_metadata_from_path(df:pd.DataFrame) -> pd.DataFrame:
    '''Add metadata columns to DataFrame df.'''
    df['group'] = df.filename_or_obj.apply(lambda s: Path(s).parent.parent.stem)
    df['T'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])//6)
    df['FOV_ID'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])%6)
    return df

def rm_metadata(df:pd.DataFrame,metadata_col:list) -> pd.DataFrame:
    '''Remove metadata columns from DataFrame df.'''
    return df.drop(columns = metadata_col,inplace=False).astype(float)

def get_dataset_name(ds_path:str,path_prefix:str=None,file_ext:str='.ome.zarr') -> str:
    '''Get dataset name from dataset path.'''
    if path_prefix is None:
        path_prefix = '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/' # default path prefix
    dataset_name = ds_path.replace(path_prefix,'') # remove path prefix, only get dataset name (file name at end of path)
    dataset_name = dataset_name.replace(file_ext,'') # remove file extension
    if '_SLDY' in dataset_name:
        dataset_name = dataset_name.replace('_SLDY','')
    if '_timelapse' in dataset_name:
        dataset_name = dataset_name.replace('_timelapse','')
    return dataset_name

def get_list_of_datasets(df:pd.DataFrame,ds_metadata:str,verbose:bool=False,print_path:bool=False) -> list:
    '''Get list of unique datasets from metadata column in DataFrame df.'''
    mylist = np.unique(df[ds_metadata].values).tolist()
    if verbose:
        print(f'List of datasets represented in feature data: ')
        for ds in mylist:
            if print_path:
                print(ds)
            else:
                print(get_dataset_name(ds))
    return np.unique(df[ds_metadata].values).tolist()

def get_one_dataset(df:pd.DataFrame,ds_metadata:str,ds_ID:str) -> pd.DataFrame:
    '''Get DataFrame corresponding to one dataset, identified by value ds_ID in the
      metadata column ds_metadata in DataFrame df.'''
    return df[df[ds_metadata] == ds_ID].copy()

def add_crop_index(df:pd.DataFrame) -> pd.DataFrame:
    '''Add crop index column to DataFrame df. (Crops are currently identified by their starting position in x and y.)'''
    # df['crop_index'] = np.tile(np.arange(216), df.shape[0]//216) # placeholder for now
    # return df
    
    start_x = df[df['T']==0]['start_x'].values.tolist()
    start_y = df[df['T']==0]['start_y'].values.tolist()
    FOV_ID = df[df['T']==0]['FOV_ID'].values.tolist()
    tup_list = list(zip(start_x,start_y,FOV_ID))

    def pos_to_index(x,y,FOV):
        return tup_list.index((x,y,FOV))

    df['crop_index'] = df.apply(lambda x: pos_to_index(x['start_x'],x['start_y'],
                                                       x['FOV_ID']),axis=1)
    return df

def get_PCA(df:pd.DataFrame,n_components:int=None) -> PCA:
    '''Get PCA of feature array (scaled or not) X.
    Default is to return all components, unless n_components
    is specified. Returns singular values, explained variance
    ratio, and principal components.
    '''
    if n_components is not None:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=df.shape[1])
    pca.fit((df - df.mean()) / df.std())

    return pca

def project_PCA_one_dataset(df:pd.DataFrame,pca:PCA,ds_metadata:str,ds_ID:str,metadata_cols:list = ['start_x','start_y','T','crop_index']) -> np.ndarray:
    '''Project feature data of one dataset onto PCA components.'''
    df_ = add_crop_index(get_one_dataset(df,ds_metadata,ds_ID))
    metadata_cols_ = metadata_cols + ['crop_index']

    num_T = df_['T'].nunique() # number of timepoints in the movie
    num_crop = df_['crop_index'].nunique() # number of crops made at each timepoint

    feats_proj = df_.drop(columns = metadata_cols_).astype(float) # get feature data (no metadata)
    feats_proj = pca.transform(feats_proj).reshape(num_T,num_crop,-1) # reshape to (num_T, num_crop, num_components)
    feats_proj = np.swapaxes(feats_proj,0,1) # swap T and crop_index axes (num_crop, num_T, num_components)

    return feats_proj