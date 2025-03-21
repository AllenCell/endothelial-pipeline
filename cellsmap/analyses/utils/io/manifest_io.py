import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import pickle

from cellsmap.util import dataset_io
from cellsmap.analyses.utils.manifest_pca import get_outliers

def make_savedir(savedir:str='',subfolders:bool=True) -> None:
    '''Create directory savedir if it does not exist and/or subfolders
    for various outputs of model fitting and analysis if those do not exist.'''
    if savedir == '': # default save directory called 'dynamics_output' made in head of repo
        parent_folder = Path(__file__).resolve().parent.parent
        savedir = str(parent_folder / 'dynamics_output/')
    if not os.path.exists(savedir):
        if not savedir.endswith('/'):
            savedir += '/'
        print("*** Creating directory to save results... \n")
        print(f"Directory: {savedir} \n")
        os.makedirs(savedir)
        os.makedirs(savedir+'outputs')
        os.makedirs(savedir+'figs')

def load_df(file_path:str) -> pd.DataFrame:
    '''Load Pandas DataFrame from file_path.'''
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path,sep='\t')
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f'File extension not supported: {file_path}')
    
def load_manifest_to_df(verbose=True) -> pd.DataFrame:
    '''Load manifest files of DiffAE model predictions to DataFrame.
    Right now, this is hard-coded to load the manifest files for specific
    datasets. This will be updated in the future once we standardize the
    data handoff process.

    Also adds necessary metadata columns to DataFrame:
        - dataset_name: name of dataset crop is from
        - T: timepoint of crop
        - FOV_ID: FOV of crop
        - description: descriptive metadata for dataset (flow conditions)
        - outlier: boolean indicator for outlier crops (bubble detection)
    
    Returns:
        - df (pd.DataFrame): DataFrame of feature data with metadata columns
    '''
    # manifest files for most (older) datasets
    path_to_data_multi = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
    # manifest files for newer datasets
    path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
    path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

    df = load_df(path_to_data_multi)
    df_1217 = load_df(path_to_20241217)
    df_0224 = load_df(path_to_20250224)

    df = pd.concat([df,df_1217,df_0224],ignore_index=True)

    # add metadata columns for dataset, FOV, and timepoint
    df = add_metadata_from_path(df,verbose=verbose)

    # add descriptive metadata for each dataset (flow conditions for each dataset)
    description_dic = get_descriptive_metadata(df)
    df = add_descriptive_metadata(df,description_dic)

    # add outlier indicator column to DataFrame
    df = get_outliers(df)

    return df

def add_metadata_from_path(df:pd.DataFrame,verbose:bool=True) -> pd.DataFrame:
    '''Add metadata columns to DataFrame df of single-crop features:
        - name of dataset
        - FOV 
        - timepoint
    from which the crop was taken.
    '''
    df['dataset_name'] = df.filename_or_obj.apply(lambda s: get_dataset_name(Path(s).parent.parent.stem))
    df['T'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])//6)
    df['FOV_ID'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])%6)

    # filepath for this dataset in manifest includes barcode, so we need to change the 
    # dataset_name value in df to match the name int data_config.yaml
    # something that should be fixed in the manifest in the future
    df.loc[df.dataset_name.str.contains('20250224'),'dataset_name'] = '20250224_20X'

    # drop filename_or_obj column
    df.drop(columns=['filename_or_obj'],inplace=True)

    if verbose:
        _ = get_list_of_datasets(df,verbose=True)

    return df

def get_descriptive_metadata(df:pd.DataFrame) -> dict:
    '''Get descriptive metadata for each dataset in list_of_datasets.
    Describes the experimental conditions for each dataset, 
    e.g., "48H low flow (date)".'''
    if 'dataset_name' not in df.columns:
        raise ValueError('Data must have a column for dataset_name')
    list_of_datasets = get_list_of_datasets(df)
    description_dic = {}
    for mv_name in list_of_datasets:
        data_config = dataset_io.get_dataset_info(mv_name) # get dataset info from data_config.yaml
        flow_config = data_config['flow'] # get flow conditions for dataset
        num_flows = len(flow_config) # number of flow conditions in dataset
        shear_rate = [flow_config[i][-1] for i in range(num_flows)] # get shear rate for each flow condition, last element in each list in flow_config
        shear_rate_str = [str(i)+' dyn/cm^2' for i in shear_rate] # convert shear rates to strings
        time_str = [str(flow_config[i][1]-flow_config[i][0])+' hours' for i in range(num_flows)] # get time of each flow condition
        description = ', '.join([time_str[i]+' at '+shear_rate_str[i] for i in range(num_flows)]) # concatenate time and shear rate for each flow condition
        description_dic[mv_name] = description
    return description_dic

def add_descriptive_metadata(df:pd.DataFrame,description_dic:dict) -> pd.DataFrame:
    '''Add metadata columns to DataFrame df.'''
    # if key in description_dic is in dataset_name column, add description to description column
    for key in description_dic.keys():
        df.loc[df['dataset_name'].str.contains(key),'description'] = description_dic[key]
    return df

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

def get_list_of_datasets(df:pd.DataFrame,verbose:bool=False,print_path:bool=False) -> list:
    '''Get list of unique datasets from metadata column in DataFrame df.'''
    if 'dataset_name' not in df.columns:
        raise ValueError('Data must have a column for dataset_name')
    mylist = np.unique(df['dataset_name'].values).tolist()
    if verbose:
        print(f'List of datasets represented in feature data: ')
        for ds in mylist:
            if print_path:
                print(ds)
            else:
                print(get_dataset_name(ds))
    return np.unique(df['dataset_name'].values).tolist()

def get_one_dataset(df:pd.DataFrame,ds_name:str) -> pd.DataFrame:
    '''Get DataFrame corresponding to dataset named ds_name only,
    as identitified by the dataset_name column.'''
    if 'dataset_name' not in df.columns:
        raise ValueError('Data must have a column for dataset_name')
    return df[df['dataset_name'] == ds_name].copy()

def get_flow_change_frame(ds_name:str) -> int:
    '''Get frame number at which flow changes in dataset ds_name.'''
    data_config = dataset_io.get_dataset_info(ds_name)
    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    return change_frame

def add_crop_index(df:pd.DataFrame) -> pd.DataFrame:
    '''Add crop index column to DataFrame df. (Crops are currently identified by their starting position in x and y.)'''    
    start_x = df[df['T']==df['T'].min()]['start_x'].values.tolist()
    start_y = df[df['T']==df['T'].min()]['start_y'].values.tolist()
    FOV_ID = df[df['T']==df['T'].min()]['FOV_ID'].values.tolist()
    tup_list = list(zip(start_x,start_y,FOV_ID))

    def pos_to_index(x,y,FOV):
        return tup_list.index((x,y,FOV))

    df['crop_index'] = df.apply(lambda x: pos_to_index(x['start_x'],x['start_y'],
                                                       x['FOV_ID']),axis=1)
    return df

def save_pca_model(pca:Pipeline,savedir:str) -> None:
    '''Save PCA model to file.'''
    if not savedir.endswith('/'):
        savedir += '/'
    with open(savedir+'pca_model.pkl','wb') as f:
        pickle.dump(pca,f)

def load_pca_model(savedir:str) -> Pipeline:
    '''Load PCA model from file.'''
    if not savedir.endswith('/'):
        savedir += '/'
    with open(savedir+'pca_model.pkl','rb') as f:
        pca = pickle.load(f)
    return pca

def project_PCA_one_dataset(df:pd.DataFrame,pca:Pipeline,ds_name:str,feat_cols:list=[str(i) for i in range(8)]) -> pd.DataFrame:
    '''Project feature data of crops from one dataset onto principal component axes.'''
    df_ = add_crop_index(get_one_dataset(df,ds_name))
    df_.loc[:,feat_cols] = pca.transform(df_[feat_cols].values)

    return df_

def df_to_array(df_:pd.DataFrame,feat_cols:list=[str(i) for i in range(8)]) -> np.ndarray:
    '''Convert DataFrame of features corresponding to one movie to array
    of shape num_crops x num_timepoints x num_features.'''

    num_T = df_['T'].nunique() # number of timepoints in the movie
    num_crop = df_['crop_index'].nunique() # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array([df_[df_['crop_index']==ii].sort_values(by='T')[feat_cols].values for ii in range(num_crop)])
    assert feats.shape == (num_crop,num_T,len(feat_cols))
    return feats