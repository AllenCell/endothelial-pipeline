import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
import pickle
from cellsmap.util import dataset_io
import platform


try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import fms, FileLevelMetadataKeys
except ImportError:
    fms = None


def replace_base_url(file_path: str) -> str:
    """
    Replace the base URL 'production.files.allencell.org' with '/allen/programs/allencell/data/proj0/' in the given file path.
    
    Parameters:
    file_path (str): The original file path.
    
    Returns:
    str: The modified file path.
    """
    base_url = "production.files.allencell.org"
    new_base_path = "/allen/programs/allencell/data/proj0/"
    
    if base_url in file_path:
        modified_path = file_path.replace(base_url, new_base_path)
        return modified_path
    else:
        raise ValueError(f"The base URL '{base_url}' was not found in the provided file path.")


def get_valid_path(fpath) -> str:
    """
    Converts a FMS path to one that can be read cross-platform
    """
    if platform.system() == "Windows":
        fpath = "/" + fpath
    return fpath


def read_file_to_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a file into a pandas dataframe
    """
    if path.endswith("csv"):
        return pd.read_csv(path)
    elif path.endswith("parquet"):
        return pd.read_parquet(path)
    elif path.endswith("tsv"):
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unknown format {path.split('.')[-1]}")


def get_dataframe_by_fmsid(fmsid: str) -> pd.DataFrame:
    if fms is not None and os.path.exists("/allen/aics"):
        annotations = {FileLevelMetadataKeys.FILE_ID.value: fmsid}
        record = list(fms.find(annotations=annotations))[0]
        file_path = replace_base_url(record.path)
        path = get_valid_path(file_path)
    else:
        print("aicsfiles not installed or not on AICS intranet")
        # in the future this else statement will load from S3

    df = read_file_to_dataframe(path)
    return df


def get_nuclear_manifest(dataset_name: str) -> pd.DataFrame:
    fmsid = dataset_io.get_dataset_info(dataset_name)["nuclear_seg_manifest_fmsid"]
    df = get_dataframe_by_fmsid(fmsid)
    return df


def get_diffae_manifest(dataset_name: str) -> pd.DataFrame:
    fmsid = dataset_io.get_dataset_info(dataset_name)["diffae_manifest_fmsid"]
    if fmsid == "" or fmsid is None:
        print(f'No DiffAE manifest found for dataset {dataset_name}')
        return None
    df = get_dataframe_by_fmsid(fmsid)
    return df

def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Extract columns corresponding to DiffAE model features from dataframe (loaded DiffAE manifest).
    """
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split('_')[1]))
    return feat_cols

def list_datasets_with_manifest(manifest_name: str) -> list:
    """
    List all dataset names that have a 'nuclear_seg_manifest_fmsid' or 'diffae_manifest_fmsid'.
    """
    all_datasets = (
        dataset_io.get_available_datasets(verbose = False)
    ) 

    dataset_list = []
    for dataset_name in all_datasets:
        dataset_info = dataset_io.get_dataset_info(dataset_name)
        if manifest_name in dataset_info and dataset_info[manifest_name] != "":
            dataset_list.append(dataset_name)
    return dataset_list

def get_dataset_descriptions(list_of_datasets:list[str],simple:bool=False) -> dict:
    '''
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset, e.g., "48_hours_at_30_dyncm2".
    
    Inputs:
    - list_of_datasets: list, list of dataset names to get descriptions for
        - Each string should match the appropriate dataset name in data_config.yaml
    - simple (optional): bool, whether to use simple description (e.g., "48hr_High")


    Outputs:
    - description_dic: dict, dictionary of dataset names and their descriptive metadata
    '''

    # initialize dictionary to store descriptions
    description_dic = {}
    for name in list_of_datasets:
        data_config = dataset_io.get_dataset_info(name) # get dataset info from data_config.yaml

        flow_config = data_config['flow'] # get flow conditions for dataset
        num_flows = len(flow_config) # number of flow conditions in dataset

        shear_rate = [int(flow_config[i][-1]) for i in range(num_flows)] # get shear rate for each flow condition, last element in each list in flow_config
        if simple: # if simple description, use qualitative description of shear stress level
            shear_rate_str = []
            for shear in shear_rate:
                if shear >= 20:
                    shear_rate_str.append('High')
                elif shear > 7: 
                    shear_rate_str.append(f'Intermediate_{int(shear)}')
                elif shear > 0:
                    shear_rate_str.append('Low')
                else:
                    shear_rate_str.append('No')
        else:
            shear_rate_str = [f'{int(i)}_dyncm2' for i in shear_rate] # convert shear rates to strings

        time_str = [f'{int((flow_config[i][1]-flow_config[i][0])*5/60)}hr' for i in range(num_flows)] # get duration of each flow condition in hours
        description = '_'.join([time_str[i]+'_'+shear_rate_str[i] for i in range(num_flows)]) # concatenate time and shear rate for each flow condition
        description_dic[name] = description # add description to dictionary

    return description_dic

## Functions below load the manifest files from hard coded paths
## This is a temporary solution until we standardize the data handoff process
## In the future some of the functionality below will live in manifest_preprocessing

def add_crop_index(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Add crop index column to DataFrame df. (Crops are currently identified by their starting position in x and y.)
    
    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns for start_x, start_y, and FOV_ID
        - IMPORTANT: DataFrame must be restricted to one dataset only, as identified by the dataset_name column
    
    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one dataset with added crop index column
    '''
    assert 'start_x' in df.columns, 'Data must have a column for start_x'
    assert 'start_y' in df.columns, 'Data must have a column for start_y'
    assert 'FOV_ID' in df.columns, 'Data must have a column for FOV_ID'

    # get list of unique starting positions and FOV_IDs
    start_x = df[df['T']==df['T'].min()]['start_x'].values.tolist()
    start_y = df[df['T']==df['T'].min()]['start_y'].values.tolist()
    FOV_ID = df[df['T']==df['T'].min()]['FOV_ID'].values.tolist()
    tup_list = list(zip(start_x,start_y,FOV_ID))

    # function to convert starting position and FOV_ID to crop index
    def pos_to_index(x,y,FOV):
        return tup_list.index((x,y,FOV))

    # apply function to DataFrame to get crop index
    df['crop_index'] = df.apply(lambda x: pos_to_index(x['start_x'],x['start_y'], x['FOV_ID']),axis=1)

    return df

def save_pca_model(pca:Pipeline,savedir:str) -> None:
    '''
    Save fit PCA model to file using pickle.

    Inputs:
    - pca: Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - can include any preprocessing steps before PCA, e.g., scaling
    - savedir: str, directory to save PCA model to

    Outputs:
    - None, saves PCA model to file
    '''
    if not savedir.endswith('/'):
        savedir += '/'
    with open(savedir+'pca_model.pkl','wb') as f:
        pickle.dump(pca,f)

def load_pca_model(savedir:str) -> Pipeline:
    '''
    Load PCA model from file.
    
    Inputs:
    - savedir: str, directory to load PCA model from

    Outputs:
    - pca: Pipeline, fit PCA model loaded from file
    '''
    if not savedir.endswith('/'):
        savedir += '/'
    with open(savedir+'pca_model.pkl','rb') as f:
        pca = pickle.load(f)
    return pca

def project_PCA_one_dataset(df:pd.DataFrame,pca:Pipeline,ds_name:str) -> pd.DataFrame:
    '''
    Project feature data for crops from one dataset onto principal component axes of fit PCA model.
    
    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns for dataset_name, T, FOV_ID, start_x, start_y
    - pca: Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - can include any preprocessing steps before PCA, e.g., scaling
    - ds_name: str, name of dataset to project feature data for
        - This string must match the dataset name in the dataset_name column of df, same 
           as the name of the dataset in data_config.yaml
    
    Outputs:
    - df_: pd.DataFrame, DataFrame of feature data for crops from dataset ds_name projected onto PCA axes
    '''
    # feature columns to project onto PCA axes, currently all columns except metadata columns
    # this is assuming that there are 8 feature columns, will need to change if this is not the case
    feat_cols=get_feature_cols(df)

    df_ = df.copy() # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, replace feature columns with features projected onto PCA axes
    df_.loc[:,feat_cols] = pca.transform(df_[feat_cols].values)

    return df_

def df_to_array(df_:pd.DataFrame,feat_cols:list) -> np.ndarray:
    '''
    Convert DataFrame of features corresponding to one dataset to array 
    of shape num_crops x num_timepoints x num_features.
    
    Inputs:
    - df_: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T

    Outputs:
    - feats: np.ndarray, array of feature data for all crops at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    '''
    assert 'crop_index' in df_.columns, 'DataFrame must have a column for crop_index'
    assert 'T' in df_.columns, 'DataFrame must have a column for T'

    num_T = df_['T'].nunique() # number of timepoints in the movie
    num_crop = df_['crop_index'].nunique() # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array([df_[df_['crop_index']==ii].sort_values(by='T')[feat_cols].values for ii in range(num_crop)])
    
    # check that array shape is correct
    assert feats.shape == (num_crop,num_T,len(feat_cols))
    
    return feats