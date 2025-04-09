import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import pickle
from cellsmap.util import dataset_io
from cellsmap.analyses.utils.manifest_pca import get_outliers
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
    df = get_dataframe_by_fmsid(fmsid)
    return df

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


## Functions below load the manifest files from hard coded paths
## This is a temporary solution until we standardize the data handoff process
## In the future some of the functionality below will live in manifest_preprocessing

    
def load_manifest_to_df(verbose:bool=True) -> pd.DataFrame:
    '''
    Load manifest files of DiffAE model predictions to DataFrame.
    Right now, this is hard-coded to load the manifest files for specific
    datasets. This will be updated in the future once we standardize the
    data handoff process.

    Also adds necessary metadata columns to DataFrame:
        - dataset_name: name of dataset crop is from
        - T: timepoint of crop
        - FOV_ID: FOV of crop
        - description: descriptive metadata for dataset (flow conditions)
        - outlier: boolean indicator for outlier crops (bubble detection)
    
    Inputs:
    - verbose (optional): bool, whether to print out information about datasets loaded
    
    Outputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns
    '''
    # manifest files for most (older) datasets
    path_to_data_multi = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
    
    # manifest files for newer datasets
    path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
    path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

    df = read_file_to_dataframe(path_to_data_multi)
    df_1217 = read_file_to_dataframe(path_to_20241217)
    df_0224 = read_file_to_dataframe(path_to_20250224)

    df = pd.concat([df,df_1217,df_0224],ignore_index=True)

    # FOR NOW: drop 20241105 and 20241210 datasets from analysis, no longer in data_config.yaml
    df = df[~df.filename_or_obj.str.contains('20241105')]
    df = df[~df.filename_or_obj.str.contains('20241210')]

    # add metadata columns for dataset, FOV, and timepoint
    df = add_metadata_from_path(df,verbose=verbose)

    # add descriptive metadata for each dataset (flow conditions for each dataset)
    description_dic = get_descriptive_metadata(df)
    df = add_descriptive_metadata(df,description_dic)

    # add outlier indicator column to DataFrame
    df = get_outliers(df)

    return df

def add_metadata_from_path(df:pd.DataFrame,verbose:bool=True) -> pd.DataFrame:
    '''
    Add metadata columns to DataFrame df of single-crop features:
        - name of dataset
        - FOV 
        - timepoint
    from which the each crop (each row of the DataFrame) was taken.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data
    - verbose (optional): bool, whether to print out information about datasets loaded

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data with added metadata columns
    '''

    # add metadata columns for dataset, FOV, and timepoint (all from filename_or_obj column)
    df['dataset_name'] = df.filename_or_obj.apply(lambda s: get_dataset_name(Path(s).parent.parent.stem))
    df['T'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])//6)
    df['FOV_ID'] = df.filename_or_obj.apply(lambda s: int(s.split('/')[-1].split('_')[-1][2:-4])%6)

    # filepath for this dataset in manifest includes barcode, so we need to change the 
    # dataset_name value in df to match the name int data_config.yaml
    # this is a temporary fix until we standardize the data handoff process
    df.loc[df.dataset_name.str.contains('20250224'),'dataset_name'] = '20250224_20X'

    # drop filename_or_obj column
    df.drop(columns=['filename_or_obj'],inplace=True)

    if verbose: # print out list of datasets represented in feature data (verbose gets passed into get_list_of_datasets)
        _ = get_list_of_datasets(df,verbose=True)

    return df

def get_descriptive_metadata(df:pd.DataFrame) -> dict:
    '''
    Get descriptive metadata for each dataset present in the DataFrame df.

    Describes the experimental conditions for each dataset, e.g., "48_hours_at_30_dyncm2".
    
    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata column for dataset_name
        - The string in the dataset_name column should match the dataset name in data_config.yaml

    Outputs:
    - description_dic: dict, dictionary of dataset names and their descriptive metadata
    '''
    if 'dataset_name' not in df.columns:
        raise ValueError('Data must have a column for dataset_name')
    
    # get list of unique datasets in DataFrame
    list_of_datasets = get_list_of_datasets(df)

    # initialize dictionary to store descriptions
    description_dic = {}
    for mv_name in list_of_datasets:
        data_config = dataset_io.get_dataset_info(mv_name) # get dataset info from data_config.yaml

        flow_config = data_config['flow'] # get flow conditions for dataset
        num_flows = len(flow_config) # number of flow conditions in dataset

        shear_rate = [int(flow_config[i][-1]) for i in range(num_flows)] # get shear rate for each flow condition, last element in each list in flow_config
        shear_rate_str = [str(i)+'_dyncm2' for i in shear_rate] # convert shear rates to strings

        time_str = [str(int((flow_config[i][1]-flow_config[i][0])*5/60))+'_hours' for i in range(num_flows)] # get duration of each flow condition in hours
        description = '_'.join([time_str[i]+'_at_'+shear_rate_str[i] for i in range(num_flows)]) # concatenate time and shear rate for each flow condition
        description_dic[mv_name] = description # add description to dictionary

    return description_dic

def add_descriptive_metadata(df:pd.DataFrame,description_dic:dict|None=None) -> pd.DataFrame:
    '''
    Add dataset description metadata to DataFrame df. Describes the experimental conditions for each dataset.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata column for dataset_name
        - The string in the dataset_name column should match the dataset name in data_config.yaml
    - description_dic (optional): dict, dictionary of dataset names and their descriptive metadata
        - Describes the experimental conditions for each dataset, e.g., "48_hours_at_30_dyncm2"
        - Keys should match dataset names in data_config.yaml
        - If not provided, will be generated using get_descriptive_metadata
    '''
    assert 'dataset_name' in df.columns, 'Data must have a column for dataset_name'
    
    # if no description dictionary provided, generate it
    if description_dic is None:
        description_dic = get_descriptive_metadata(df)

    # add description column to DataFrame
    # if key in description_dic is in dataset_name column, add that description to description column
    for key in description_dic.keys():
        df.loc[df['dataset_name'].str.contains(key),'description'] = description_dic[key]

    return df

def get_dataset_name(ds_path:str,path_prefix:str|None=None,file_ext:str='.ome.zarr') -> str:
    '''
    Get dataset name from the provided filepath. This is the name of the dataset in data_config.yaml.
    This function is used to extract the dataset name from the filepath in the manifest file,
    so it assumes that the dataset name is the last part of the path before the file extension.

    Inputs:
    - ds_path: str, path to dataset file
        - In the manifest file, this is obtained from the `filename_or_obj` column as Path(string).parent.parent.stem for string in df['filename_or_obj']
    - path_prefix (optional): str, prefix to remove from ds_path
        - If not provided, uses default path_prefix (right now, a hard-coded path)
    - file_ext (optional): str, file extension to remove from ds_path
        - If not provided, uses default file extension '.ome.zarr' (right now, a hard-coded extension)

    Outputs:
    - dataset_name: str, name of dataset in data_config.yaml
    '''
    if path_prefix is None:
        path_prefix = '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/' # default path prefix

    dataset_name = ds_path.replace(path_prefix,'') # remove path prefix, only get dataset name (file name at end of path)
    dataset_name = dataset_name.replace(file_ext,'') # remove file extension

    # remove any additional information from path (these are not included in the data_config.yaml dataset names)
    if '_SLDY' in dataset_name:
        dataset_name = dataset_name.replace('_SLDY','')
    if '_timelapse' in dataset_name:
        dataset_name = dataset_name.replace('_timelapse','')
    return dataset_name

def get_list_of_datasets(df:pd.DataFrame,verbose:bool=False,print_path:bool=False) -> list:
    '''
    Get list of unique datasets from `dataset_name` metadata column in DataFrame df.
    
    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata column for dataset_name
    - verbose (optional): bool, whether to print out information about datasets loaded
    - print_path (optional): bool, whether to print out the full path of the dataset
        - If False, only prints the dataset name (as in data_config.yaml)
    
    Outputs:
    - mylist: list, list of unique datasets in DataFrame df
    '''
    assert 'dataset_name' in df.columns, 'Data must have a column for dataset_name'

    mylist = df['dataset_name'].unique().tolist()
    if verbose:
        print(f'List of datasets represented in feature data: ')
        for ds in mylist:
            if print_path:
                print(ds)
            else:
                print(get_dataset_name(ds))
    return mylist

def get_one_dataset(df:pd.DataFrame,ds_name:str) -> pd.DataFrame:
    '''
    Get DataFrame corresponding to dataset named ds_name only, as identitified by the dataset_name column.
    
    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata column for dataset_name
    - ds_name: str, name of dataset to get from DataFrame df
        - This string must match the dataset name in the dataset_name column of df, same 
           as the name of the dataset in data_config.yaml
    
    Outputs:
    - df_one_dataset: pd.DataFrame, DataFrame of feature data corresponding to dataset ds_name
    '''
    assert 'dataset_name' in df.columns, 'Data must have a column for dataset_name'

    assert ds_name in df['dataset_name'].unique().tolist(), f'Dataset {ds_name} not found in DataFrame'

    df_one_dataset = df[df['dataset_name'] == ds_name].copy()
    return df_one_dataset

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
    feat_cols=[str(i) for i in range(8)] 

    # get DataFrame for dataset ds_name only, add crop index column
    df_ = add_crop_index(get_one_dataset(df,ds_name))

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