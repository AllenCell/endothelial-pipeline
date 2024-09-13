import yaml
import dask
import numpy as np
from pathlib import Path
from bioio import BioImage

def load_config(config_type='data') -> dict:
    if config_type not in ['data', 'model','dynamics']:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / f'{config_type}_config.yaml'
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data


# dataset methods
def get_available_datasets() -> list:
    config = load_config()
    for dataset in config:
        print(dataset['name'])

def get_dataset_info(dataset_name: str) -> dict:
    config = load_config()
    for dataset in config:
        if dataset['name'] == dataset_name:
            return dataset
    raise ValueError(f'Dataset {dataset_name} not found in config file')

def get_frame(filename):
    return int(str(filename).split('.')[0][-4:])

def load_dataset(dataset_name: str, time_start:int = 0, time_end: int=576, resolution:int=0) -> dask.array.Array:
    dataset_info = get_dataset_info(dataset_name)
    img = BioImage(dataset_info['zarr_path'])
    assert resolution in img.resolution_levels, f'Invalid resolution level {resolution}. Available levels are {img.resolution_levels}'
    img.set_resolution_level(resolution)
    img = img.get_image_dask_data("TYX",T=range(time_start, time_end+1) )
    return img

def get_zarr_path(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['zarr_path']

def get_xy_pixel_size_in_um(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['pixel_size_xy_in_um']

def get_time_interval_in_minutes(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['time_interval_in_minutes']

def get_flow_info(dataset_name: str) -> list:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['flow']


# model methods

def get_available_models() -> list:
    model_info = load_config('model')
    for model in model_info:
        print(model['name'])

def get_model_info(model_name: str) -> dict:
    config = load_config('model')
    for model in config:
        if model['name'] == model_name:
            return model
    raise ValueError(f'Model {model_name} not found in config file')

def get_model_config_path(model_name: str, task: str = 'eval') -> str:
    assert task in ['train', 'eval'], 'Invalid task. Must be either "train" or "eval"'
    model_info = get_model_info(model_name)
    return model_info[f'{task}_config_path']

# dynamics learning config functions

def get_dynamics_inputs(config_name: str) -> tuple:
    dynamics_config = load_config('dynamics')[config_name]
    dt = dynamics_config['dt'] # time interval between frames in minutes
    PCA = dynamics_config['PCA'] # if "yes", perform PCA on data before fitting dynamical model
    ndim = dynamics_config['ndim'] # number of principal components to keep if PCA is "yes"
    if PCA == 'yes':
        feats_to_analyze = None
        PCA = True
    else: # if PCA is "no", feats_to_analyze is a list of which of the original features to analyze (max 2 features)
        feats_to_analyze = dynamics_config['feats_to_analyze']
        PCA = False
    center_traj = True if dynamics_config['center_traj']=='yes' else False # if "yes", the initial conditions of all trajectories are centered at 0 before splitting (or not) and fitting dynamical model
    split_high_low = dynamics_config['split_high_low'] # if "yes", the data is split into high and low flow regimes before fitting dynamical model
    if split_high_low == 'yes':
        split_high_low = True
        split_frame = dynamics_config['split_frame']
        split_order = dynamics_config['split_order'] # if "high_low", the high flow regime comes before the low flow regime; else, low flow is first
    else:
        split_high_low = False
        split_frame = None
        split_order = None
    metadata_cols = dynamics_config['metadata_cols'] # list of names of metadata columns in the data (first is trajectory index, second is frame #)
    N = dynamics_config['N_bins'] # number of grid points in each dimension (int if 1D, tuple if 2D)
    nf = dynamics_config['poly_degree_drift'] # highest order of the polynomial terms in SINDy library for drift (int)
    ns = dynamics_config['poly_degree_diffusion'] # highest order of the polynomial terms in SINDy library for diffusion (int)
    savedir = dynamics_config['savedir'] # directory to save results
    return metadata_cols, PCA, ndim, dt, feats_to_analyze, center_traj, split_high_low, split_frame, split_order, N, nf, ns, savedir