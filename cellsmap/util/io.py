import yaml
import dask
import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
import dask.array

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
    datasets = []
    config = load_config()
    for dataset in config:
        datasets.append(dataset['name'])
        print(dataset['name'])
    return datasets

def get_dataset_info(dataset_name: str) -> dict:
    config = load_config()
    for dataset in config:
        if dataset['name'] == dataset_name:
            return dataset
    raise ValueError(f'Dataset {dataset_name} not found in config file')

def get_frame(filename):
    return int(str(filename).split('.')[0][-4:])

def get_flow(dataset_name: str, T: float) -> list:
    """
    Parameters
    ----------
        T: the time at which to get the flow value.
    Returns
    -------
        flow: the flow value at time T in dyn/cm^2.
    """
    dataset_info = get_dataset_info(dataset_name)
    flow_info = dataset_info['flow']
    flows = lambda t: [flow for t_start, t_stop, flow in flow_info if t_start <= t < t_stop]
    flow = flows(T)
    return int(*flow) if flow else np.nan

def get_flow_in_frames(dataset_name: str) -> int:
    dataset_info = get_dataset_info(dataset_name)
    flow_info = dataset_info['flow']
    flow_in_frames = [(round(t_start * 60 / dataset_info['time_interval_in_minutes']), round(t_stop * 60 / dataset_info['time_interval_in_minutes']), flow) for t_start, t_stop, flow in flow_info]
    return flow_in_frames

def get_zarr_path(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['zarr_path']

def get_available_channels(dataset_name:str) -> list:
    path = get_zarr_path(dataset_name)
    reader = BioImage(path)
    return reader.channel_names

def get_channel_index(dataset_name:str, channel_names:list) -> int:
    available_channels = get_available_channels(dataset_name)
    return [available_channels.index(channel) if channel in available_channels else None for channel in channel_names]

def load_dataset(dataset_name:str, channels:list, time_start:int=0, time_end:int=-1, level:int=0) -> dask.array.Array:
    path = get_zarr_path(dataset_name)
    reader = BioImage(path)
    available_channels = reader.channel_names
    channels_index = [available_channels.index(c) for c in channels]
    assert level in reader.resolution_levels, f'Invalid resolution level {level}. Available levels are {reader.resolution_levels}'
    reader.set_resolution_level(level)
    if time_end < 0:
        time_end = get_dataset_duration_in_frames(dataset_name)-1
    img = reader.get_image_dask_data("TCYX", T=range(time_start, time_end+1), C=channels_index)
    return img

def get_dataset_duration_in_frames(dataset_name: str) -> int:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['duration']

def get_xy_pixel_size_in_um(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['pixel_size_xy_in_um']

def get_time_interval_in_minutes(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['time_interval_in_minutes']

def get_flow_info(dataset_name: str) -> list:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['flow']

def get_dim_map(dim_order: str) -> dict:

    dims = [a for a in dim_order]
    dim_nums = tuple(range(len(dims)))
    dim_map = dict(zip(dims, dim_nums))

    return dim_map

# model methods

def get_available_models() -> list:
    model_info = load_config('model')
    for model in model_info:
        print(model['name'])

def load_precomputed_features(dataset_name:str, model_name:str) -> pd.DataFrame:
    dataset_info = get_dataset_info(dataset_name)
    return pd.read_csv(dataset_info["features"][model_name])

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

def get_available_dynamics_configs() -> list:
    config = load_config('dynamics')
    for inputs in config:
        print(inputs['name'])

def get_dynamics_inputs(config_name: str) -> tuple:
    '''Unpack the dynamics config file to get the necessary inputs for the dynamics learning pipeline (analyses/workflows/fit_SDE_model.py).'''
    dynamics_config = None

    # load the specific dict for the config_name
    for config in load_config('dynamics'):
        if config['name'] == config_name:
            dynamics_config = config
            break
    dt = dynamics_config['dt'] # time interval between frames (units depend on the data & the selected value)

    PCA = dynamics_config['PCA'] # if "yes", perform PCA on data before fitting dynamical model
    ndim = dynamics_config['ndim'] # number of principal components to keep if PCA is "yes"
    if PCA == 'yes':
        feats_to_analyze = None
        PCA = True
    else: # if PCA is "no", feats_to_analyze is a list of which of the original features to analyze (max 2 features)
        feats_to_analyze = dynamics_config['feats_to_analyze']
        PCA = False

    center_traj = True if dynamics_config['center_traj']=='yes' else False # if "yes", the initial conditions of all trajectories are centered at 0 before splitting (or not) and fitting dynamical model
    
    split_flow = dynamics_config['split_flow'] # if "yes", the data is split into high and low flow regimes before fitting dynamical model
    if split_flow == 'yes':
        split_flow = True
        split_frame = dynamics_config['split_frame'] # frame(s) # at which to split the data into the flow regimes listed below in split_order
        split_order = dynamics_config['split_order'] # temporal order of the flow regimes (list of strings)
    else:
        split_flow = False
        split_frame = None
        split_order = None

    metadata_cols = dynamics_config['metadata_cols'] # list of names of metadata columns in the data (first is trajectory index, second is frame #)

    N = dynamics_config['N_bins'] # number of grid points in each dimension (int if 1D, tuple if 2D)
    auto_bin = dynamics_config['auto_bin'] # if "yes", automatically determine the number of bins for each dimension
    auto_bin = True if auto_bin == 'yes' else False
    bin_limits = dynamics_config['bin_limits'] # limits of the grid in each dimension (None if auto_bin True, else list of tuples)

    nf = dynamics_config['poly_degree_drift'] # highest order of the polynomial terms in SINDy library for drift (int)
    ns = dynamics_config['poly_degree_diffusion'] # highest order of the polynomial terms in SINDy library for diffusion (int)

    savedir = dynamics_config['savedir'] # directory where model outputs will be saved

    logging = dynamics_config['logging'] # if "yes", log results to a file
    if logging == 'yes':
        log_file = savedir+'logs/langevin_regression_log.txt'
    else:
        log_file = None

    return metadata_cols, PCA, ndim, dt, feats_to_analyze, center_traj, split_flow, split_frame, split_order, N, auto_bin, bin_limits, nf, ns, savedir, log_file

