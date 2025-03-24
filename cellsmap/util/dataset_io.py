import yaml
import dask
import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
import dask.array
from typing import List, Dict, Any, Union, Tuple

# general methods
def get_prj_dir(is_test=False) -> Path:
    prj_dir = Path(__file__).parents[2] if not is_test else Path(__file__).parents[3] / 'tests/results'
    return prj_dir

def get_results_dir(results_folder_name: str, is_test=False) -> Path:
    prj_dir = get_prj_dir(is_test=is_test)
    out_dir = prj_dir / f'results/{results_folder_name}'
    return out_dir

# model methods
def load_config(config_type: str = 'data') -> List[Dict[str, Any]]:
    if config_type not in ['data', 'model','dynamics']:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / f'{config_type}_config.yaml'
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

# dataset methods
def get_available_datasets() -> List[str]:
    datasets = []
    config = load_config()
    for dataset in config:
        datasets.append(dataset['name'])
        print(dataset['name'])
    return datasets

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    config = load_config()
    for dataset in config:
        if dataset['name'] == dataset_name:
            return dataset
    raise ValueError(f'Dataset {dataset_name} not found in config file')

def get_frame(filename):
    return int(str(filename).split('.')[0][-4:])

def get_flow(dataset_name: str, T: float) -> Union[int, float]:
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
    flows = [flow for t_start, t_stop, flow in flow_info if t_start <= T < t_stop]
    return int(flows[0]) if flows else np.nan

def get_flow_in_frames(dataset_name: str) -> List[Tuple[Any, Any, Any]]:
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

def get_channel_index(dataset_name: str, channel_names: List[str]) -> List[int]:
    available_channels = get_available_channels(dataset_name)
    return [available_channels.index(channel) if channel in available_channels else None for channel in channel_names]

def get_specific_channel_order(dataset_name:str):
    gfp_index = get_dataset_info(dataset_name)['egfp_channel_index']
    bf_index = get_dataset_info(dataset_name)['brightfield_channel_index']
    return gfp_index, bf_index

def get_number_of_positions(dataset_name:str) -> int:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['n_positions']

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

def get_original_path(dataset_name: str) -> Path:
    """
    Example path format: /{date}/{dataset_name}.dir/{dataset_name_number}.imgdir
    """
    dataset_info = get_dataset_info(dataset_name)
    return Path(dataset_info['original_path'])

def get_barcode(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['barcode']

def get_microscope(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['microscope']

# model methods

def get_available_models():
    model_info = load_config('model')
    model_names = [model['name'] for model in model_info]
    for name in model_names:
        print(name)

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
