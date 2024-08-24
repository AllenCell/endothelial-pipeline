import yaml
import dask
import numpy as np
from pathlib import Path
from bioio import BioImage

def load_config(config_type='data') -> dict:
    if config_type not in ['data', 'model']:
        raise ValueError('Invalid config type. Must be either "data" or "model"')
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

def get_zarr_path(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['zarr_path']

def get_available_channels(dataset_name:str) -> list:
    path = get_zarr_path(dataset_name)
    reader = BioImage(path)
    return reader.channel_names

def load_dataset(dataset_name:str, channels:list, time_start:int=0, time_end:int=576, level:int=0) -> dask.array.Array:
    path = get_zarr_path(dataset_name)
    reader = BioImage(path)
    available_channels = reader.channel_names
    channels_index = [available_channels.index(c) for c in channels]
    assert level in reader.resolution_levels, f'Invalid resolution level {level}. Available levels are {reader.resolution_levels}'
    reader.set_resolution_level(level)
    img = reader.get_image_dask_data("TCYX", T=range(time_start, time_end+1), C=channels_index)
    return img

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