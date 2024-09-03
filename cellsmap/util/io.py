import yaml
import dask
import numpy as np
from pathlib import Path
from bioio import BioImage
import dask.array

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

def load_dataset(dataset_name: str, time_start:int = 0, time_end: int=576, resolution:int=0, structure: str='CDH5_Tubulin') -> dask.array.Array:
    path = get_zarr_path(dataset_name)
    img = BioImage(path)
    assert resolution in img.resolution_levels, f'Invalid resolution level {resolution}. Available levels are {img.resolution_levels}'
    img.set_resolution_level(resolution)

    assert structure in img.channel_names, f"Invalid structure name {structure}, availabel structures are {img.channel_names}"
    structure_ch = img.channel_names.index(structure)

    img = img.get_image_dask_data("TYX",T=range(time_start, time_end+1), C=structure_ch)
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
