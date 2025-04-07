import yaml
import dask
import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
import dask.array
from typing import List, Dict, Any, Union, Tuple, Optional

def load_config(config_type: str = 'data') -> List[Dict[str, Any]]:
    if config_type not in ['data', 'model','dynamics']:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / f'{config_type}_config.yaml'
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

# dataset methods
def get_available_datasets(verbose: bool = True) -> List[str]:
    datasets = []
    config = load_config()
    for dataset in config:
        datasets.append(dataset['name'])
        if verbose:
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
    return [available_channels.index(channel) for channel in channel_names]

def get_specific_channel_order(dataset_name:str):
    gfp_index = get_dataset_info(dataset_name)['egfp_channel_index']
    bf_index = get_dataset_info(dataset_name)['brightfield_channel_index']
    return gfp_index, bf_index

def get_total_number_of_positions(dataset_name:str) -> int:
    """
    n positions is the product of n_scenes x n_positions_per_scene
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['n_total_positions']

def load_dataset(dataset_name:str, channels:List=["EGFP", "BF"], time_start:int=0, time_end:int=-1, level:int=0, zarr_name:Optional[str]=None) -> dict[str, dask.array.Array]:
    data_dir = get_zarr_path(dataset_name)
    dataset = {}

    if zarr_name:
        filepath = Path(data_dir) / zarr_name
        assert filepath.exists(), f'Zarr file {filepath} does not exist.'
        filepath_list = [filepath]
    else:
        filepath_list = [fp for fp in Path(data_dir).glob('*.zarr')]

    for filepath in filepath_list:
        reader = BioImage(filepath)
        available_channels = reader.channel_names
        channels_index = [available_channels.index(c) for c in channels]
        assert level in reader.resolution_levels, f'Invalid resolution level {level}. Available levels are {reader.resolution_levels}'
        reader.set_resolution_level(level)
        if time_end < 0:
            time_end = get_dataset_duration_in_frames(dataset_name)-1
        img = reader.get_image_dask_data("TCYX", T=range(time_start, time_end+1), C=channels_index)
        dataset[filepath.name] = img
    return dataset

def load_dataset_position_as_dask_array(dataset_name:str, position:int|str, channels:List=["EGFP", "BF"], time_start:int=0, time_end:int=-1, level:int=0) -> dask.array.Array:
    """
    position can be either an integer or a string.
    If it is a string then it must the name of a zarr file found in
    dataset (e.g. a folder ending with the .ome.zarr extension).
    If it is an integer then it will be used as the index to
    get the zarr file name from the dataset.
    """
    if isinstance(position, int):
        data_dir = get_zarr_path(dataset_name)
        filepath_list = sorted([fp for fp in Path(data_dir).glob('*.zarr')])
        zarr_name = filepath_list[position].name
    else:
        zarr_name = position
    img_dict = load_dataset(dataset_name, channels, time_start, time_end, level, zarr_name)
    img_dask_arr = img_dict[zarr_name]
    return img_dask_arr

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

def get_flow_change_frame(dataset_name:str) -> int:
    '''
    Get frame number at which flow changes in dataset ds_name.
    
    Inputs:
    - dataset_name: str, name of dataset to get flow change frame for
        - This string must match the dataset name in data_config.yaml
    
    Outputs:
    - change_frame: int, frame number at which flow changes in dataset dataset_name
    '''
    # load config for dataset from data_config.yaml
    flow_info = get_flow_info(dataset_name)

    # get frame number at which flow changes
    change_frame = flow_info[0][1]

    return change_frame

def get_flow_for_frame(dataset_name: str, frame: int) -> float | None:
    flow_list = get_flow_info(dataset_name)
    for t_start, t_stop, flow in flow_list:
        if t_start <= frame <= t_stop:
            return flow
    print(f"Frame {frame} not found in flow list.")
    return None

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

def get_fmsid(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['fmsid']

def get_nuclear_prediction_path(dataset_name: str, position: int) -> str:
    dataset_info = get_dataset_info(dataset_name)
    base_path = dataset_info['nuclear_label_free_seg_path']
    position_path = f"{base_path}/P{position}/"
    return position_path

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