import yaml
from pathlib import Path
import numpy as np
from bioio import BioImage
import dask.array

def extract_key_from_config(key: str) -> str:
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / 'config.yaml'

    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data.get(key)

def get_frame(filename):
    return int(str(filename).split('.')[0][-4:])

def load_dataset(movie_name: str, time_start:int = 0, time_end: int=576, resolution:int=0) -> dask.array.Array:
    movie_path = extract_key_from_config(movie_name)
    # movie_path = localize_remote_filepath(extract_key_from_config(movie_name))
    img = BioImage(movie_path)
    assert resolution in img.resolution_levels, f'Invalid resolution level {resolution}. Available levels are {img.resolution_levels}'
    img.set_resolution_level(resolution)
    img = img.get_image_dask_data("TYX",T=range(time_start, time_end+1) )
    return img
