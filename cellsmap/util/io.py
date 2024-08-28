import yaml
from pathlib import Path
import numpy as np
from bioio import BioImage
import dask.array
import re

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


def get_dim_map(dim_order: str):

    dims = [a for a in dim_order]
    dim_nums = tuple(range(len(dims)))
    dim_map = dict(zip(dims, dim_nums))

    return dim_map# -> tuple(int)


def extract_T(fp_as_string, int_only=True):
    try:
        if isinstance(fp_as_string, Path):
            fp_as_string = str(fp_as_string)
    except ImportError:
        pass
    t = re.search('T[0-9]+', fp_as_string)
    if t:
        t = int(t.group(0).split('T')[-1]) 
    else:
        t = 0 
        print("""No 'T[0-9]+' found in filename. Assuming only 
              1 timepoint and assuming T = 0.""")
        
    return t if int_only else f'T{t}'


def extract_C(fp_as_string, int_only=True):
    try:
        if isinstance(fp_as_string, Path):
            fp_as_string = str(fp_as_string)
    except ImportError:
        pass
    c = re.search('_C[0-9]+', fp_as_string)
    if c:
        c = int(c.group(0).split('_C')[-1]) 
    else:
        c = 0 
        print("""No 'C[0-9]+' found in filename. Assuming only
              1 channel and assuming C = 0.""")
        
    return c if int_only else f'C{c}'
