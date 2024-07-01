import yaml
from pathlib import Path
import numpy as np
from bioio.bio_image import imread

def extract_key_from_config(key: str) -> str:
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / 'config.yaml'

    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data.get(key)

def get_tp(filename):
    return int(str(filename).split('.')[0][-4:])

def load_movie(movie_name: str, time_start:int = 0, time_end: int=576) -> np.ndarray:
    movie_path = Path(extract_key_from_config('cdh5_dir'))
    movie = np.stack([imread(fn) for fn in sorted(movie_path.glob('*tif*')) if time_start <= get_tp(fn) <= time_end])

    return movie
