import json
import fire
from typing import Dict
from cellsmap.util import io
from cyto_dl.api import CytoDLModel
from cellsmap.util import get_model_config_path, load_config
import json
from pathlib import Path


def apply_model(model_name:str, dataset_name, save_dir='results', structure: str = 'cdh5', overrides:Dict={}):
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    elif not isinstance(overrides, dict):
        raise ValueError('Overrides must be a dictionary or a string')
    # load model
    model = CytoDLModel()
    cfg_path = get_model_config_path(model_name)
    model.load_config_from_file(cfg_path)
    # apply overrides
    movie_path = io.get_zarr_path(dataset_name)
    overrides['data.dict_meta.path'] = movie_path
    overrides['paths.output_dir'] = save_dir

    model.override_config(overrides)
    model.print_config()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # apply model
    model.predict()


def apply_all(dataset_name):
    config = load_config('model')
    for model_config in config:
        name = model_config['name']
        apply_model(name, dataset_name, save_dir=f'results/{name}')


if __name__ == '__main__':
    fire.Fire()