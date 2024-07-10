import fire
from typing import Dict
from cyto_dl.api import CytoDLModel
from cellsmap.util import get_dataset_info
import json


def apply_model(cfg_path:str, dataset_name, save_dir='results', overrides:Dict={}):
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    elif not isinstance(overrides, dict):
        raise ValueError('Overrides must be a dictionary or a string')
    # load model
    model = CytoDLModel()
    model.load_config_from_file(cfg_path)
    # apply overrides
    movie_path = get_dataset_info(dataset_name)['zarr_path']
    overrides['data.dict_meta.path'] = movie_path
    overrides['paths.output_dir'] = save_dir

    model.override_config(overrides)
    model.print_config()

    # apply model
    model.predict()


if __name__ == '__main__':
    fire.Fire(apply_model)