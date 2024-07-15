import json
import fire
from typing import Dict
from cellsmap.util import io
from cyto_dl.api import CytoDLModel


def apply_model(cfg_path:str, dataset_name, save_dir='results', overrides:Dict={}):
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    elif not isinstance(overrides, dict):
        raise ValueError('Overrides must be a dictionary or a string')
    # load model
    model = CytoDLModel()
    model.load_config_from_file(cfg_path)
    # apply overrides
    movie_path = io.get_zarr_path(dataset_name)
    overrides['data.dict_meta.path'] = movie_path
    overrides['paths.output_dir'] = save_dir

    model.override_config(overrides)
    model.print_config()

    # apply model
    model.predict()


if __name__ == '__main__':
    fire.Fire(apply_model)