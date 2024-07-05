import fire
from typing import Dict
from cyto_dl.api import CytoDLModel
from cellsmap.util import extract_key_from_config


def apply_model(cfg_path:str, dataset_name, overrides:Dict):
    # load model
    model = CytoDLModel()
    model.load_config_from_file(cfg_path)
    # apply overrides
    movie_path = extract_key_from_config(dataset_name)
    overrides['data.dict_meta.path'] = movie_path

    model.override_config(overrides)
    model.print_config()

    # apply model
    model.predict()


if __name__ == '__main__':
    fire.Fire(apply_model)