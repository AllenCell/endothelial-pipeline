import pickle

import numpy as np

from cellsmap.util.dataset_io import load_config


def load_dynamics_config(config_name: str = "default") -> dict:
    """
    Load specific config from config file for running stochastic dynamics related workflows.
    Config file is `dynamics_config.yaml` at the head of the project.

    Inputs:
    - config_name: str, name of the config to load, default is 'default'.

    Outputs:
    - config: dict, config dictionary.
    """
    # Load config file
    configs = load_config("dynamics")

    # Find the config with the given name
    for config in configs:
        if config["name"] == config_name:
            return config

    # Raise error if config not found
    raise ValueError(f"Config {config_name} not found in dynamics_config.yaml")


def save_train_test(train_test_dict: dict, savedir: str) -> None:
    """
    Save train test data to file in savedir, using `numpy.savez` function.

    Inputs:
    - train_test_dict: dict, dictionary containing train and test data (numpy arrays).
    - savedir: str, directory to save the file.

    Outputs:
    - None, save the file to savedir.
    """
    np.savez(savedir + "train_test_data", **train_test_dict)


def load_train_test(file_path: str) -> dict:
    """
    Load train test data from file_path.

    Inputs:
    - file_path: str, path to the file containing train test data.

    Outputs:
    - train_test_dict: dict, dictionary containing train and test data (numpy arrays).
    """
    return dict(np.load(file_path, allow_pickle=True))


def save_model(model_dict: dict, savedir: str) -> None:
    """
    Save fit SDE model to file in savedir.

    Inputs:
    - model_dict: dict, dictionary containing fit drift and diffusion functions.
    - savedir: str, directory to save the file.

    Outputs:
    - None, save the file to savedir.
    """
    with open(savedir + "drift_diffusion_model.pkl", "wb") as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_path: str) -> dict:
    """
    Load fit SDE model from file_path.

    Inputs:
    - file_path: str, path to the file containing fit drift and diffusion functions.

    Outputs:
    - model_dict: dict, dictionary containing fit drift and diffusion functions.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
