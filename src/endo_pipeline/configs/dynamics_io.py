from endo_pipeline.configs.dataset_io import load_config


def load_dynamics_config(config_name: str = "default") -> dict:
    """
    Load specific config from config file for running stochastic
    dynamics related workflows. Config file is
    `dynamics_config.yaml` at the head of the project.

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
