import logging
import subprocess
from os import scandir
from pathlib import Path

import dask
import dask.array
import dask.dataframe as dd
import pandas as pd
import yaml
from bioio import BioImage
from deprecated import deprecated  # type:ignore[import-untyped]

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import fire

from src.endo_pipeline.configs import get_datasets_in_collection
from src.endo_pipeline.configs.dataset_config_io import (
    get_available_dataset_names,
    load_dataset_config,
)
from src.endo_pipeline.io import load_dataframe_from_fms

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get path to the config directory."""

    return Path(__file__).resolve().parents[0]


def save_to_yaml(object: dict, path: Path, list_representer: bool = True) -> None:
    """Save dictionary object to YAML at given path."""

    if list_representer:
        yaml.SafeDumper.add_representer(
            list,
            lambda dumper, data: dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            ),
        )

    yaml_content = yaml.safe_dump(
        object, default_flow_style=False, sort_keys=False, width=80, indent=2
    )
    path.open("w").write(yaml_content)


def separate_data_config() -> None:
    """Separate combined dataset configs into individual dataset configs."""

    separated_path = get_config_dir() / "datasets"
    combined_path = Path(__file__).resolve().parents[1] / "data_config.yaml"

    combined_data_config = yaml.safe_load(combined_path.open())

    for dataset, contents in combined_data_config.items():
        data_config_path = separated_path / f"{dataset}.yaml"
        single_data_config = {"name": dataset}
        single_data_config.update(contents)
        save_to_yaml(single_data_config, data_config_path)


def separate_model_config() -> None:
    """Separate combined model configs into individual model configs."""

    separated_path = get_config_dir() / "models"
    combined_path = Path(__file__).resolve().parents[1] / "model_config.yaml"

    combined_model_config = yaml.safe_load(combined_path.open())

    for model, contents in combined_model_config.items():
        data_config_path = separated_path / f"{model}.yaml"
        single_model_config = {"name": model}
        single_model_config.update(contents)
        save_to_yaml(single_model_config, data_config_path, False)


def combine_data_config(save: bool = False) -> dict:
    """Combine individual dataset configs into combined dataset keyed by name."""

    separated_path = get_config_dir() / "datasets"
    combined_path = Path(__file__).resolve().parents[1] / "data_config.yaml"

    separate_data_configs = [
        yaml.safe_load(config.open()) for config in sorted(separated_path.glob("*.yaml"))
    ]
    combined_data_config = {config["name"]: config for config in separate_data_configs}

    if save:
        save_to_yaml(combined_data_config, combined_path)

    return combined_data_config


def combine_model_config(save: bool = False) -> dict:
    """Combine individual model configs into combined config keyed by name."""

    separated_path = get_config_dir() / "models"
    combined_path = Path(__file__).resolve().parents[1] / "model_config.yaml"

    separate_data_configs = [
        yaml.safe_load(config.open()) for config in sorted(separated_path.glob("*.yaml"))
    ]
    combined_model_config = {config["name"]: config for config in separate_data_configs}

    if save:
        save_to_yaml(combined_model_config, combined_path)

    return combined_model_config


# model methods


@deprecated(
    """
NOTE: you can ignore this warning when loading "dynamics" configs.

With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing dataset
configs is to use one of the following replacement methods. If you need configs
for all datasets, use:

        configs.load_all_dataset_configs

If you need the config for a single dataset, use:

        configs.load_dataset_config(dataset_name)

If you need only need dataset names, use:

        configs.get_available_dataset_names

With the switch to loading model configs using the ModelConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing model
configs is to use one of the following replacement methods.

If you need the config for a single model, use:

        configs.load_model_config(model_name)

If you need only need dataset names, use:

        configs.get_available_model_names
"""
)
def load_config(config_type: str = "data") -> dict[Any, Any]:
    """Load configuration from YAML file."""
    if config_type not in ["data", "model", "dynamics"]:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')

    # If loading the data config, load combined from all individual dataset
    # configs. This is part of a change to manage datasets with dataclasses.
    if config_type == "data":
        return combine_data_config()

    # If loading the model config, load combined from all individual model
    # configs. This is part of a change to manage models with dataclasses.
    if config_type == "model":
        return combine_model_config()

    config_dir = get_config_dir()
    config_file = config_dir / f"{config_type}_config.yaml"
    with open(config_file) as file:
        config_data = yaml.safe_load(file)
    return config_data


@deprecated(
    """
NOTE: you can ignore this warning when writing "dynamics" configs.

With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for saving updated dataset
configs is to directly adjust values in the config:

        dataset.field = (new value)

The dataset config can then be saved using:

        configs.save_dataset_config(dataset)

With the switch to loading model configs using the ModelConfig dataclass
(instead of as dictionaries) the recommended pattern for saving updated model
configs is to directly adjust values in the config:

        model.field = (new value)

The model config can then be saved using:

        configs.save_model_config(model)
"""
)
def write_config(config: dict[str, dict[str, Any]], config_type: str = "data") -> None:
    """Write configuration to YAML file."""
    if config_type not in ["data", "model", "dynamics"]:
        raise ValueError('Invalid config type. Must be either "data", "model", or "dynamics."')
    parent_folder = Path(__file__).resolve().parent
    config_file = parent_folder.parent / f"{config_type}_config.yaml"

    # Write lists with brackets, not dashes
    def represent_list(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(list, represent_list)

    with open(config_file, "w") as file:
        #                        one key per line            keep ordering    wrap lines
        yaml.dump(config, file, default_flow_style=False, sort_keys=False, width=80, indent=2)

    # If writing the data config, split the combined data config file that was
    # saved above into individual dataset config files (and delete the combined
    # config file).
    if config_type == "data":
        separate_data_config()
        config_file.unlink()

    if config_type == "model":
        separate_model_config()
        config_file.unlink()


@deprecated(
    """
With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for saving updated dataset
configs is to directly adjust values in the config:

        dataset.field = (new value)

The dataset config can then be saved using:

        configs.save_dataset_config(dataset)
"""
)
def update_dataset_config(dataset_name: str, new_config: dict[str, Any]) -> None:
    """
    Update the dataset config file with new values.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to update.
    new_config: dict
        Dictionary with new values to update in the config file.
    """
    cfg = load_config("data")
    cfg[dataset_name].update(new_config)
    write_config(cfg, "data")


# dataset methods
@deprecated(
    """
With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing datasets is:

1. If you need a list of available datasets by name, before selecting specific
   dataset(s) to load, use the following replacement method:

        configs.get_available_dataset_names

   instead of:

        configs.dataset_io.get_available_datasets

   Individual dataset(s) can then be loaded with:

        configs.load_dataset_config(dataset_name)

2. If you want to load all available datasets, use the following method to load
   configs for all available datasets:

        configs.load_all_dataset_configs
"""
)
def get_available_datasets(verbose: bool = True) -> list[str]:
    """Get a list of available datasets from the config file."""
    cfg = load_config("data")
    datasets = list(cfg.keys())
    if verbose:
        print("\n".join(datasets))
    return datasets


@deprecated(
    """
With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing reference
datasets is to instead load the appropriate reference dataset collection.

        configs.get_datasets_in_collection("pca_reference")

which will load the reference dataset objects (not the dataset names). If you
need the names of the reference datasets, access the .name field of the
reference dataset objects returned by the above method.
"""
)
def get_reference_datasets() -> list[str]:
    """Get a list of reference datasets for PCA from the config file."""
    return [
        name
        for name in get_available_datasets(verbose=False)
        if get_dataset_info(name).get("is_reference", False)
    ]


@deprecated(
    """
With the switch to loading dataset configs using the DatasetConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing dataset info
is directly from loaded DatasetConfig objects. These configs can be loaded using
one of the following:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

Fields can then be accessed using dot notation:

        dataset.field

Available fields and descriptions for each field for DatasetConfig objects are
provided in configs.dataset_config.
"""
)
def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """Load specific dataset information from the config file."""
    config = load_config()
    if dataset_name not in config:
        raise ValueError(f"Dataset {dataset_name} not found in config file")
    return config[dataset_name]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.zarr_path
"""
)
def get_zarr_dir(dataset_name: str) -> str:
    """Get the directory path for the zarr files of a given dataset."""
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["zarr_path"]


@deprecated(
    """
This method is deprecated and will be removed. Use the following replacement:

    from src.endo_pipeline.configs import get_available_zarr_files

This method will return a list of Path objects to Zarr files for all positions
in the given dataset config. If you need the name of the Zarr file, use .name on
the returned Path object.
"""
)
def get_zarr_path(
    dataset_name: str,
    zarr_name: str | None | None = None,
) -> dict[str, str]:
    """Get the zarr file paths for a given dataset."""
    data_dir = get_zarr_dir(dataset_name)
    zarr_paths = {}
    if zarr_name:
        filepath = Path(data_dir) / zarr_name
        assert filepath.exists(), f"Zarr file {filepath} does not exist."
        filepath_list = [filepath]
    else:
        filepath_list = list(Path(data_dir).glob("*.zarr"))

    for filepath in filepath_list:
        zarr_paths[filepath.name] = str(filepath)

    return zarr_paths


@deprecated(
    """
This method is deprecated and will be removed. Instead use:

    get_available_channels_for_all_positions(dataset_config)

The recommended pattern is:

    from src.endo_pipeline.configs import load_dataset_config, get_available_channels_for_all_positions

    dataset_config = load_dataset_config(dataset_name)
    channels = get_available_channels_for_all_positions(dataset_config)
"""
)
def get_available_channels(
    dataset_name: str, zarr_name: str | None | None = None
) -> dict[str, list[str]]:
    """Get the available channels for a given dataset."""
    zarr_paths = get_zarr_path(dataset_name, zarr_name)
    channel_names = {}
    for filename, filepath in zarr_paths.items():
        reader = BioImage(filepath)
        channel_names[filename] = reader.channel_names
    return channel_names


@deprecated(
    """
This method is deprecated and will be removed. Instead use:

    get_available_channels_for_position(dataset_config, position)

To recreate the behavior of this method, use:

    from src.endo_pipeline.configs import load_dataset_config, get_available_channels_for_all_positions

    dataset_config = load_dataset_config(dataset_name)
    channels = get_available_channels_for_position(dataset_config, 0)
"""
)
def get_channel_names(dataset_name: str) -> list[str]:
    """
    Retrieve the list of channel names for a specific dataset. The function
    test_channel_names_consistency validates that all positions have the same channels
    within a dataset so we can use the first position to get the channel names.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        list[str]: A list of channel names available in the dataset at the specified position.
    """
    zarr_name = get_zarr_name(dataset_name, position=0)
    zarr_paths = get_zarr_path(dataset_name, zarr_name)
    path_of_interest = zarr_paths[zarr_name]
    reader = BioImage(path_of_interest)
    channel_names = reader.channel_names
    return channel_names


@deprecated(
    """
This method is deprecated and will be removed. Instead use:

    from src.endo_pipeline.configs import get_channel_indices_for_all_positions
    get_channel_indices_for_all_positions(dataset_config, position, channel_names)
"""
)
def get_channel_index(
    dataset_name: str, channel_names: list[str], zarr_name: str | None | None = None
) -> dict[str, list[int | None]]:
    """Get the indices of specified channels in the dataset."""
    zarr_paths = get_zarr_path(dataset_name, zarr_name)
    channel_indices = {}
    for filename in zarr_paths.keys():
        available_channels = get_available_channels(dataset_name, filename)
        # available_channels[filename].update([available_channels.index(channel) if channel in available_channels else None for channel in channel_names])
        channel_indices[filename] = [
            (
                available_channels[filename].index(channel)
                if channel in available_channels[filename]
                else None
            )
            for channel in channel_names
        ]
    return channel_indices


@deprecated(
    """
This method is deprecated and will be removed. Use the following replacement:

    from src.endo_pipeline.configs import get_zarr_file_for_position

This method will a Path to the Zarr file for the given dataset and position. If
you need the name of the Zarr file, use .name on the returned Path object.
"""
)
def get_zarr_name(dataset_name: str, position: int) -> str:
    """
    Get the zarr name for a given dataset and position.
    """
    zarr_paths = get_zarr_path(dataset_name)
    zarr_found_for_position = position in [extract_P(zarr_name) for zarr_name in zarr_paths.keys()]
    assert (
        zarr_found_for_position
    ), f"Zarr file for position {position} not found in dataset {dataset_name}."
    for zarr_name in zarr_paths.keys():
        if position == extract_P(zarr_name):
            break
    return zarr_name


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

Then use this replacement method:

        configs.get_specific_channel_order(dataset)
"""
)
def get_specific_channel_order(dataset_name: str) -> tuple:
    """Get the specific channel order for a given dataset."""
    dataset_info = get_dataset_info(dataset_name)
    gfp_index = dataset_info.get("channel_488_index")
    bf_index = dataset_info.get("brightfield_channel_index")
    index_405 = dataset_info.get("channel_405_index", None)
    index_561 = dataset_info.get("channel_561_index", None)
    index_640 = dataset_info.get("channel_640_index", None)

    return gfp_index, bf_index, index_405, index_561, index_640


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.n_total_positions
"""
)
def get_total_number_of_positions(dataset_name: str) -> int:
    """
    Get the total number of positions in a dataset.

    Number of positions is the product of n_scenes x n_positions_per_scene
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["n_total_positions"]


@deprecated(
    """
This method is deprecated and will be removed. The new pattern for loading Zarr
datasets is:

    from src.endo_pipeline.configs import load_dataset_config, get_zarr_file_for_position
    from src.endo_pipeline.io import load_zarr_as_dask_array

    dataset_config = load_dataset_config(dataset_name)
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    zarr = load_zarr_as_dask_array(zarr_file)

To recreate the behavior of this specific method (loading Zarrs for all positions
of a dataset into a dictionary, use:

    from src.endo_pipeline.configs import load_dataset_config, get_available_zarr_files
    from src.endo_pipeline.io import load_zarr_as_dask_array

    dataset_config = load_dataset_config(dataset_name)
    zarr_files = get_available_zarr_files(dataset_config)
    zarrs = {zarr_file.name: load_zarr_as_dask_array(zarr_file) for zarr_file in zarr_files}
"""
)
def load_dataset(
    dataset_name: str,
    channels: list = ["EGFP", "BF"],
    time_start: int = 0,
    time_end: int = -1,
    level: int = 0,
    zarr_name: str | None = None,
) -> dict[str, dask.array.Array]:
    """Load a dataset as a dictionary of Dask arrays."""
    zarr_paths = get_zarr_path(dataset_name, zarr_name)
    dataset = {}

    for filename, filepath in zarr_paths.items():
        reader = BioImage(filepath)
        available_channels = reader.channel_names
        channels_index = [available_channels.index(c) for c in channels]
        assert (
            level in reader.resolution_levels
        ), f"Invalid resolution level {level}. Available levels are {reader.resolution_levels}"
        reader.set_resolution_level(level)
        if time_end < 0:
            time_end = get_dataset_duration_in_frames(dataset_name) - 1
        img = reader.get_image_dask_data(
            "TCZYX", T=range(time_start, time_end + 1), C=channels_index
        )
        dataset[filename] = img
    return dataset


@deprecated(
    """
This method is deprecated and will be removed. The new pattern for loading Zarr
datasets is:

    from src.endo_pipeline.configs import load_dataset_config, get_zarr_file_for_position
    from src.endo_pipeline.io import load_zarr_as_dask_array

    dataset_config = load_dataset_config(dataset_name)
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    zarr = load_zarr_as_dask_array(zarr_file)
"""
)
def load_dataset_position_as_dask_array(
    dataset_name: str,
    position: int | str,
    channels: list = ["EGFP", "BF"],
    time_start: int = 0,
    time_end: int = -1,
    level: int = 0,
) -> dask.array.Array:
    """
    Load a specific position of a dataset as a Dask array.

    Position can be either an integer or a string.
    If it is a string then it must the name of a zarr file found in
    dataset (e.g. a folder ending with the .ome.zarr extension).
    If it is an integer then it will be used as the index to
    get the zarr file name from the dataset.
    """
    zarr_path_list = get_zarr_path(dataset_name)
    if isinstance(position, int):
        if position >= len(zarr_path_list):
            raise ValueError(
                f"Position {position} is out of range. There are only {len(zarr_path_list)} zarr files in the dataset."
            )
        zarr_name = list(zarr_path_list.keys())[position]
        for zarr_name in zarr_path_list.keys():
            if position == extract_P(zarr_name):
                break
    elif isinstance(position, str):
        if position not in zarr_path_list:
            raise ValueError(f"Zarr file {position} not found in dataset {dataset_name}.")
        zarr_name = position

    img_dict = load_dataset(
        dataset_name, channels, time_start, time_end, level, zarr_name=zarr_name
    )
    img_dask_arr = img_dict[zarr_name]
    return img_dask_arr


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.duration
"""
)
def get_dataset_duration_in_frames(dataset_name: str) -> int:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["duration"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.pixel_size_xy_in_um
"""
)
def get_xy_pixel_size_in_um(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["pixel_size_xy_in_um"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.time_interval_in_minutes
"""
)
def get_time_interval_in_minutes(dataset_name: str) -> float:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["time_interval_in_minutes"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.flow
"""
)
def get_flow_info(dataset_name: str) -> list:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["flow"]


def get_flow_change_frame(dataset_name: str) -> int:
    """
    Get frame number at which flow changes in dataset ds_name.

    Inputs:
    - dataset_name: str, name of dataset to get flow change frame for
        - This string must match the dataset name in data_config.yaml

    Outputs:
    - change_frame: int, frame number at which flow changes in dataset dataset_name
    """
    # load config for dataset from data_config.yaml
    flow_info = get_flow_info(dataset_name)

    # get frame number at which flow changes
    change_frame = flow_info[0][1]

    return change_frame


def get_flow_for_frame(dataset_name: str, frame: int) -> float:
    """
    Retrieve the flow value for a specific frame in a dataset.

    This function searches the flow list for the given dataset and returns the
    flow value corresponding to the specified frame. If the frame is not found
    in the flow list, a ValueError is raised.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to retrieve the flow information from.
    frame : int
        The frame index for which to retrieve the flow value.

    Returns
    -------
    float
        The flow value for the specified frame.
    """
    config = load_dataset_config(dataset_name)
    flow_list = config.flow
    for t_start, t_stop, flow in flow_list:
        if t_start <= frame <= t_stop:
            return flow
    raise ValueError(f"Frame {frame} not found in flow list for dataset '{dataset_name}'.")


def add_flow_to_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add flow in dyn/cm^2 to a DataFrame containing dataset information.
    Currently does not work for datasets with -1 as the timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing dataset information, including 'dataset_name' and 'frame'.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'flow' column containing flow values for each frame.
    """
    # Group by dataset_name and image_index, calculate flow once per group
    flow_mapping = df.groupby(["dataset_name", "image_index"]).apply(
        lambda group: get_flow_for_frame(group.name[0], group.name[1])
    )

    # Map the calculated flow values back to the original DataFrame
    df["sheer_stress"] = df.set_index(["dataset_name", "image_index"]).index.map(flow_mapping)
    return df


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.valid_timepoints
"""
)
def get_valid_timepoints(dataset_name: str) -> dict:
    """
    Get the frames marked for use in DiffAE feature
    analysis workflows for a given dataset.
    These are determined by an experimentalist by eye
    and are added to the dataset config file.
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info.get("valid_timepoints")


def get_dim_map(dim_order: str) -> dict:

    dims = [a for a in dim_order]
    dim_nums = tuple(range(len(dims)))
    dim_map = dict(zip(dims, dim_nums, strict=False))

    return dim_map


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.original_path
"""
)
def get_original_path(dataset_name: str) -> Path:
    """
    Example path format: /{date}/{dataset_name}.dir/{dataset_name_number}.imgdir
    """
    dataset_info = get_dataset_info(dataset_name)
    return Path(dataset_info["original_path"])


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.barcode
"""
)
def get_barcode(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["barcode"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.microscope
"""
)
def get_microscope(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["microscope"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

The field can then be accessed using:

        dataset.fmsid
"""
)
def get_fmsid(dataset_name: str) -> str:
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["fmsid"]


@deprecated(
    """
Use one of the following methods to load the dataset config:

        configs.load_all_dataset_configs
        configs.load_dataset_config(dataset_name)

Then use this replacement method:

        configs.get_nuclear_prediction_path(dataset, channel, nuc_seg_type)
"""
)
def get_nuclear_prediction_path(
    dataset_name: str, position: int, nuc_seg_type: str = "nuclear_label_free_seg_path"
) -> str:
    """
    Get the file path for the nuclear prediction data for a specific dataset and position.

    Args:
        dataset_name (str): The name of the dataset.
        position (int): The position index within the dataset.
        nuc_seg_type (str, optional): The type of nuclear segmentation path to retrieve.
            ie. "nuclear_label_free_seg_path", "nuclear_stain_seg_path"

    Returns:
        str: The file path to the nuclear prediction data for the specified dataset and position.
    """
    dataset_info = get_dataset_info(dataset_name)
    base_path = dataset_info[nuc_seg_type]
    position_path = f"{base_path}/P{position}/"
    return position_path


def load_nuclei_prediction(
    dataset_name: str,
    position: int,
    T: int,
    dim_order: str = "ZYX",
    nuc_seg_type: str = "nuclear_label_free_seg_path",
) -> dask.array.Array:
    """
    Load the nuclei prediction data as a Dask array for a given dataset, position, and timepoint.

    Args:
        dataset_name (str): The name of the dataset.
        position (int): The position index within the dataset.
        T (int): The timepoint index to load.
        dim_order (str, optional): The dimension order of the data (e.g., "ZYX").
            Defaults to "ZYX".
        nuc_seg_type (str, optional): The type of nuclear segmentation path to retrieve.
            ie. "nuclear_label_free_seg_path", "nuclear_stain_seg_path"

    Returns:
        dask.array.Array: A Dask array containing the nuclei prediction data for the specified
        dataset, position, and timepoint. If the file is not found, an empty Dask array is returned.
    """
    nuc_dir = Path(get_nuclear_prediction_path(dataset_name, position, nuc_seg_type))
    nuc_path_dict = {extract_T(fp.stem): fp for fp in nuc_dir.glob("*.ome.tif*")}
    nuc_path = nuc_path_dict[T]

    if nuc_path.exists():
        # Load the nuclei prediction as a Dask array
        nuc_dask_arr = BioImage(nuc_path).get_image_dask_data(dim_order, T=0)
        return nuc_dask_arr
    else:
        print(
            f"Nuclei prediction file not found for T={T} in {nuc_dir}, returning empty dask array."
        )
        return dask.array.empty(shape=[0] * len(dim_order))


def get_cdh5_classic_segmentation_path(
    dataset_name: str,
    position: int,
    T: int | None = None,
    missing_file_exception: Literal["raise", "warn"] = "warn",
) -> Path | None:
    """
    Get the path to the CDH5 classic segmentation file for a given dataset, position, and timepoint.
    If T is None, it returns the directory for the position.
    If T is provided, it returns the specific file for that timepoint.
    If the file is not found, it raises a FileNotFoundError or logs a warning based on the
    `missing_file_exception` parameter.
    """

    cdh5_seg_dir = load_dataset_config(dataset_name).cdh5_seg_path

    if cdh5_seg_dir is None:
        logger.warn(f"No Cdh5 segmentations for {dataset_name}.")
        return None
    else:
        position_path = Path(cdh5_seg_dir) / f"P{position}"

    if T is None:
        return position_path
    else:
        cdh5_seg_path_dict = {
            extract_T(fp.stem): fp
            for fp in position_path.glob("*.ome.tif*")
            if extract_T(fp.name) == T
        }
        cdh5_seg_path = cdh5_seg_path_dict.get(T, None)
        if cdh5_seg_path is not None:
            return cdh5_seg_path

        match missing_file_exception:
            case "raise":
                logger.error(f"CDH5 segmentation for T={T} not found in {position_path}.")
                raise FileNotFoundError()
            case "warn":
                logger.warn(
                    f"CDH5 segmentation for T={T} not found in {position_path}. Skipping..."
                )
                return None


def load_cdh5_classic_segmentation(
    dataset_name: str,
    position: int,
    T: int,
    dim_order: str = "ZYX",
) -> dask.array.Array:
    """
    Load the CDH5 classic segmentation for a given dataset, position, and timepoint.
    """
    cdh5_seg_path = get_cdh5_classic_segmentation_path(dataset_name, position, T)
    if cdh5_seg_path is not None and cdh5_seg_path.exists():
        # Load the CDH5 classic segmentation as a Dask array
        cdh5_dask_arr = BioImage(cdh5_seg_path).get_image_dask_data(dim_order, T=0)
        return cdh5_dask_arr
    else:
        print(
            f"CDH5 classic segmentation file not found for T={T} in {cdh5_seg_path}, returning empty dask array."
        )
        return dask.array.empty(shape=[0] * len(dim_order))


@deprecated(
    """
    This function was replaced by the function get_measured_segmentation_table
    in the same location as this one.
    """
)
def get_tracking_data_paths(
    dataset_name: str,
    position: int,
) -> Path:
    # NOTE the tracking paths should probably be added to some
    # sort of config file at some point, but in the interest of
    # going fast they are hardcoded here for now
    base_path = Path(
        "//allen/aics/endothelial/morphological_features/analysis/cdh5_classic_seg_tracking"
    )
    base_path = base_path / f"{dataset_name}/P{position}"
    data_path = base_path / f"{dataset_name}_P{position}_tracking.tsv"
    return data_path


@deprecated(
    """
    This function was replaced by the function get_measured_segmentation_table
    in the same location as this one.
    """
)
def get_tracking_data_raws(
    dataset_name_list: list,
    position: int | None = None,
    as_dask: bool = True,
) -> pd.DataFrame:
    # get all the filepaths and check that none of the requested
    # datasets-position-kind combinations are missing data paths
    # first before opening them
    table_reader = dd if as_dask else pd
    tracking_data_list = []
    for dataset_name in dataset_name_list:
        position_list = (
            range(get_total_number_of_positions(dataset_name)) if position == None else [position]
        )
        for pos in position_list:
            data_path = Path(get_tracking_data_paths(dataset_name, pos))
            if not data_path.exists():
                print(f"No tracking data found for {dataset_name} P{pos}. Skipping...")
                continue
            else:
                # open the data tables
                tracking_data = table_reader.read_csv(data_path, sep="\t")
                # the tracking data by default does not have the
                # dataset name or the position, so add those in
                tracking_data["dataset_name"] = dataset_name
                tracking_data["position"] = pos
                # also include the path to the table that this
                # part of the dataframe was loaded from
                tracking_data["source_tracking_table_path"] = data_path.as_posix()
                tracking_data_list.append(tracking_data)
    # concatenate the dataframes into a single dataframe and return it
    if tracking_data_list:
        tracking_dataframe = table_reader.concat(tracking_data_list, axis=0, ignore_index=True)
    else:  # create an empty dataframe
        tracking_dataframe = table_reader.DataFrame.from_dict({})
    return tracking_dataframe


@deprecated(
    """
    This function was replaced by the function get_measured_segmentation_table
    in the same location as this one.
    """
)
def get_tracking_data_filtered(dataset_name_list: list, as_dask: bool = False) -> pd.DataFrame:
    """
    NOTE: Cannot use only dask here because if it is called in the
    same script that a multiprocessing workflow that later uses
    dask delayed reading (such as opening files with bioio) then
    the script will hang when trying to execute the later dask
    delayed .compute() function. This is the case even if this
    function get_tracking_data_filtered is called outside of
    multiprocessing.
    """
    table_reader = dd if as_dask else pd
    base_path = Path("//allen/aics/endothelial/morphological_features/analysis/track_filtering")
    tracking_data_list = []
    for dataset_name in dataset_name_list:
        data_path = base_path / f"{dataset_name}_filtered_tracking_data.tsv"
        if data_path.exists():
            # open the data tables
            tracking_data = table_reader.read_csv(data_path, sep="\t")
            # include path to file that this data was loaded from
            tracking_data["source_filtered_tracking_table_path"] = data_path.as_posix()
            tracking_data_list.append(tracking_data)
        else:
            print(f"No filtered tracking data found for {dataset_name}. Skipping...")
            continue
    # concatenate the dataframes into a single dataframe and return it
    tracking_dataframe = table_reader.concat(tracking_data_list, axis=0, ignore_index=True)
    return tracking_dataframe


@deprecated(
    """
    This function was replaced by the function get_measured_segmentation_table
    in the same location as this one.
    """
)
def get_measurement_data_paths(
    dataset_name: str, kind: Literal["alignments", "segmentation_properties"]
) -> Path:
    # NOTE the tracking paths should probably be added to some
    # sort of config file at some point, but in the interest of
    # going fast they are hardcoded here for now
    base_path = Path(
        "//allen/aics/endothelial/morphological_features/analysis/cdh5_nodes_and_edges"
    )
    base_path = base_path / dataset_name
    data_path = base_path / f"{dataset_name}_{kind}.csv"
    return data_path


@deprecated(
    """
    This function was replaced by the function get_measured_segmentation_table
    in the same location as this one.
    """
)
def get_measurement_data_raws(
    dataset_name_list: list,
    kind: Literal["alignments", "segmentation_properties"],
    as_dask: bool = True,
) -> pd.DataFrame:
    table_reader = dd if as_dask else pd
    measurement_data_list = []
    # get all the filepaths and check that none of the requested
    # datasets-position-kind combinations are missing data paths
    # first before opening them
    for dataset_name in dataset_name_list:
        data_path = Path(get_measurement_data_paths(dataset_name, kind))
        if not data_path.exists():
            print(f"No {kind} tracking data found for {dataset_name}. Skipping...")
            continue
        else:
            measurement_data = table_reader.read_csv(data_path)
            measurement_data["source_measurement_table_path"] = data_path.as_posix()
            measurement_data_list.append(measurement_data)
    # open the files and concatenate them into a single dataframe
    if measurement_data_list:
        measurement_dataframe = table_reader.concat(
            measurement_data_list, axis=0, ignore_index=True
        )
    else:  # create an empty dataframe
        measurement_dataframe = table_reader.DataFrame.from_dict({})
    return measurement_dataframe


def get_measured_segmentation_table(
    dataset_name_list: list,
    kind: Literal["cdh5_segmentations", "nuclei_labelfree", "cdh5_tracking"],
) -> pd.DataFrame:
    """
    Loads one of the available kinds of segmentation features tables
    for a given dataset.
    Different kinds of segmentation features tables include:
    - cdh5_segmentations: properties for segmentations based on cdh5
        - includes cell segmentation centroids, orientations, number of neighbors,
            neighbor information, elongation, and other properties
        - does not contain dynamics-dependent features such as velocities
            (those can be computed from this dataset with
            src.endo_pipeline.workflows.make_seg_feats_manifest.calculate_derived_data_dynamics_dependent
    - nuclei_labelfree: properties for segmentations based on nuclei label-free
        - primarily predicted nuclei centroids
    - cdh5_tracking: properties for segmentations based on cdh5 tracking
        - primarily contains tracking IDs mapped to segmentations label IDs
    """
    match kind:
        case "cdh5_segmentations":
            fmsid_field = "cdh5_classic_seg_manifest_fmsid"
        case "nuclei_labelfree":
            fmsid_field = "nuclei_label_free_seg_manifest_fmsid"
        case "cdh5_tracking":
            fmsid_field = "cdh5_classic_seg_tracking_manifest_fmsid"
        case _:
            raise ValueError(
                f"Invalid kind {kind}. Must be one of 'cdh5_segmentations', 'nuclei_labelfree', or 'cdh5_tracking'."
            )
    measured_data_list = []
    for dataset_name in dataset_name_list:
        fmsid = getattr(load_dataset_config(dataset_name), fmsid_field)
        if fmsid is not None:
            measured_data = load_dataframe_from_fms(fmsid)
        else:
            logger.info(f"No {kind} data found for {dataset_name}. Skipping...")
            continue
        # add the fmsid name to the dataframe
        measured_data[f"fmsid_{kind}_measurements_table"] = fmsid
        # add the dataframe to the list of datasets
        measured_data_list.append(measured_data)
    # concatenate the dataframes into a single dataframe and return it
    if measured_data_list:
        measured_dataframe = pd.concat(measured_data_list, axis=0, ignore_index=True)
    else:  # create an empty dataframe
        measured_dataframe = pd.DataFrame.from_dict({})

    return measured_dataframe


def get_live_segmentation_features_manifest(
    dataset_name_list: list,
) -> pd.DataFrame:
    """
    Get the segmentation features manifest for a given dataset.
    The manifest is a TSV file that contains the measurements
    from the tracked segmentations of a dataset.
    These datasets are raw / unfiltered.
    """

    seg_feat_data_list = []
    for dataset_name in dataset_name_list:
        # get the fmsid of the live data segmentation
        # features manifest for the dataset
        fmsid = load_dataset_config(dataset_name).live_merged_seg_features_manifest_fmsid
        # load the manifest associated with this fmsid as a dataframe
        if fmsid is not None:
            seg_feat_data = load_dataframe_from_fms(fmsid)
        else:
            logger.info(f"No segmentation features manifest found for {dataset_name}. Skipping...")
            continue
        # add the fmsid name to the dataframe
        seg_feat_data["live_merged_seg_features_manifest_fmsid"] = fmsid
        # add the dataframe to the list of datasets
        seg_feat_data_list.append(seg_feat_data)
    # concatenate the dataframes into a single dataframe and return it
    seg_feat_dataframe = pd.concat(seg_feat_data_list, axis=0, ignore_index=True)

    return seg_feat_dataframe


# fire argparsing methods
def fire_parse_list_from_CLI(fire_str_or_list_like_input: Sequence) -> list[str]:
    if isinstance(fire_str_or_list_like_input, str):
        list_of_strings = [fire_str_or_list_like_input]
    elif isinstance(fire_str_or_list_like_input, Sequence):
        list_of_strings = list(fire_str_or_list_like_input)
    else:
        raise ValueError(
            f"Invalid input {fire_str_or_list_like_input}. Must be a string or list of strings."
        )
    return list_of_strings


def fire_parse_generate_dataset_name_list(
    fire_dataset_name_input: Sequence | None,
) -> list[str]:
    """
    Parse a list of dataset names from the command line.
    The input can be a string or a list of strings.
    If it is a string, it will be turned into a list of strings.
    If it is a list of strings, it will be returned as is.

    To enter a list of datasets to analyze, use the following format:
    '\"20241016_20X\",\"20241120_20X\"'
    """
    if fire_dataset_name_input is None:
        dataset_name_list = get_datasets_in_collection("pca_reference")
    else:
        dataset_name_list = fire_parse_list_from_CLI(fire_dataset_name_input)

    # check that the dataset names are valid
    available_datasets = get_available_dataset_names()
    for dataset_name in dataset_name_list:
        assert (
            dataset_name in available_datasets
        ), f"Invalid dataset name {dataset_name}. Must be a string or list of strings that are found in the available datasets {available_datasets}."

    return dataset_name_list


@deprecated(
    """
With the switch to loading model configs using the ModelConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing models is:

1. If you need a list of available models by name, before selecting specific
   dataset(s) to load, use the following replacement method:

        configs.get_available_model_names

   instead of:

        configs.dataset_io.get_available_models

   Individual models(s) can then be loaded with:

        configs.load_model_config(model_name)

2. If you want to load all available models, use the following method to load
   configs for all available models:

        configs.load_all_model_configs
"""
)
def get_available_models() -> list[str]:
    model_info = load_config("model")
    model_names = list(model_info.keys())
    for name in model_names:
        print(name)
    return model_names


@deprecated(
    """
With the switch to loading model configs using the ModelConfig dataclass
(instead of as dictionaries) the recommended pattern for accessing model info is
directly from loaded ModelConfig objects. These configs can be loaded using
one of the following:

        configs.load_all_model_configs
        configs.load_model_config(model_name)

Fields can then be accessed using dot notation:

        model.field

Available fields and descriptions for each field for ModelConfig objects are
provided in configs.model_config.
"""
)
def get_model_info(model_name: str) -> dict[str, Any]:
    config = load_config("model")
    if model_name not in config:
        raise ValueError(f"Model {model_name} not found in config file")
    return config[model_name]


# Other miscellaneous methods
def ipython_cli_flexecute(
    function: Callable[..., Any],
    return_results: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute function with arguments and keyword arguments in
    an IPython shell or via command line interface.
    """
    # The following try-except statement will run 'main' without
    # fire.Fire if an interactive shell is in use,
    # otherwise it will run 'main' through fire.Fire so that
    # arguments can easily be passed to 'main' through
    # some non-interactive shell like bash
    try:
        # the following line will return a string if an interactive shell is in use,
        # otherwise raises NameError since get_ipython is not imported from IPython
        # or returns None if get_ipython is present but script is being executed
        # from a non-interactive shell
        if get_ipython().__class__.__name__ != "NoneType":
            print(f"Using interactive shell {get_ipython().__class__.__name__}.")
            results = function(*args, **kwargs)
        else:
            raise NameError
    except NameError:
        print("Using non-interactive shell.")
        results = fire.Fire(function)

    return results if return_results else None


def extract_T(
    fp_as_string: str | Path,
    int_only: bool = True,
    use_last_match: bool = True,
    default_if_not_found: int | str = "",
) -> str | int:
    """
    Extract the timepoint value from a string or Path.name.
    Searches for the pattern "T[0-9]+" to find the timepoint.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    fp_as_string: str or Path
        A string or Path.name to get the timepoint from.
    int_only: bool
        Whether to return just the timepoint as an integer or
        an entire string (i.e. 10 vs 'T010')
        Default is True.
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        timepoint values were found in the string).
        If False then the first match will be used.
        E.g. image_name_T1_etc_T57.tif can return either T1 or T57, but
        will return 57 by default. Ideally the timepoint in fp_as_string
        would be unambiguous.
        Default is True.

    Returns
    -------
    t: int or str
        The timepoint represented as an integer if int_only is True, otherwise
        the timepoint represented as a string including the T before.
    """

    if isinstance(fp_as_string, Path):
        fp_as_string = str(fp_as_string)

    index = -1 if use_last_match else 0
    t = re.findall("T[0-9]+", fp_as_string)
    t_value = int(t[index].split("T")[-1]) if t else default_if_not_found
    if not t:
        print("""No 'T[0-9]+' found in filename. Using T == default_if_not_found.""")

    return t_value if int_only else f"T{t_value}"


@deprecated(
    """
    Use extract_position_from_filepath in library.process.image_filepath_utils instead
    """
)
def extract_P(
    fp_as_string: str | Path,
    int_only: bool = True,
    use_last_match: bool = True,
    default_if_not_found: int | str = "",
) -> str | int:
    """
    Extract the position value from a string or Path.name.
    Searches for the pattern "P[0-9]+" to find the position.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    fp_as_string: str or Path
        A string or Path.name to get the position from.
    int_only: bool
        Whether to return just the position as an integer or
        an entire string (i.e. 10 vs 'P10')
        Default is True (i.e. just an integer).
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        position values were found in the string).
        If False then the first match will be used.
        E.g. image_name_P1_P3_etc_T57.tif can return either P1 or P3, but
        will return 3 by default. Ideally the position in fp_as_string
        would be unambiguous.
        Default is True.
    default_if_not_found: int or str
        The value to return if no position is found in the string.

    Returns
    -------
    P: int or str
        The position represented as an integer if int_only is True, otherwise
        the position represented as a string including the P before.
    """

    if isinstance(fp_as_string, Path):
        fp_as_string = str(fp_as_string)

    index = -1 if use_last_match else 0
    p = re.findall("P[0-9]+", fp_as_string)
    position_value = int(p[index].split("P")[-1]) if p else default_if_not_found
    if not p:
        print("""No 'P[0-9]+' found in filename. Using P == default_if_not_found.""")

    return position_value if int_only else f"P{position_value}"


@deprecated(
    """
This method is deprecated and will be removed. To provide git versioning
information when uploading files to FMS, use the `include_git_info=True` flag
(`True` by default) in `src.endo_pipeline.io.build_fms_annotations`.
"""
)
def get_git_versioning_info() -> dict[str, str]:
    """
    Return versioning info about the script, including the branch
    name, commit hash, uncommitted changes, and timestamp of when
    the script was run.
    """
    # get some versioning info about when this script was run and
    # what version of the script was used to produce the output
    # to save alongside the output
    # the branch name:
    git_branch_name = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )
    # the current commit hash:
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    # if there were any uncommitted changes when this script was run:
    git_uncommitted_changes = (
        subprocess.check_output(["git", "diff", "HEAD", "--name-only"]).decode("ascii").strip()
        or "None"
    )
    # the timestamp that this script was run:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %X")

    git_branch_info = {
        "timestamp": str(timestamp),
        "git_branch_name": str(git_branch_name),
        "git_commit_hash": str(git_commit_hash),
        "git_uncommitted_changes": str(git_uncommitted_changes),
    }
    return git_branch_info


@deprecated(
    """
This method is deprecated and will be removed. Git versioning info should only
be saved with files uploaded to FMS. Use the `include_git_info=True` flag
(`True` by default) in `src.endo_pipeline.io.build_fms_annotations` to include
git versioning info when uploading to FMS.
"""
)
def save_git_versioning_info(
    out_dir: Path,
    filename_prefix: str,
    verbose: bool = True,
) -> None:
    """
    Save git versioning info to a .txt file in the specified output directory.
    The filename will be prepended with the provided filename_prefix.
    output_dir should be a path that exists already, it will not be created.
    """
    git_info = get_git_versioning_info()
    output_path = out_dir / f"{filename_prefix}_git_versioning_info.txt"
    with output_path.open("w") as git_versioning_file:
        for key, value in git_info.items():
            git_versioning_file.write(f"{key}: {value}\n")
    print(f"Git versioning info saved to {output_path}.") if verbose else None
    return None


def concatenate_and_save_feature_tables(
    out_dir: Path,
    dataset_name: str,
    out_file_suffix: str = "",
    input_filename_contains: str = "",
    file_extension: str = ".csv",
    sort_by_T: bool = True,
    check_saved_dataframe: bool = True,
    remove_initial_files_and_folders: bool = False,
) -> None:
    """
    Concatenate the nuclei feature tables for all positions and
    timepoints for a given dataset in an out_dir and then saves
    the concatenated table to the output directory.
    The expected file structure in out_dir is:
    out_dir/dataset_name/position/*filename_contains*.file_extension.
    """
    out_subdir = out_dir / dataset_name
    feats_dfs = []
    sep = "\t" if file_extension == ".tsv" else ","

    file_extension = f".{file_extension}" if not file_extension.startswith(".") else file_extension
    if input_filename_contains and not input_filename_contains.endswith("*"):
        input_filename_contains = f"{input_filename_contains}*"
    feats_filepaths = list(out_subdir.glob(f"**/*{input_filename_contains}{file_extension}"))
    if sort_by_T:
        feats_filepaths = sorted(feats_filepaths, key=lambda fp: extract_T(fp.stem))
    feats_dfs = [pd.read_csv(fp, sep=sep) for fp in feats_filepaths]

    # define the output path for the concatenated dataframe
    if out_file_suffix:
        out_file_suffix = (
            f"_{out_file_suffix}" if not out_file_suffix.startswith("_") else f"{out_file_suffix}"
        )
    concatenated_df_out_path = out_dir / f"{dataset_name}{out_file_suffix}{file_extension}"

    if feats_dfs:
        concatenated_df = pd.concat(feats_dfs, ignore_index=True)
        concatenated_df.to_csv(concatenated_df_out_path, sep=sep, index=False)
    else:
        print(f"No feature tables found for {dataset_name}.")

    if check_saved_dataframe:
        # check that the concatenated dataframe at least has the same shape
        # and column names as a proxy for checking if it was saved correctly
        saved_df = pd.read_csv(concatenated_df_out_path, sep=sep)
        same_shape = saved_df.shape == concatenated_df.shape
        same_column_names = all(saved_df.columns == concatenated_df.columns)
        if not (same_shape and same_column_names):
            raise ValueError(
                f"Saved dataframe {concatenated_df_out_path} \
                    does not match the concatenated dataframe."
            )
        print(f"Concatenated dataframe saved to {concatenated_df_out_path}.")

    if remove_initial_files_and_folders:
        # remove files that match input_filename_contains
        for fp in feats_filepaths:
            fp.unlink()
    dirs_to_remove = list(out_subdir.glob("**/"))
    # remove the empty directory now that old tables are deleted
    # (note this must be done in reverse order because a folder with
    # subfolders does not count as empty and therfore raises an error)
    for dir_path in dirs_to_remove[::-1]:
        # NOTE that rmdir only removes empty directories
        # and will raise an error if it is not empty. If
        # a directory is not empty then we will skip it
        if not any(list(scandir(dir_path))):
            dir_path.rmdir()
            print(f"Removed empty directory {dir_path}.")
        else:
            print(f"Directory {dir_path} is not empty, skipping removal.")
            continue
