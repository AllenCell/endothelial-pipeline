import logging
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
from collections.abc import Callable
from typing import Any

from endo_pipeline.settings import DIMENSION_ORDER

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

    from endo_pipeline.configs import get_available_zarr_files

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

    from endo_pipeline.configs import load_dataset_config, get_available_channels_for_all_positions

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

    from endo_pipeline.configs import load_dataset_config, get_available_channels_for_all_positions

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

    from endo_pipeline.configs import get_channel_indices_for_all_positions
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

    from endo_pipeline.configs import get_zarr_file_for_position

This method will a Path to the Zarr file for the given dataset and position. If
you need the name of the Zarr file, use .name on the returned Path object.
"""
)
def get_zarr_name(dataset_name: str, position: int) -> str:
    """
    Get the zarr name for a given dataset and position.
    """
    zarr_paths = get_zarr_path(dataset_name)
    zarr_found_for_position = position in [extract_p(zarr_name) for zarr_name in zarr_paths.keys()]
    assert (
        zarr_found_for_position
    ), f"Zarr file for position {position} not found in dataset {dataset_name}."
    for zarr_name in zarr_paths.keys():
        if position == extract_p(zarr_name):
            break
    return zarr_name


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

    from endo_pipeline.configs import load_dataset_config, get_zarr_file_for_position
    from endo_pipeline.io import load_image_from_path

    dataset_config = load_dataset_config(dataset_name)
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    zarr = load_image_from_path(zarr_file)

To recreate the behavior of this specific method (loading Zarrs for all positions
of a dataset into a dictionary, use:

    from endo_pipeline.configs import load_dataset_config, get_available_zarr_files
    from endo_pipeline.io import load_image_from_path

    dataset_config = load_dataset_config(dataset_name)
    zarr_files = get_available_zarr_files(dataset_config)
    zarrs = {zarr_file.name: load_image_from_path(zarr_file) for zarr_file in zarr_files}
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
            DIMENSION_ORDER, T=range(time_start, time_end + 1), C=channels_index
        )
        dataset[filename] = img
    return dataset


@deprecated(
    """
This method is deprecated and will be removed. The new pattern for loading Zarr
datasets is:

    from endo_pipeline.configs import load_dataset_config, get_zarr_file_for_position
    from endo_pipeline.io import load_image_from_path

    dataset_config = load_dataset_config(dataset_name)
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    zarr = load_image_from_path(zarr_file)
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
            if position == extract_p(zarr_name):
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


# Other miscellaneous methods
def ipython_cli_flexecute(
    function: Callable[..., Any],
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
            function(*args, **kwargs)
        else:
            raise NameError
    except NameError:
        print("Using non-interactive shell.")
        from endo_pipeline.__main__ import workflow_cli

        workflow_cli(function)


def extract_t(
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


def extract_p(
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

    file_extension = f".{file_extension}" if not file_extension.startswith(".") else file_extension
    if input_filename_contains and not input_filename_contains.endswith("*"):
        input_filename_contains = f"{input_filename_contains}*"
    feats_filepaths = list(out_subdir.glob(f"**/*{input_filename_contains}{file_extension}"))
    if sort_by_T:
        feats_filepaths = sorted(feats_filepaths, key=lambda fp: extract_t(fp.stem))

    if file_extension == ".tsv":
        sep = "\t"
        table_reader = lambda fp: pd.read_csv(fp, sep=sep)
        table_writer = lambda df, fp: df.to_csv(fp, sep=sep, index=False)
    elif file_extension == ".csv":
        sep = ","
        table_reader = lambda fp: pd.read_csv(fp, sep=sep)
        table_writer = lambda df, fp: df.to_csv(fp, sep=sep, index=False)
    elif file_extension == ".parquet":
        table_reader = lambda fp: pd.read_parquet(fp)
        table_writer = lambda df, fp: df.to_parquet(fp, index=False)
    else:
        raise ValueError(
            f"Invalid file extension {file_extension}. Must be .csv, .tsv., or .parquet."
        )
    feats_dfs = [table_reader(fp) for fp in feats_filepaths]

    # define the output path for the concatenated dataframe
    if out_file_suffix:
        out_file_suffix = (
            f"_{out_file_suffix}" if not out_file_suffix.startswith("_") else f"{out_file_suffix}"
        )
    concatenated_df_out_path = out_dir / f"{dataset_name}{out_file_suffix}{file_extension}"

    if feats_dfs:
        concatenated_df = pd.concat(feats_dfs, ignore_index=True)
        table_writer(concatenated_df, concatenated_df_out_path)
    else:
        print(f"No feature tables found for {dataset_name}.")

    if check_saved_dataframe:
        # check that the concatenated dataframe at least has the same shape
        # and column names as a proxy for checking if it was saved correctly
        saved_df = table_reader(concatenated_df_out_path)
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
