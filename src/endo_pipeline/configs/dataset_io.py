import logging
from os import scandir
from pathlib import Path

import pandas as pd
import yaml
from deprecated import deprecated  # type:ignore[import-untyped]

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass
import re
from collections.abc import Callable
from typing import Any

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
