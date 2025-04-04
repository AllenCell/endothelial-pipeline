from cellsmap.util.dataset_io import get_dataset_info, get_available_datasets
import platform
import pandas as pd
import os

try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import fms, FileLevelMetadataKeys
except ImportError:
    fms = None


def replace_base_url(file_path: str) -> str:
    """
    Replace the base URL 'production.files.allencell.org' with '/allen/programs/allencell/data/proj0/' in the given file path.
    
    Parameters:
    file_path (str): The original file path.
    
    Returns:
    str: The modified file path.
    """
    base_url = "production.files.allencell.org"
    new_base_path = "/allen/programs/allencell/data/proj0/"
    
    if base_url in file_path:
        modified_path = file_path.replace(base_url, new_base_path)
        return modified_path
    else:
        raise ValueError(f"The base URL '{base_url}' was not found in the provided file path.")


def get_valid_path(fpath) -> str:
    """
    Converts a FMS path to one that can be read cross-platform
    """
    if platform.system() == "Windows":
        fpath = "/" + fpath
    return fpath


def read_file_to_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a file into a pandas dataframe
    """
    if path.endswith("csv"):
        df = pd.read_csv(path)
        return df
    elif path.endswith("parquet"):
        df = pd.read_parquet(path)
        return df
    elif path.endswith("tsv"):
        df = pd.read_csv(path, sep="\t")
        return df
    else:
        raise ValueError(f"Unknown format {path.split('.')[-1]}")


def get_dataframe_by_fmsid(fmsid: str) -> pd.DataFrame:
    if fms is not None and os.path.exists("/allen/aics"):
        annotations = {FileLevelMetadataKeys.FILE_ID.value: fmsid}
        record = list(fms.find(annotations=annotations))[0]
        file_path = replace_base_url(record.path)
        path = get_valid_path(file_path)
    else:
        print("aicsfiles not installed or not on AICS intranet")
        # in the future this else statement will load from S3

    df = read_file_to_dataframe(path)
    return df


def get_nuclear_manifest(dataset_name: str) -> pd.DataFrame:
    fmsid = get_dataset_info(dataset_name)["nuclear_seg_manifest_fmsid"]
    df = get_dataframe_by_fmsid(fmsid)
    return df


def get_diffae_manifest(dataset_name: str) -> str:
    fmsid = get_dataset_info(dataset_name)["diffae_manifest_fmsid"]
    df = get_dataframe_by_fmsid(fmsid)
    return df

def list_datasets_with_manifest(manifest_name: str) -> list:
    """
    List all dataset names that have a 'nuclear_seg_manifest_fmsid' or 'diffae_manifest_fmsid'.
    """
    all_datasets = (
        get_available_datasets(verbose = False)
    ) 

    dataset_list = []
    for dataset_name in all_datasets:
        dataset_info = get_dataset_info(dataset_name)
        if manifest_name in dataset_info and dataset_info[manifest_name] != "":
            dataset_list.append(dataset_name)
    return dataset_list
