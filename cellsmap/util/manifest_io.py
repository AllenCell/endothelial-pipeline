from cellsmap.util.dataset_io import get_dataset_info
import platform
import pandas as pd
import os

try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import fms, FileLevelMetadataKeys
except ImportError:
    fms = None





def get_valid_path(record) -> str:
    """
    Converts a FMS path to one that can be read cross-platform
    """
    recordpath = record.path
    if platform.system() == "Windows":
        recordpath = "/" + recordpath
    return recordpath


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
        record = fms.find(annotations=annotations)
        path = get_valid_path(record)
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
