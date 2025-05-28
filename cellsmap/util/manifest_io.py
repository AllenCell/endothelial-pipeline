import os
import pickle
import platform

import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util import dataset_io

try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import FileLevelMetadataKeys, fms
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
        raise ValueError(
            f"The base URL '{base_url}' was not found in the provided file path."
        )


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
        return pd.read_csv(path)
    elif path.endswith("parquet"):
        return pd.read_parquet(path)
    elif path.endswith("tsv"):
        return pd.read_csv(path, sep="\t")
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
    fmsid = dataset_io.get_dataset_info(dataset_name)["nuclear_seg_manifest_fmsid"]
    df = get_dataframe_by_fmsid(fmsid)
    return df


def get_valid_subset(
    df: pd.DataFrame, dataset_name: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Select timepoints from a dataframe annotated as valid if annotation is present, otherwise use all teimpoints

    Inputs:
    - df: pd.DataFrame, containing the metadata for the dataset name and timepoints
    - dataset_name: str, name of the dataset to get valid timepoints for

    Outputs:
    - df: pd.DataFrame, subset of the input dataframe containing only the valid timepoints
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = dataset_io.get_valid_timepoints(dataset_name)
    if valid_timepoints is None:
        if verbose:
            print(f"Using all timepoints from dataset {dataset_name} for PCA")
        df["valid"] = True
    else:
        if verbose:
            print(f"Valid timepoints for dataset {dataset_name}: ")
        tps = []
        for start, stop in zip(valid_timepoints["start"], valid_timepoints["stop"]):
            tps.extend(list(range(start, stop + 1)))
            if verbose:
                print(f"   - {start} to {stop}")
        valid_subset = df.frame_number.isin(tps)
        df["valid"] = valid_subset
    return df[df.valid]


def get_diffae_manifest(
    dataset_name: str, filter_to_valid: bool = False
) -> pd.DataFrame:
    fmsid = dataset_io.get_dataset_info(dataset_name)["diffae_manifest_fmsid"]
    if fmsid == "" or fmsid is None:
        print(f"No DiffAE manifest found for dataset {dataset_name}")
        return None
    df = get_dataframe_by_fmsid(fmsid)
    if filter_to_valid:
        df = get_valid_subset(df, dataset_name)
    return df


def get_track_diffae_manifest(dataset_name: str) -> pd.DataFrame:
    fmsid = dataset_io.get_dataset_info(dataset_name).get(
        "diffae_tracking_integration_fmsid", None
    )
    if fmsid:
        return get_dataframe_by_fmsid(fmsid)
    else:
        print(f"No DiffAE manifest found for dataset {dataset_name}")
        return None


def get_cell_mean_features_manifest(dataset_name: str) -> pd.DataFrame:
    fmsid = dataset_io.get_dataset_info(dataset_name).get("cell_mean_features", None)
    if fmsid:
        return get_dataframe_by_fmsid(fmsid)
    else:
        print(f"No cell mean features manifest found for dataset {dataset_name}")
        return None


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Extract columns corresponding to DiffAE model features from dataframe (loaded DiffAE manifest).
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split("_")[1]))
    return feat_cols


def list_datasets_with_manifest(
    manifest_name: str,
    verbose: bool = False,
    timelapse_only: bool = False,
) -> list:
    """
    List all dataset names that have a 'nuclear_seg_manifest_fmsid' or 'diffae_manifest_fmsid'.
    """
    all_datasets = dataset_io.get_available_datasets(verbose=False)

    if verbose:
        manifest_type = (
            "nuclear segmentation"
            if manifest_name == "nuclear_seg_manifest_fmsid"
            else "DiffAE"
        )
        if timelapse_only:
            print(f"Available timelapse datasets with {manifest_type} manifest data: ")
        else:
            print(f"Available datasets with {manifest_type} manifest data: ")
    dataset_list = []
    for dataset_name in all_datasets:
        dataset_info = dataset_io.get_dataset_info(dataset_name)
        # get time_interval_in_minutes - any dataset
        # that is fixed or is a 20X/40X pair has default
        # time_interval_in_minutes of -1.0, so we skip
        time_interval_in_minutes = dataset_info.get("time_interval_in_minutes", -1.0)
        if timelapse_only and time_interval_in_minutes < 0:
            continue
        if manifest_name in dataset_info and dataset_info[manifest_name] != "":
            dataset_list.append(dataset_name)
            if verbose:
                print(f" - {dataset_name}")
    return dataset_list


def save_pca_model(pca: Pipeline, savedir: str) -> None:
    """
    Save fit PCA model to file using pickle.

    Inputs:
    - pca: Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - can include any preprocessing steps before PCA, e.g., scaling
    - savedir: str, directory to save PCA model to

    Outputs:
    - None, saves PCA model to file
    """
    if not savedir.endswith("/"):
        savedir += "/"
    with open(savedir + "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)


def load_pca_model(savedir: str) -> Pipeline:
    """
    Load PCA model from file.

    Inputs:
    - savedir: str, directory to load PCA model from

    Outputs:
    - pca: Pipeline, fit PCA model loaded from file
    """
    if not savedir.endswith("/"):
        savedir += "/"
    with open(savedir + "pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    return pca
