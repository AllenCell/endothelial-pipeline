import pandas as pd

from src.endo_pipeline.configs import (
    DatasetConfig,
    get_model_manifest,
    load_single_dataset_config,
    load_single_model_config,
)
from src.endo_pipeline.library.analyze.fms_utils import get_dataframe_by_fmsid


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Extract columns corresponding to DiffAE model
    features from dataframe (loaded DiffAE manifest).
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split("_")[1]))
    return feat_cols


def get_valid_subset(df: pd.DataFrame, config: DatasetConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Select timepoints from a dataframe annotated as valid
    if annotation is present, otherwise use all timepoints.

    Inputs:
    - df: pd.DataFrame, containing the metadata
        for the dataset name and timepoints
    - config: DatasetConfig, configuration object
        for the dataset

    Outputs:
    - df: pd.DataFrame, subset of the input dataframe
        containing only the valid timepoints
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = config.valid_timepoints
    if valid_timepoints is None:
        if verbose:
            print(f"Using all timepoints from dataset {config.name} for PCA")
        df["valid"] = True
    else:
        if verbose:
            print(f"Valid timepoints for dataset {config.name}: ")
        tps = []
        for start, stop in zip(valid_timepoints.start, valid_timepoints.stop, strict=True):
            tps.extend(list(range(start, stop + 1)))
            if verbose:
                print(f"   - {start} to {stop}")
        valid_subset = df.frame_number.isin(tps)
        df["valid"] = valid_subset
    return df[df.valid]


def load_model_manifest_dataframe(
    dataset_config: DatasetConfig, model_name: str = "diffae_04_10", filter_to_valid: bool = False
) -> pd.DataFrame:
    """Load manifest data for a given dataset and model."""
    model_config = load_single_model_config(model_name)
    fmsid = get_model_manifest(dataset_config.name, model_config).fmsid
    df = get_dataframe_by_fmsid(fmsid)
    if filter_to_valid:
        df = get_valid_subset(df, dataset_config, verbose=False)
    return df
