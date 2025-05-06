import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cellsmap.util import manifest_io
from cellsmap.util.dataset_io import get_dataset_info, get_reference_datasets

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def get_pca_reference(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Select reference timepoints for fitting PCA based on the dataset annotations

    Inputs:
    - df: pd.DataFrame, containing the metadata for the dataset name and timepoints

    Outputs:
    - df: pd.DataFrame, with an additional column 'pca_ref' indicating whether the timepoint is a reference timepoint
    """
    df["pca_ref"] = False
    dataset_info = get_dataset_info(dataset_name)
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = dataset_info.get("valid_timepoints")
    if valid_timepoints is None:
        df["pca_ref"] = True
    else:
        tps = []
        for start, stop in zip(valid_timepoints["start"], valid_timepoints["stop"]):
            tps.extend(list(range(start, stop + 1)))
        valid_subset = df.frame_number.isin(tps)
        df["pca_ref"] = valid_subset
    return df[df.pca_ref]


def fit_pca(num_pcs: int = 8, scale: bool = False, verbose: bool = True) -> Pipeline:
    """
    Helper function for fitting PCA pipeline.

    Args:
        num_pcs (int, optional): Number of principal components to keep (default: 8, i.e., full PCA)
        scale (bool, optional): Whether to scale the data before fitting PCA (default: False)
        verbose (bool): Whether to print the explained variance ratios (default: True)

    Returns:
        pipe (Pipeline): Fitted PCA pipeline (may include scaling)
    """
    # first, get list of reference datasets to use for PCA
    reference_datasets = get_reference_datasets()
    if verbose:
        print(f"Reference datasets for PCA:")
    data_ref = []
    for name in reference_datasets:
        # if name == '20250402_20X':
        #     continue # skip this dataset, it is not a reference dataset
        if verbose:
            print(f"- {name}")
        df_ = manifest_io.get_diffae_manifest(name)  # get the manifest for the dataset
        df_ = get_pca_reference(
            df_, name
        )  # get df with only the reference timepoints for fitting PCA
        data_ref.append(df_)  # append the reference timepoints to the list

    data_ref = pd.concat(
        data_ref, ignore_index=True
    )  # concatenate the reference timepoints into a single dataframe

    # fit PCA
    if scale:  # scale the data before fitting PCA
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=num_pcs, svd_solver="full")),
            ]
        )
    else:  # don't scale the data before fitting PCA
        pipe = Pipeline([("pca", PCA(n_components=num_pcs, svd_solver="full"))])
    # get the feature columns from the data, these are the columns that start with 'feat_'
    feature_cols = manifest_io.get_feature_cols(data_ref)
    pipe.fit(data_ref[feature_cols].values)  # fit PCA

    if verbose:  # print explained variance ratios
        print(
            f'Cumulative Explained Variance: {np.round(np.cumsum(pipe["pca"].explained_variance_ratio_),4)}'
        )

    # return the fit PCA pipeline
    return pipe
