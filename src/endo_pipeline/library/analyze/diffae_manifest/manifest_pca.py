import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cellsmap.util import manifest_io_temp
from src.endo_pipeline.configs.dataset_config import load_reference_datasets

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(num_pcs: int = 8, scale: bool = False, verbose: bool = True) -> Pipeline:
    """
    Fit PCA model to fixed set of reference datasets.
    Reference datasets are flagged in the data configs,
    and the configs of the datasets to use for PCA is obtained
    by calling the `load_reference_datasets` function.

    Args:
        num_pcs (int, optional): Number of principal components
            to keep (default: 8, i.e., full PCA)
        scale (bool, optional): Whether to scale the data before
            fitting PCA (default: False)
        verbose (bool): Whether to print the explained variance
            ratios (default: True)

    Returns:
        pipe (Pipeline): Fitted PCA pipeline (may include scaling)
    """
    # first, load configs for reference datasets for PCA
    reference_datasets = load_reference_datasets()
    if verbose:
        print(f"\nReference datasets for PCA: {reference_datasets}")
    data_ref = pd.concat(
        [
            manifest_io_temp.get_diffae_manifest(config, filter_to_valid=True)
            for config in reference_datasets
        ],
        ignore_index=True,
    )

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

    # get the feature columns from the data,
    # these are the columns that start with 'feat_'
    feature_cols = manifest_io_temp.get_feature_cols(data_ref)
    pipe.fit(data_ref[feature_cols].values)  # fit PCA

    if verbose:  # print explained variance ratios
        cumul_exp_var = np.cumsum(pipe["pca"].explained_variance_ratio_)
        print(
            "Cumulative Explained Variance:",
            f"{np.round(cumul_exp_var, 4).tolist()} \n",
        )

    # return the fit PCA pipeline
    return pipe
