import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.endo_pipeline.configs import get_pca_reference_model_manifests, load_model_config
from src.endo_pipeline.io import load_dataframe_from_fms

from .diffae_manifest_utils import get_feature_column_names

logger = logging.getLogger(__name__)

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(
    model_name: str = "diffae_04_10", num_pcs: int = 8, scale: bool = False, verbose: bool = True
) -> Pipeline:
    """
    Fit PCA model to fixed set of reference datasets.
    Reference datasets are flagged in the data configs,
    and the list of datasets to use for PCA is obtained
    by calling the `get_reference_datasets` function.

    Args:
        model_name (str): Name of the model to use for PCA.
            This is used to load the model config and get
            the reference datasets for PCA.
            Default is "diffae_04_10".
        num_pcs (int, optional): Number of principal components
            to keep (default: 8, i.e., full PCA)
        scale (bool, optional): Whether to scale the data before
            fitting PCA (default: False)
        verbose (bool): Whether to print the explained variance
            ratios (default: True)

    Returns:
        pipe (Pipeline): Fitted PCA pipeline (may include scaling)
    """
    # load model config to get avaiable manifest names
    model_config = load_model_config(model_name)
    model_manifest_list = get_pca_reference_model_manifests(model_config)
    if verbose:
        print(
            "\nReference datasets for PCA:",
            f"{[model_manifest.dataset_name for model_manifest in model_manifest_list]}\n",
        )
    data_ref = pd.concat(
        [load_dataframe_from_fms(model_manifest.fmsid) for model_manifest in model_manifest_list],
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
    feature_cols = get_feature_column_names(data_ref)
    pipe.fit(data_ref[feature_cols].values)  # fit PCA

    if verbose:  # print explained variance ratios
        cumul_exp_var = np.cumsum(pipe["pca"].explained_variance_ratio_)
        print(
            "Cumulative Explained Variance:",
            f"{np.round(cumul_exp_var, 4).tolist()} \n",
        )

    # return the fit PCA pipeline
    return pipe
