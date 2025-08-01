import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.endo_pipeline.configs import get_pca_reference_model_manifests, load_model_config
from src.endo_pipeline.io import load_dataframe_from_fms

from .diffae_manifest_utils import get_feature_column_names

logger = logging.getLogger(__name__)

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(model_name: str = "diffae_04_10", num_pcs: int = 8) -> PCA:
    """
    Fit PCA model to fixed set of reference datasets, as defined in the
    'pca_reference' dataset collection.

    Parameters
    ----------
    model_name
        Name of the DiffAE model whose features to fit PCA on.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """
    # load model config to get avaiable manifest names
    model_config = load_model_config(model_name)
    model_manifest_list = get_pca_reference_model_manifests(model_config)
    logger.info(
        "\nReference datasets for PCA: \n %s",
        [model_manifest.dataset_name for model_manifest in model_manifest_list],
    )
    data_ref = pd.concat(
        [load_dataframe_from_fms(model_manifest.fmsid) for model_manifest in model_manifest_list],
        ignore_index=True,
    )

    # fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")

    # get the feature columns from the data,
    # these are the columns that start with 'feat_'
    feature_cols = get_feature_column_names(data_ref)
    pca.fit(data_ref[feature_cols].values)  # fit PCA

    cumul_exp_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        "Cumulative Explained Variance: %s",
        np.round(cumul_exp_var, 4).tolist(),
    )

    # return the fit PCA pipeline
    return pca
