import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_model_manifest,
    get_pca_reference_model_manifests,
    load_model_config,
)
from src.endo_pipeline.io import load_dataframe_from_fms

from .diffae_manifest_utils import get_feature_column_names

logger = logging.getLogger(__name__)

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(
    dataset_collection_name: str = "pca_reference",
    model_name: str = "diffae_04_10",
    num_pcs: int = 8,
) -> PCA:
    """
    Fit PCA model to fixed set of reference datasets, as defined in the given
    dataset collection name.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
        This is used to load the model manifests that contain the reference datasets.
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
    if dataset_collection_name == "pca_reference":
        # use default function
        model_manifest_list = get_pca_reference_model_manifests(model_config)
    else:
        # load model manifests for the given dataset collection
        dataset_names = get_datasets_in_collection(dataset_collection_name)
        model_manifest_list = [
            get_model_manifest(dataset_name, model_name) for dataset_name in dataset_names
        ]

    logger.info(
        "\nDatasets being used to fit PCA: \n %s",
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

    # log info about explained variance ratio
    logger.info(
        "Explained variance ratios: %s",
        np.round(pca.explained_variance_ratio_, 4).tolist(),
    )

    cumul_exp_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        "Cumulative explained variance: %s",
        np.round(cumul_exp_var, 4).tolist(),
    )

    # return the fit PCA pipeline
    return pca


def get_pca_loadings(pca: PCA, scaled: bool = False, magnitude: bool = False) -> np.ndarray:
    """
    Get the PCA loading matrix, which contains the contribution of each feature to each principal component.
    The loading matrix is the transpose of the PCA components matrix.

    Parameters
    ----------
    pca : PCA
        The fitted PCA object.
    kind : str, optional
        Whether to return the loading matrix unscaled or scaled by the square root of the explained variance.
        Options are "unscaled" or "scaled". Default is "unscaled".
    magnitude : bool, optional
        Whether to return the absolute values of the loadings. Default is False.

    Returns
    -------
    loading_matrix : np.ndarray
        The PCA loading matrix. Has shape (n_features, n_components).
    """

    if scaled:  # create scaled loading matrix
        loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
    else:  # create unscaled loading matrix
        loading_matrix = pca.components_.T

    if magnitude:
        loading_matrix = np.abs(loading_matrix)

    return loading_matrix


def get_pca_loadings_as_df(
    pca: PCA,
    scaled: bool = False,
    magnitude: bool = False,
    df_format: Literal["long", "wide"] = "long",
) -> pd.DataFrame:
    """
    Get the PCA loading matrix as a DataFrame.
    This is a wrapper around `get_pca_loadings` that formats the output as a DataFrame.

    Parameters
    ----------
    pca : PCA
        The fitted PCA object.
    kind : str, optional
        Whether to return the loading matrix unscaled or scaled by the square root of the explained variance.
        Options are "unscaled" or "scaled". Default is "unscaled".
    magnitude : bool, optional
        Whether to return the absolute values of the loadings. Default is False.
    df_format : str, optional
        The format of the DataFrame to return. Options are "long" or "wide".
        If "long", the DataFrame will have columns for 'feature', 'PC', and 'loading_value'.
        If "wide", the DataFrame will have one column per PC, indexed by feature.
        Default is "long".

    Returns
    -------
    loading_matrix_df : pd.DataFrame
        The PCA loading matrix as a DataFrame.

    """
    loading_matrix = get_pca_loadings(pca, scaled, magnitude)

    num_features, num_pcs = loading_matrix.shape
    feat_col_names = [f"feat_{i}" for i in range(num_features)]
    pc_col_names = [f"pc{i+1}" for i in range(num_pcs)]

    loading_matrix_df = pd.DataFrame(loading_matrix, columns=pc_col_names, index=feat_col_names)
    if df_format == "long":
        loading_matrix_df = loading_matrix_df.reset_index().melt(
            id_vars="index", var_name="pc", value_name="loading_value"
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df
