import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cellsmap.util import manifest_io
from src.endo_pipeline.configs import (
    get_model_manifest,
    load_model_config,
    load_reference_dataset_configs,
)
from src.endo_pipeline.io import load_dataframe_from_fms

logger = logging.getLogger(__name__)

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(
    model_name="diffae_04_10", num_pcs: int = 8, scale: bool = False, verbose: bool = True
) -> Pipeline:
    """
    Fit PCA model to fixed set of reference datasets.
    Reference datasets are flagged in the data configs,
    and the list of datasets to use for PCA is obtained
    by calling the `get_reference_datasets` function.

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
    # load model config to get avaiable manifest names
    model_config = load_model_config(model_name)
    # load data configs to get reference datasets
    reference_datasets = load_reference_dataset_configs()
    # list of model manifests
    model_manifests = []
    for dataset in reference_datasets:
        # check if the dataset is in the model config
        try:
            model_manifests.append(get_model_manifest(dataset.name, model_config))
        except FileNotFoundError:
            logger.warning(
                f"Do not have manifests for all PCA reference datasets in model config {model_name}. "
            )
            continue
    if len(model_manifests) == 0:
        logger.error("No reference datasets found for PCA in model config %s", model_name)
        raise FileNotFoundError("Insufficient reference datasets for PCA.")
    if verbose:
        print(
            f"\nReference datasets for PCA: {[manifest.dataset_name for manifest in model_manifests]}\n"
        )
    data_ref = pd.concat(
        [load_dataframe_from_fms(manifest.fmsid) for manifest in model_manifests],
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
    feature_cols = manifest_io.get_feature_cols(data_ref)
    pipe.fit(data_ref[feature_cols].values)  # fit PCA

    if verbose:  # print explained variance ratios
        cumul_exp_var = np.cumsum(pipe["pca"].explained_variance_ratio_)
        print(
            "Cumulative Explained Variance:",
            f"{np.round(cumul_exp_var, 4).tolist()} \n",
        )

    # return the fit PCA pipeline
    return pipe
