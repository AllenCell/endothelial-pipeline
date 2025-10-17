import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import DIFFAE_FEATURE_COLUMN_NAMES, DIFFAE_PC_COLUMN_NAMES, ColumnName

from .dataframe_preprocessing import filter_dataframe_by_annotations

logger = logging.getLogger(__name__)

# this is to suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def fit_pca(
    dataset_collection_name: str = "pca_reference",
    dataframe_manifest_name: str = "diffae_04_10",
    filter_dataframe: bool = True,
    include_cell_piling: bool = False,
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
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    filter_dataframe
        Whether to remove annotated timepoints and positions from the dataframes before fitting PCA.
    include_cell_piling
        True to include cell piling timepoints in the data used to fit PCA, False to exclude them.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """
    # Load dataframe manifest for given model
    manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Get dataframe locations for manifest for all datasets in collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)
    locations = [
        get_dataframe_location_for_dataset(manifest, dataset_name) for dataset_name in dataset_names
    ]
    logger.info("Datasets being used to fit PCA: [ %s ]", ", ".join(dataset_names))

    # Load all dataframes, filter out annotated timepoints, and concatenate.
    # Filtering does or doesn't remove cell piling timepoints based on
    # the input include_cell_piling.
    dataframe_list = []
    for location, dataset_name in zip(locations, dataset_names, strict=True):
        dataframe = load_dataframe(location)
        if filter_dataframe:
            annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
            if include_cell_piling:
                annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
            timepoint_annotations = get_subset_of_timepoint_annotations(
                annotations_to_ignore=annotations_to_ignore
            )
            dataframe_filtered = filter_dataframe_by_annotations(
                dataframe,
                load_dataset_config(dataset_name),
                timepoint_annotations=timepoint_annotations,
            )
        else:
            dataframe_filtered = dataframe
        dataframe_list.append(dataframe_filtered)
    data_ref = pd.concat(dataframe_list, ignore_index=True)

    # fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")

    # get the feature columns from the data,
    # these are the columns that start with 'feat_'
    pca.fit(data_ref[DIFFAE_FEATURE_COLUMN_NAMES].values)  # fit PCA

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


def get_pca_loadings(
    pca: PCA, scaled: bool = False, magnitude: bool = False, squared_norm: bool = False
) -> np.ndarray:
    """
    Get the PCA loading matrix, which contains the contribution of each feature to each
    principal component.
    The loading matrix is the transpose of the PCA components matrix.

    Parameters
    ----------
    pca : PCA
        The fitted PCA object.
    scaled : bool, optional
        Whether to return the loading matrix unscaled or scaled by the square root of the
        explained variance.
        Default is False (i.e. return unscaled loadings).
    magnitude : bool, optional
        Whether to return the absolute values of the loadings. Default is False.
    squared_norm : bool, optional
        Whether to return the squared norm of the loadings. Default is False.
        If True, the loading matrix will be squared element-wise.

    Returns
    -------
    loading_matrix : np.ndarray
        The PCA loading matrix. Has shape (n_features, n_components).
    """

    loading_matrix = pca.components_.T  # create unscaled loading matrix

    if scaled:  # create scaled loading matrix
        loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)

    if magnitude:
        loading_matrix = np.abs(loading_matrix)

    if squared_norm:
        loading_matrix = loading_matrix**2

    return loading_matrix


def get_pca_loadings_as_df(
    pca: PCA,
    scaled: bool = False,
    magnitude: bool = False,
    squared_norm: bool = False,
    df_format: Literal["long", "wide"] = "long",
) -> pd.DataFrame:
    """
    Get the PCA loading matrix as a DataFrame.

    This is a wrapper around `get_pca_loadings` that formats the output as a DataFrame.

    **DataFrame format options**

    The DataFrame can be returned in either "long" or "wide" format. The "long" format
    has three columns: 'feature', 'PC', and 'loading_value'. The "wide" format has one
    column per PC, indexed by feature.

    Parameters
    ----------
    pca
        The fit PCA object.
    scaled
        Whether to return the scaled loading matrix or unscaled.
    magnitude
        Whether to return the absolute values of the loadings
    squared_norm
        Whether to return the squared norm of the loadings.
    df_format
        The format of the DataFrame to return, either "long" or "wide".

    Returns
    -------
    :
        The PCA loading matrix as a DataFrame.

    """
    loading_matrix = get_pca_loadings(pca, scaled, magnitude, squared_norm)

    num_features, num_pcs = loading_matrix.shape
    feat_col_names = DIFFAE_FEATURE_COLUMN_NAMES[:num_features]
    pc_col_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]

    loading_matrix_df = pd.DataFrame(loading_matrix, columns=pc_col_names, index=feat_col_names)
    if df_format == "long":
        loading_matrix_df = loading_matrix_df.reset_index().melt(
            id_vars="index", var_name=ColumnName.PCA_FEATURE_PREFIX, value_name="loading_value"
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df
