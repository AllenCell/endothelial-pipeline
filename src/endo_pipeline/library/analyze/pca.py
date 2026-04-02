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
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


def build_pca_input_dataframe(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
) -> pd.DataFrame:
    """
    Build input dataframe for fitting PCA model using given dataset collection.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.

    Returns
    -------
    :
        Input dataframe for fitting PCA.
    """

    # Get dataframe manifest name if not provided based on default model manifest
    if dataframe_manifest_name is None:
        dataframe_manifest_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"

    # Load dataframe manifest
    manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Get datasets in collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)

    # Load and filter out annotated timepoints (if requested) for each dataset
    dataframe_list = []
    for dataset_name in dataset_names:
        location = get_dataframe_location_for_dataset(manifest, dataset_name)
        dataframe = load_dataframe(location)
        # filter out annotate timepoints and positions except for timepoints
        # annotate at "not steady state"
        annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
        timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)
        dataframe_filtered = filter_dataframe_by_annotations(
            dataframe,
            load_dataset_config(dataset_name),
            timepoint_annotations=timepoint_annotations,
        )
        dataframe_list.append(dataframe_filtered)

    # Merge dataframes for all datasets and return just the feature columns for
    # PCA input
    data_ref = pd.concat(dataframe_list, ignore_index=True)
    return data_ref[DIFFAE_FEATURE_COLUMN_NAMES]


def fit_pca(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
    num_pcs: int = NUM_LATENT_FEATURES,
) -> PCA:
    """
    Fit PCA model using given datasets in given dataset collection.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """

    # Build PCA input dataframe
    pca_input_dataframe = build_pca_input_dataframe(
        dataset_collection_name, dataframe_manifest_name
    )

    # Fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")
    pca.fit(pca_input_dataframe.values)

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
            id_vars="index",
            var_name=Column.DiffAEData.PCA_FEATURE_PREFIX,
            value_name="loading_value",
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df
