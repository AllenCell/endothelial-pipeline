"""Methods for fitting and working with PCA models."""

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
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.polar_coords import pcs_to_polar_r, pcs_to_polar_theta
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.dynamics_workflows import RESCALE_THETA
from endo_pipeline.settings.literal_types import PatchTypeLiteral
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
        dataframe_manifest_name = (
            f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid_based"
        )

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

    # Merge dataframes for all datasets
    data_ref = pd.concat(dataframe_list, ignore_index=True)

    # check required columns: DIFFAE_FEATURE_COLUMN_NAMES
    check_required_columns_in_dataframe(data_ref, DIFFAE_FEATURE_COLUMN_NAMES)

    # return just the feature columns for PCA input (i.e., no metadata)
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
            var_name=ColumnTemplate.PCA_FEATURE.replace("%d", ""),
            value_name="loading_value",
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df


def project_features_to_pcs(
    df: pd.DataFrame,
    pca: PCA,
    feat_cols: list[str],
    compute_polar: bool = True,
    rescale_theta: bool = RESCALE_THETA,
    flip_pc3_sign: bool = True,
) -> pd.DataFrame:
    """
    Project feature data onto principal component axes of fit PCA model.

    **Variable transformation**

    The feature data in the input DataFrame is projected onto the principal
    component axes defined by the input PCA model. New columns are added to the
    DataFrame for each principal component (e.g., pc_1, pc_2, pc_3, ...).

    Optionally, based on the input ``compute_polar`` flag, polar coordinates (r,
    theta) are computed from the first two principal components and added as new
    columns.

    Also optionally, based on the input ``flip_pc3_sign`` flag, an additional
    column (rho) that is equivalent to pc_3 but with the sign flipped is added.
    This sign flip is done for consistency such that higher rho values
    correspond to higher cell density in the original image crops.

    Parameters
    ----------
    df
        DataFrame of feature data.
    pca
        Fit PCA model.
    feat_cols
        List of feature column names to project. If None, will automatically
        detect latent feature columns in the DataFrame.
    compute_polar
        Whether to compute polar coordinates (r, theta) from the first two PCs.
    rescale_theta
        Whether to rescale the polar angle theta to be in the range [0, pi].
    flip_pc3_sign
        True to add an addtional column with the sign of PC3 flipped for
        consistency, False otherwise.

    Returns
    -------
    :
        DataFrame with added columns for each principal component.
    """
    # check that required columns are present in dataframe
    check_required_columns_in_dataframe(df, feat_cols)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    num_pcs = pca.components_.shape[0]  # number of principal components
    pc_cols = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    # optionally, compute polar coordinates (r, theta) from first two PCs
    if compute_polar:
        if num_pcs < 2:
            logger.error(
                "Cannot compute polar coordinates from PC1 and PC2 because number of PCs [ %s ] < 2",
                num_pcs,
            )
            raise ValueError("At least 2 PCs are required to compute polar coordinates.")
        else:
            polar_radius_and_polar_angle_cols = {
                Column.DiffAEData.POLAR_RADIUS: pcs_to_polar_r(df_[pc_cols[0]], df_[pc_cols[1]]),
                Column.DiffAEData.POLAR_ANGLE: pcs_to_polar_theta(
                    df_[pc_cols[0]], df_[pc_cols[1]], rescale=rescale_theta
                ),
            }
            df_ = df_.assign(**polar_radius_and_polar_angle_cols)
    if flip_pc3_sign:
        if num_pcs >= 3:
            pc3_flipped_col = {Column.DiffAEData.PC3_FLIPPED: -df_[pc_cols[2]]}
            df_ = df_.assign(**pc3_flipped_col)
        else:
            logger.error("Cannot add column for -(PC3) because number of PCs [ %s ] < 3", num_pcs)
            raise ValueError("At least 3 PCs are required to add column for -(PC3).")

    return df_


def add_crop_index(
    df: pd.DataFrame,
    patch_type: PatchTypeLiteral = "grid_based",
) -> pd.DataFrame:
    """
    Add crop index column to DataFrame df. (Crops are currently identified by
        their starting position in x and y.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata
        columns for start_x, start_y, and FOV_ID
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added crop index column
    """
    if patch_type not in ["grid_based", "cell_centered"]:
        logger.error("Patch type must be 'cell_centered' or 'grid_based', got [ %s ]", patch_type)
        raise ValueError("Input patch type must be 'grid_based' or 'cell_centered'")

    if patch_type == "cell_centered" and Column.TRACK_ID in df.columns:
        required_columns = [Column.POSITION, Column.TRACK_ID]
    elif patch_type == "grid_based":
        required_columns = [Column.POSITION, Column.DiffAEData.START_X, Column.DiffAEData.START_Y]

    check_required_columns_in_dataframe(df, required_columns)

    # group by the required columns and assign a unique integer (the crop_index)
    # to each group based on the index of that group
    df[Column.CROP_INDEX] = df.groupby(required_columns, as_index=False).ngroup().astype(int)

    return df
