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
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    check_required_columns_in_dataframe,
    filter_dataframe_by_annotations,
    filter_dataframe_by_track_length,
    pcs_to_polar_r,
    pcs_to_polar_theta,
)
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.dynamics_workflows import METADATA_COLUMNS_TO_KEEP, RESCALE_THETA
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
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
    crop_pattern: Literal["grid", "tracked"] = "grid",
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
    if crop_pattern not in ["grid", "tracked"]:
        logger.error("Crop pattern must be 'tracked' or 'grid', got [ %s ]", crop_pattern)
        raise ValueError("Input crop_pattern must be 'grid' or 'tracked'")

    if crop_pattern == "tracked" and Column.TRACK_ID in df.columns:
        required_columns = [Column.POSITION, Column.TRACK_ID]
    elif crop_pattern == "grid":
        required_columns = [Column.POSITION, Column.DiffAEData.START_X, Column.DiffAEData.START_Y]

    check_required_columns_in_dataframe(df, required_columns)

    # group by the required columns and assign a unique integer (the crop_index)
    # to each group based on the index of that group
    df[Column.CROP_INDEX] = df.groupby(required_columns, as_index=False).ngroup().astype(int)

    return df


def get_dataframe_for_dynamics_workflows(
    dataset_name: str,
    manifest: DataframeManifest,
    columns_to_keep: list[str] | None = None,
    pca: PCA | None = None,
    filter_by_annotations: bool = True,
    include_cell_piling: bool = True,
    include_not_steady_state: bool = True,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    compute_polar: bool = True,
    rescale_theta: bool = True,
    flip_pc3_sign: bool = True,
    minimum_track_length: int | None = None,
    segmentation_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
) -> pd.DataFrame:
    """
    Load DiffAE dataframe data projected onto given PC axes for downstream
    analysis in the stochastic dynamics workflow. Adds crop index column to
    DataFrame, and projects feature data onto PC axes.

    **Column selection and memory optimization**

    The input DataFrame is filtered to keep only necessary columns when
    initially loaded to save memory. At a minimum, the metadata columns defined
    in METADATA_COLUMNS_TO_KEEP and the latent feature columns needed for PCA
    projection are kept. Additional columns can be specified to keep via the
    input ``columns_to_keep``.

    In the case that the input ``pca`` is not None, the feature data is
    projected onto the PC axes defined by the PCA model, and the original latent
    feature columns are dropped to save memory.

    **Dataframe filtering**

    The input ``filter_by_annotations`` flag determines whether to filter out
    annotated timepoints and positions from the dataframe. If filtering is
    applied, the input flags ``include_cell_piling`` and
    ``include_not_steady_state`` determine whether to include or exclude
    timepoints annotated as "cell piling" and "not steady state", respectively.

    If ``minimum_track_length`` is specified and the crop pattern is 'tracked',
    tracks below the minimum track length will be filtered out.

    Parameters
    ----------
    dataset_name
        Name of dataset
    manifest
        Dataframe manifest for loading model features.
    columns_to_keep
        List of additional column names to keep in the output DataFrame.
    pca
        PCA model to fit to feature data. If None, do not project feature data.
    filter_by_annotations
        Whether to filter out annotated timepoints and positions from the
        dataframe.
    include_cell_piling
        True keep timepoints annotated as "cell_piling", False to remove them.
    include_not_steady_state
        True to keep timepoints annotated as "not_steady_state", False to remove
        them.
    crop_pattern
        Crop pattern used to generate the feature dataframe. Either 'grid' or
        'tracked'.
    compute_polar
        Whether to compute polar coordinates (r, theta) from the first two PCs.
    rescale_theta
        Whether to rescale the polar angle theta to be in the range [0, pi].
    flip_pc3_sign
        True to add an additional column with the sign of PC3 flipped for
        consistency, False otherwise.
    minimum_track_length
        If crop_pattern is 'tracked', minimum track length (in number of
        timepoints) of tracks to keep in the dataframe. If None, do not filter
        by track length.

    Returns
    -------
    :
        DataFrame with specified feature columns (PC projected or not),
        specified metadata columns, and crop index column for downstream
        analysis in various "dynamics workflows".
    """

    location = get_dataframe_location_for_dataset(manifest, dataset_name)
    df = load_dataframe(location, delay=True)
    feat_cols = DIFFAE_FEATURE_COLUMN_NAMES

    # start with default metadata columns to keep
    # temporarily drop the "crop_index" column while workflows that use this
    # method are being refactored
    columns_to_keep_ = [
        column for column in METADATA_COLUMNS_TO_KEEP[crop_pattern] if column != "crop_index"
    ]
    if columns_to_keep is not None:
        columns_to_keep_.extend(columns_to_keep)  # add any additional specified columns to keep
    columns_to_keep_.extend(feat_cols)  # also keep feature columns for PCA projection
    columns_to_keep_ = list(set(columns_to_keep_))  # remove duplicates, if any

    # keep only necessary columns to save memory
    df_ = df[columns_to_keep_].compute()

    # the crop indices have to be added before any filtering
    # so that they are consistently assigned across datasets
    # for the grid crop pattern, which is critical for the
    # grid-based TFE workflow to run correctly
    df_ = add_crop_index(df_, crop_pattern)

    # filter out annotated timepoints, including or excluding
    # "cell piling" and "not steady state" annotations as specified
    if filter_by_annotations:
        annotations_to_ignore = []
        if include_cell_piling:
            annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
        if include_not_steady_state:
            annotations_to_ignore.append(TimepointAnnotation.NOT_STEADY_STATE)
        timepoint_annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=annotations_to_ignore
        )
        df_filtered = filter_dataframe_by_annotations(
            df_,
            load_dataset_config(dataset_name),
            timepoint_annotations=timepoint_annotations,
        )
    else:
        df_filtered = df_

    if crop_pattern == "tracked":
        # some additional filtering with the "is_included" column from the
        # segmentation features dataframe is needed to remove some of the
        # incorrect segmentations that are present in the tracked crops data
        seg_feat_manifest = load_dataframe_manifest(segmentation_feature_manifest_name)
        seg_feat_loc = get_dataframe_location_for_dataset(seg_feat_manifest, dataset_name)
        df_segmentations_delayed = load_dataframe(seg_feat_loc, delay=True)
        cols_to_compute = [
            Column.DATASET,
            Column.POSITION,
            Column.TIMEPOINT,
            Column.TRACK_ID,
            Column.TRACK_LENGTH,
            Column.SegDataFilters.IS_INCLUDED,
        ]
        df_segmentations = df_segmentations_delayed[cols_to_compute].compute()
        # NOTE the 2 lines below are temporary until we update how we store position
        # and change the column names to be consistent across dataframes
        df_segmentations[Column.POSITION] = df_segmentations[Column.POSITION].transform(
            lambda pos: f"P{pos}"
        )
        original_df_length = len(df_filtered)
        df_filtered = df_filtered.merge(
            df_segmentations,
            on=[
                Column.DATASET,
                Column.POSITION,
                Column.TIMEPOINT,
                Column.TRACK_ID,
            ],
            how="left",
            validate="one_to_one",
        )
        if original_df_length != len(df_filtered):
            raise ValueError(
                f"Length of df_diffae_dynamics changed after merging with segmentation features dataframe. "
                f"Original length: {original_df_length}, new length: {len(df_filtered)}. "
                f"Check the merge keys and the dataframes to ensure that the merge is correct."
            )
        df_filtered = df_filtered[df_filtered[Column.SegDataFilters.IS_INCLUDED]]

    if minimum_track_length is not None:
        df_filtered = filter_dataframe_by_track_length(
            df_filtered, Column.TRACK_LENGTH, minimum_track_length
        )

    # add dataset duration description column
    dataset_config = load_dataset_config(dataset_name)
    df_filtered[Column.DURATION] = dataset_config.duration

    if pca is None:
        # do not project feature data onto PCA axes
        return df_filtered

    else:
        # project feature data onto PC axes
        df_with_pcs = project_features_to_pcs(
            df_filtered,
            pca,
            feat_cols=feat_cols,
            compute_polar=compute_polar,
            rescale_theta=rescale_theta,
            flip_pc3_sign=flip_pc3_sign,
        )
        df_drop_original_feats = df_with_pcs.drop(
            columns=feat_cols
        )  # drop original feature columns to save memory
        return df_drop_original_feats
