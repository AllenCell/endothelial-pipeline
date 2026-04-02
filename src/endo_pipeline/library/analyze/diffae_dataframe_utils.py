import logging
import re
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_frame_after_flow_change,
    get_subset_of_timepoint_annotations,
    get_unannotated_positions,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.pca import build_pca_input_dataframe
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
from endo_pipeline.settings.dynamics_workflows import (
    METADATA_COLUMNS_TO_KEEP,
    PERIOD_THETA_RESCALED,
    RESCALE_THETA,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def check_required_columns_in_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Check that required columns are present in a given dataframe.

    Parameters
    ----------
    df
        DataFrame to check.
    required_columns
        List of required column names to check for.
    """

    for col in required_columns:
        if col not in df.columns:
            logger.error("DataFrame must contain column [ %s ]", col)
            raise ValueError(f"DataFrame must contain column [ {col} ]")


def pcs_to_polar_r(pc1_values: np.ndarray, pc2_values: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate r.

    The polar coordinate r is given by the formula:
        r = sqrt(pc1^2 + pc2^2)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.

    Returns
    -------
    :
        Polar coordinate r values.
    """
    return np.sqrt(pc1_values**2 + pc2_values**2)


def pcs_to_polar_theta(
    pc1_values: np.ndarray,
    pc2_values: np.ndarray,
    rescale: bool = True,
) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate theta.

    The polar coordinate theta is given by the formula:
        theta = arctan2(pc2, pc1)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.
    rescale
        Whether to rescale the angle to be in the range [0, pi] instead of [-pi, pi].

    Returns
    -------
    :
        Polar coordinate theta values.
    """
    # angle in range [-pi, pi]
    theta = np.arctan2(pc2_values, pc1_values)

    if rescale:
        # rescale angle to range [0, pi]
        # by adding pi and dividing by 2
        # (values now have period pi instead of 2pi)
        theta = (theta + np.pi) / 2

    return theta


def polar_to_pcs(
    theta_values: np.ndarray, r_values: np.ndarray, is_theta_rescaled: bool = RESCALE_THETA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates (theta, r) back to Cartesian coordinates (pc1, pc2).

    The conversion from polar to Cartesian coordinates is given by the formulas:
        pc1 = r * cos(theta)
        pc2 = r * sin(theta)

    If the input theta values are rescaled to be in the range [0, pi], they will be
    unrescaled back to the range [-pi, pi] before conversion.

    Parameters
    ----------
    theta_values
        Polar coordinate theta values.
    r_values
        Polar coordinate r values.
    is_theta_rescaled
        Whether the input theta values were rescaled to be in the range [0, pi].
    """

    if is_theta_rescaled:
        # unrescale theta back to range [-pi, pi]
        theta_values = (theta_values * 2) - np.pi

    pc1_values = r_values * np.cos(theta_values)
    pc2_values = r_values * np.sin(theta_values)

    return pc1_values, pc2_values


@overload
def rewrap_polar_angle(unwrapped_angle: float, original_range: tuple[float, float]) -> float: ...


@overload
def rewrap_polar_angle(
    unwrapped_angle: np.ndarray, original_range: tuple[float, float]
) -> np.ndarray: ...


def rewrap_polar_angle(
    unwrapped_angle: float | np.ndarray, original_range: tuple[float, float]
) -> float | np.ndarray:
    """
    Rewrap unwrapped polar angle value to be within original range.

    Unwrapped angles computed, e.g., using numpy.unwrap can extend beyond the original
    periodic range of polar angle values. This function rewraps the unwrapped angle back
    to be within the original range.

    Example:
        original_range = (0, pi)
        unwrapped_angle = pi + 0.5
        rewrapped_angle = 0.5

    Parameters
    ----------
    unwrapped_angle
        Unwrapped polar angle value.
    original_range
        Original range of polar angle values.
    """
    angle_period = original_range[1] - original_range[0]
    rewrapped_angle = ((unwrapped_angle - original_range[0]) % angle_period) + original_range[0]
    return rewrapped_angle


def unwrap_nonsequential_array(
    wrapped_array: np.ndarray,
    period: float,
    reference_angle: float | None = None,
) -> np.ndarray:
    """
    Unwrap array of periodic values that may have non-sequential entries.

    Unlike numpy.unwrap, which assumes sequential entries, this function handles
    non-sequential entries by unwrapping each entry relative to a fixed reference point.
    If no reference point is provided, the function uses the first entry in the array
    as the (arbitrary) reference point.

    When applying numpy.unwrap to periodic data with non-sequential entries, the
    resulting unwrapped values may still have large jumps between entries that are not
    next to each other in the original sequence.

    Parameters
    ----------
    wrapped_array
        Array of periodic values to unwrap.
    period
        Period of the values.
    """
    reference_angle_ = wrapped_array[0] if reference_angle is None else reference_angle
    unwrapped_array = np.array(
        [
            np.unwrap(np.array([reference_angle_, wrapped_angle]), period=period)[-1]
            for wrapped_angle in wrapped_array
        ]
    )
    return unwrapped_array


def filter_dataframe_by_track_length(
    dataframe: pd.DataFrame, track_length_column: str, minimum_track_length: int
) -> pd.DataFrame:
    """
    Filter dataframe to only include tracks above a minimum track length.

    **Error handling**

    If no tracks remain after filtering, a ValueError is raised with a message
    indicating that no tracks with length >= minimum_track_length remain after
    filtering, and suggesting to check the track length distribution and/or
    adjust the minimum_track_length threshold.

    Parameters
    ----------
    dataframe
        DataFrame containing data of interest, which must include a column for
        track length.
    track_length_column
        Name of the column containing track length values.
    minimum_track_length
        Minimum track length to filter tracks.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only tracks with length >=
        minimum_track_length.
    """

    logger.debug(
        "Filtering dataframe to only include tracks with length >= [ %s ] timepoints.",
        minimum_track_length,
    )
    logger.debug("Dataframe length before filtering: [ %s ] rows.", len(dataframe))
    # check that required columns are present in dataframe
    check_required_columns_in_dataframe(dataframe, [track_length_column])
    dataframe_filtered = dataframe[dataframe[track_length_column] >= minimum_track_length]

    # if empty dataframe after filtering, raise error
    if dataframe_filtered.empty:
        logger.error(
            "No tracks with length >= minimum_track_length [ %s ] after filtering. "
            "Check track length distribution and/or adjust minimum_track_length.",
            minimum_track_length,
        )
        raise ValueError(
            f"No tracks with length >= minimum_track_length [ {minimum_track_length} ] after filtering. "
            "Check track length distribution and/or adjust minimum_track_length."
        )

    # reset index of filtered dataframe
    dataframe_filtered = dataframe_filtered.reset_index(drop=True)

    logger.debug("Dataframe length after filtering: [ %s ] rows.", len(dataframe_filtered))

    return dataframe_filtered


def filter_dataframe_by_annotations(
    dataframe: pd.DataFrame,
    dataset_config: DatasetConfig,
    position_annotations: list[PositionAnnotation] | None = None,
    timepoint_annotations: list[TimepointAnnotation] | None = None,
) -> pd.DataFrame:
    """
    Remove annotated timepoints and positions from a dataframe of DiffAE features for one dataset.

    Default behavior is to remove all annotated timepoints and positions.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    dataset_config
        Dataset config for the dataset.
    position_annotations
        List of position annotations to remove. Use None to remove all annotated positions.
    timepoint_annotations
        List of timepoint annotations to remove. Use None to remove all annotated timepoints.

    Returns
    -------
    :
        Dataframe with annotated timepoints removed.
    """

    # check that required columns are present in dataframe
    required_columns = [Column.DATASET, Column.POSITION, Column.TIMEPOINT]
    check_required_columns_in_dataframe(dataframe, required_columns)

    if dataframe[Column.DATASET].nunique() != 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")

    if dataframe[Column.DATASET].unique()[0] != dataset_config.name:
        logger.error("Dataset name in dataframe does not match dataset name in dataset config.")
        raise ValueError("Dataset name in dataframe does not match dataset name in dataset config.")

    # get positions and timepoints to include based on annotations
    only_include_positions = get_unannotated_positions(dataset_config, position_annotations)
    only_include_positions_str = [f"P{pos}" for pos in only_include_positions]
    only_include_frames = get_all_unannotated_timepoints(dataset_config, timepoint_annotations)
    if dataframe[Column.POSITION].nunique() != len(dataset_config.zarr_positions):
        logger.warning("Expected dataframe to contain all positions in dataset, but it does not.")

    # filter dataframe to only include non-annotated positions
    # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
    # this checks if all entries in the `.POSITION` column are strings that start with 'P'
    all_position_vals_start_with_P = (
        dataframe[Column.POSITION].transform(lambda pos: "P" in str(pos)).all()
    )
    if all_position_vals_start_with_P:
        position_type = "str"
        dataframe_exclude_positions = dataframe[
            dataframe[Column.POSITION].isin(only_include_positions_str)
        ]
    # otherwise it is assumed that the position column can be cast to `int`
    # (and if it can't be cast to `int`, an error will be raised later)
    else:
        position_type = "int"
        dataframe_exclude_positions = dataframe[
            dataframe[Column.POSITION].isin(only_include_positions)
        ]
    # filter dataframe to only include non-annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(Column.POSITION):
        # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
        if position_type == "str":
            position_as_int = int(cast(str, position)[1:])
        else:
            position_as_int = cast(int, position)
        include_frames_for_position = only_include_frames.get(position_as_int, [])
        df_position_filtered = df_position[
            df_position[Column.TIMEPOINT].isin(include_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def fit_pca(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
    filter_by_annotations: bool = True,
    include_cell_piling: bool = False,
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
    filter_by_annotations
        True to remove annotated timepoints and positions, False otherwise.
    include_cell_piling
        True to include cell piling timepoints, False otherwise.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """

    # Build PCA input dataframe
    pca_input_dataframe = build_pca_input_dataframe(
        dataset_collection_name, dataframe_manifest_name, filter_by_annotations, include_cell_piling
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


def get_dataset_descriptions(
    list_of_datasets: list[str],
    include_duration: bool = True,
    simple: bool = False,
    include_shear_stress: bool = False,
) -> dict[str, str]:
    """
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset,
        e.g., "48hr_Maximum_Shear_Stress_30_dyncm2".

    Parameters
    ----------
    list_of_datasets
        List of dataset names for which to get descriptions
    include_duration
        Include duration of each flow condition in description if true.
    simple
        Include description of shear regime (e.g., "High_Shear_Stress") if true.
    include_shear_stress
        Include exact shear stress value (e.g., "30_dyncm2") in description if true.

    Returns
    -------
    :
        A dictionary where keys are dataset names and values are descriptions.
    """

    description_dict = {}

    for dataset_name in list_of_datasets:
        config = load_dataset_config(dataset_name)
        description = []

        for condition, regime in zip(
            config.flow_conditions, config.shear_stress_regime, strict=True
        ):
            if include_duration:
                duration_in_frames = condition.stop - condition.start
                duration_in_hours = int(duration_in_frames * 5 / 60)
                description.append(f"{duration_in_hours}hr")

            if simple:
                description.append(regime.value)

            if not simple or include_shear_stress:
                description.append(f"{int(condition.shear_stress)}dyncm2")

        description_dict[dataset_name] = "_".join(description)

    return description_dict


def parse_dataset_description(dataset_description: str) -> str:
    """Parse dataset description for better readability in plot titles."""
    # replace underscores with spaces for better readability
    description_parsed = dataset_description.replace("_", " ")
    # find [0-9]dyncm2, put comma and space before, put a space between number and unit,
    # and change dyncm2 to dyn/cm^2 for better readability
    description_parsed = re.sub(r"(\d+)dyncm2", r", \1 dyn/cm$^2$", description_parsed)
    # turn capital 'S' into lowercase 's' for shear stress
    description_parsed = description_parsed.replace(" Shear Stress", " shear stress")
    # remove unwanted space before comma
    description_parsed = description_parsed.replace(" ,", ",")
    return description_parsed


def add_description_column(
    df: pd.DataFrame, dataset_name: str, simple: bool = False
) -> pd.DataFrame:
    """
    Add description column to DataFrame df.
    (Descriptions are currently based on the dataset name.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for dataset dataset_name
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column
    - dataset_name: str, name of dataset to add description for
    - simple (optional): bool, whether to use simple description
        (e.g., "48hr_High")

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added description column
    """
    # get descriptions for each dataset name
    description = get_dataset_descriptions([dataset_name], simple=simple)

    # add description column to DataFrame
    df["description"] = description[dataset_name]  # add description to DataFrame

    return df


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


def df_to_array(df: pd.DataFrame, column_names: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.
    This function fills missing timepoints (for example filtered as outliers)
    with NaNs such that there is a row for every timepoint within the dataset
    duration for each crop.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T
    - column_names: list[str], list of column names for features to include
        in output array

    Outputs:
    - feats: np.ndarray, array of feature data for all crops
        at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    """
    # check that required columns are present in dataframe
    required_columns = [Column.CROP_INDEX, Column.TIMEPOINT, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    # get array of num crops x valid timepoints x num PCs, padding with NaNs
    # where timepoints are missing
    full_timepoint_range = (df[Column.TIMEPOINT].min(), df[Column.TIMEPOINT].max())

    feats = []
    for _, data_crop in df.groupby(Column.CROP_INDEX):
        data_crop = data_crop.sort_values(by=Column.TIMEPOINT)
        data_crop_filled = fill_missing_timepoints(data_crop, full_timepoint_range)
        feats.append(data_crop_filled[column_names].values)

    return np.array(feats)


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list[pd.DataFrame], list[float]]:
    """
    Get crop-based feature data (Diffusion AE output) for each flow condition present in a dataset.

    If there is only one flow condition, this method returns a lists of length 1
    containing the original dataframe and single shear stress value.

    Parameters
    ----------
    df_proj
        DataFrame containing the PCA-projected feature data for one dataset.
    dataset_config
        DatasetConfig object for the given dataset.

    Returns
    -------
    :
        List of DataFrames, each containing the feature data for one flow condition.
    :
        List of shear stress values for each flow condition.
    """
    # check that required columns are present
    check_required_columns_in_dataframe(df_proj, [Column.TIMEPOINT])

    # get flow condition information from dataset config
    flow_conditions = dataset_config.flow_conditions

    # split out data by flow condition,
    # starting with first flow condition
    first_shear = flow_conditions[0].shear_stress
    # initialize list of shear stress conditions
    shear_list = [first_shear]
    # if there is a change in flow condition
    if len(flow_conditions) > 1:
        # get frame number where second flow condition starts
        change_frame = get_frame_after_flow_change(dataset_config)
        # get second shear stress condition
        second_shear = flow_conditions[1].shear_stress
        shear_list.append(second_shear)
        logger.debug("Shear stress [ %s ] dyn/cm^2 until frame [ %s ]", first_shear, change_frame)
        logger.debug("Shear stress [ %s ] dyn/cm^2 after frame [ %s ]", second_shear, change_frame)
        # separate data into two dataframes based on
        # frame number where flow condition changes
        data_flow1 = df_proj[df_proj[Column.TIMEPOINT] < change_frame].copy()
        data_flow2 = df_proj[df_proj[Column.TIMEPOINT] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        logger.debug("Constant shear stress [ %s ] dyn/cm^2", first_shear)
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list


@overload
def _take_dataframe_column_diff(
    dataframe_column: pd.Series, diff_step: int, fillna_value: float | None = None
) -> pd.Series: ...


@overload
def _take_dataframe_column_diff(
    dataframe_column: pd.DataFrame, diff_step: int, fillna_value: float | None = None
) -> pd.DataFrame: ...


def _take_dataframe_column_diff(
    dataframe_column: pd.Series | pd.DataFrame, diff_step: int, fillna_value: float | None = None
) -> pd.Series | pd.DataFrame:
    """
    Helper function to take the difference along a columns of a DataFrame, given
    a specified step size.

    The returned Series or DataFrame will contain the differences along the
    input column(s), with NaN values at the end where the difference could not
    be computed due to shifting. If a fillna_value is provided, NaN values will
    be replaced with the fillna_value.

    Parameters
    ----------
    dataframe_column
        A column of a DataFrame.
    diff_step
        The number of rows ahead to take the difference with.
    fillna_value
        Optional, value to fill NaN values with after taking the difference.
    """
    diffed_column = dataframe_column.diff(periods=diff_step).shift(-diff_step)
    if fillna_value is not None:
        diffed_column = diffed_column.fillna(fillna_value)
    return diffed_column


def compute_forward_differences_along_trajectory(
    df_traj: pd.DataFrame,
    column_names: list,
    polar_angle_period: float = PERIOD_THETA_RESCALED,
    time_lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward differences at a given time lag along a trajectory in
    feature space.

    **Polar angle handling**

    If one of the input ``column_names`` is 'polar_theta', the function will
    compute circular differences for the polar angle feature using the given
    ``polar_angle_period``. Specifically, it will unwrap the polar angle
    trajectory according to the given period for each crop before computing
    differences.

    **Time lag handling**

    The input ``time_lag`` determines the time lag (in number of frames) to use
    when computing forward differences. By default, it is set to 1, which
    corresponds to forward differences between consecutive timepoints. If a
    different time lag is specified, the function will compute differences
    between timepoints that are separated by ``time_lag`` number of frames.

    Parameters
    ----------
    df_traj
        DataFrame containing the feature data for one crop trajectory.
    column_names
        List of column names corresponding to the features of interest in the
        DataFrame.
    polar_angle_period
        Period of the polar angle feature, used to compute circular differences
        for angular data.
    time_lag
        Time lag (in number of frames) for forward difference calculation.

    Returns
    -------
    :
        Array of feature values along the trajectory for the specified columns.
    :
        Array of forward differences in feature values along the trajectory for
        the specified columns.
    """
    # initialize name for difference columns
    diff_column_names = [f"{col}{Column.DiffAEData.DIFFERENCE_SUFFIX}" for col in column_names]
    timepoint_diff_column = f"{Column.TIMEPOINT}{Column.DiffAEData.DIFFERENCE_SUFFIX}"

    # add column giving difference in timepoint between rows separated by
    # time_lag convert NaN to 0 -- occurs at end of trajectory
    df_traj[timepoint_diff_column] = _take_dataframe_column_diff(
        df_traj[Column.TIMEPOINT], time_lag, fillna_value=0
    )

    # add columns giving difference in feature values between consecutive
    # dataframe rows
    df_traj[diff_column_names] = _take_dataframe_column_diff(
        df_traj[column_names], time_lag, fillna_value=0
    )

    # if one of the column names is `polar_theta`, need to replace with the
    # circular difference for angular data instead of simple difference
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        angle_diff_column = f"{Column.DiffAEData.POLAR_ANGLE}{Column.DiffAEData.DIFFERENCE_SUFFIX}"
        df_traj[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"] = np.unwrap(
            df_traj[Column.DiffAEData.POLAR_ANGLE].values, period=polar_angle_period
        )
        df_traj[angle_diff_column] = _take_dataframe_column_diff(
            df_traj[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"], time_lag, fillna_value=0
        )
        df_traj.drop(columns=[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"], inplace=True)

    # trajectory values to keep -- only keep steps where time difference is <=
    # time_lag which includes the last point in the trajectory (which has time
    # difference set to 0)
    traj_mask = df_traj[timepoint_diff_column] <= time_lag
    filtered_traj_array = df_traj[traj_mask][column_names].to_numpy()
    if time_lag > 1:
        # drop last time_lag - 1 points, as there is no valid difference there
        filtered_traj_array = filtered_traj_array[: -time_lag + 1]

    # for the gradient, only keep steps where time difference is exactly
    # time_lag frames i.e., no valid difference at the end of the trajectory
    # (only forward differences)
    gradient_mask = df_traj[timepoint_diff_column] == time_lag
    filtered_d_traj_array = df_traj[gradient_mask][diff_column_names].to_numpy()

    return filtered_traj_array, filtered_d_traj_array


def get_traj_and_diff(
    df: pd.DataFrame,
    column_names: list,
    polar_angle_period: float = PERIOD_THETA_RESCALED,
    time_lag: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Get trajectories and single-timepoint displacement vectors (forward
    differences) for each single-crop trajectory in feature space.

    **Input dataframe**

    The input dataframe should have columns for:
        - frame_number: timepoint of the crop
        - crop_index: unique index for each crop
        - columns for each feature (e.g., pc_0, pc_1, pc_2, ...)
           matching input ``column_names``

    See documentation for `compute_forward_differences_along_trajectory` for
    more details on the numerical calculation of the forward differences.

    Parameters
    ----------
    df
        DataFrame with columns for each feature.
    column_names
        List of column names corresponding to the features of interest in the
        DataFrame.
    polar_angle_period
        Period of the polar angle feature, used to compute circular differences
        for angular data.
    time_lag
        Time lag (in number of frames) for forward difference calculation.

    Returns
    -------
    :
        List of individual crop trajectories in feature space.
    :
        List of displacement vectors along each trajectory in feature space.
    """
    # check that required columns are present
    required_columns = [Column.TIMEPOINT, Column.CROP_INDEX, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for _, df_crop in df.groupby(Column.CROP_INDEX):
        # skip if time_lag is larger than number of timepoints in this trajectory
        if time_lag > df_crop[Column.TIMEPOINT].nunique():
            continue

        # sort by timepoint to ensure that trajectory is in correct order before
        # computing differences
        df_crop_ = df_crop.sort_values(by=Column.TIMEPOINT)

        # compute forward differences along trajectory for this crop, and filter
        # to keep only differences between timepoints that are separated by
        # time_lag number of frames (accounts for any missing timepoints in the
        # trajectory, for example due to outlier filtering)
        filtered_traj, filtered_d_traj = compute_forward_differences_along_trajectory(
            df_crop_, column_names, polar_angle_period, time_lag
        )

        # if either the returned trajectory or difference arrays are empty, skip
        # this trajectory
        if filtered_traj.size == 0 or filtered_d_traj.size == 0:
            continue

        # else, append and continue through the loop
        traj_list.append(filtered_traj)
        d_traj_list.append(filtered_d_traj)

    # if lists are empty, log warning
    if len(traj_list) == 0 or len(d_traj_list) == 0:
        logger.warning(
            "No valid trajectories found after computing forward differences with time lag [ %s ]. "
            "Check that the input dataframe has the required columns and that the time lag is not "
            "larger than the number of timepoints in the trajectories.",
            time_lag,
        )
    return traj_list, d_traj_list


def fill_missing_timepoints(
    data_crop: pd.DataFrame,
    full_timepoint_range: tuple[float, float],
) -> pd.DataFrame:
    """
    Fill missing timepoints in dataframe for a single crop using NaN padding.
    Note: this function resets the index of the input crop-based dataframe.

    Parameters
    ----------
    data_crop
        DataFrame for a single crop.
    full_timepoint_range
        Tuple specifying the full range of timepoints (start, end) for the dataset.

    Returns
    -------
    :
        DataFrame with missing timepoints filled with NaNs.
    """

    # use full timepoint range for the dataset to ensure that all timepoints are
    # included
    all_timepoints = np.arange(full_timepoint_range[0], full_timepoint_range[1] + 1)

    # reindex dataframe to include all timepoints in full range
    data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

    # reset index to restore timepoint column
    data_crop_filled = data_crop_filled.reset_index()

    return data_crop_filled
