import logging
import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_datasets_in_collection,
    get_frame_after_flow_change,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_annotations,
    filter_dataframe_by_track_length,
)
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.polar_coords import pcs_to_polar_r, pcs_to_polar_theta
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAME_GROUPS,
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


def get_latent_feature_column_names(num_latent_dims: int) -> list[str]:
    """
    Get list of latent feature column names for given number of latent dimensions.

    Parameters
    ----------
    num_latent_dims
        Number of latent dimensions.

    Returns
    -------
    :
        List of latent feature column names.
    """
    feat_cols = [f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}" for i in range(num_latent_dims)]
    return feat_cols


def get_pc_column_names(num_pcs: str | int) -> list[str]:
    """
    Get list of PCA feature column names for given number of principal components.

    Parameters
    ----------
    num_pcs
        Either number of principal components (if an integer is provided) or a
        group of principal components to include (if a string is provided,
        e.g., "default" or "polar_coords").

    Returns
    -------
    :
        List of PCA feature column names.
    """
    if isinstance(num_pcs, int):
        pc_cols = [f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i+1}" for i in range(int(num_pcs))]
    elif isinstance(num_pcs, str):
        pc_cols = DIFFAE_PC_COLUMN_NAME_GROUPS.get(num_pcs, [])
        if not pc_cols:
            raise ValueError(
                f"Invalid num_pcs string [ {num_pcs} ]. Must be either an integer or\
                one of the following strings: {list(DIFFAE_PC_COLUMN_NAME_GROUPS.keys())}"
            )
    else:
        raise ValueError("num_pcs must be either an integer or a string.")
    return pc_cols


def get_latent_feature_column_names_from_dataframe(dataframe: pd.DataFrame) -> list[str]:
    """
    Get list of latent feature column names for given number of latent dimensions.

    Matches columns that start with the latent feature column name prefix
    as defined in ColumnName.DiffAEData.LATENT_FEATURE_PREFIX.

    Parameters
    ----------
    dataframe
        DataFrame containing latent feature columns.

    Returns
    -------
    :
        List of latent feature column names.
    """
    # regular expression to match latent feature columns
    feat_cols_match = [
        re.match(f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}[0-9]+$", col)
        for col in dataframe.columns
    ]
    feat_cols = [col.group() for col in feat_cols_match if col is not None]
    return feat_cols


def build_pca_input_dataframe(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
    filter_by_annotations: bool = True,
    include_cell_piling: bool = False,
) -> pd.DataFrame:
    """
    Build input dataframe for fitting PCA model using given dataset collection.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    filter_by_annotations
        Whether to remove annotated timepoints and positions from the dataframes before fitting PCA.
    include_cell_piling
        True to include cell piling timepoints, False otherwise.

    Returns
    -------
    :
        Input dataframe for fitting PCA.
    """

    # Get dataframe manifest name if not provided based on default model manifest
    if dataframe_manifest_name is None:
        dataframe_manifest_name = get_feature_dataframe_manifest_name(
            load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
            DEFAULT_MODEL_RUN_NAME,
        )

    # Load dataframe manifest
    manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Get datasets in collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)
    logger.info("Datasets being used to fit PCA: [ %s ]", ", ".join(dataset_names))

    # Load and filter out annotated timepoints (if requested) for each dataset
    dataframe_list = []
    for dataset_name in dataset_names:
        location = get_dataframe_location_for_dataset(manifest, dataset_name)
        dataframe = load_dataframe(location)
        if filter_by_annotations:
            annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
            if include_cell_piling:
                annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
            timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)
            dataframe_filtered = filter_dataframe_by_annotations(
                dataframe,
                load_dataset_config(dataset_name),
                timepoint_annotations=timepoint_annotations,
            )
        else:
            dataframe_filtered = dataframe
        dataframe_list.append(dataframe_filtered)

    # Merge dataframes for all datasets and filter for feature columns ("feat_" prefix)
    data_ref = pd.concat(dataframe_list, ignore_index=True)
    diffae_feature_cols = get_latent_feature_column_names_from_dataframe(data_ref)
    return data_ref[diffae_feature_cols]


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
    feat_col_names = get_latent_feature_column_names(num_features)
    pc_col_names = get_pc_column_names(num_pcs)

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
    feat_cols: list[str] | None = None,
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
    if feat_cols is None:
        feat_cols = get_latent_feature_column_names_from_dataframe(df)
    else:
        check_required_columns_in_dataframe(df, feat_cols)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    num_pcs = pca.components_.shape[0]  # number of principal components
    pc_cols = get_pc_column_names(num_pcs)
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
    feat_cols = get_latent_feature_column_names_from_dataframe(df)

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
