from pathlib import Path

import numpy as np
import pandas as pd
from cyto_dl.api import CytoDLModel
from matplotlib.patches import Ellipse
from pandas.core.groupby import DataFrameGroupBy
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe import (
    get_dataframe_for_dynamics_workflows,
    project_features_to_pcs,
)
from endo_pipeline.library.model.eval_model import generate_overrides_for_model_eval
from endo_pipeline.library.process.registration import align_all_positions
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings import Z_SLICE_OFFSETS, ColumnName


def evaluate_model_paired_fixed_live(
    fixed_dataset_name: str,
    live_dataset_name: str,
    model_save_path: Path,
    data_save_path: Path,
    model: CytoDLModel,
    align_fluo: bool = True,
    num_gpus: int | None = None,
) -> tuple[Path, Path]:
    """
    Align paired fixed and live data and apply a diffAE model to extract features.

    Parameters
    ----------
    fixed_dataset_name : str
        Dataset name to use as the fixed images (i.e. the reference against which the moving images
        are registered)
    live_dataset_name : str
        Dataset name to use as the moving images (i.e. the images to be registered to the fixed
        images)
    model_save_path : Path
        Path to directory where data is saved
    data_save_path : Path
        Path to csv file where aligned data is saved
    model : CytoDLModel
        DiffAE model to use for feature extraction
    align_fluo : bool
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    num_gpus: int
        Number of GPUs to use

    Returns
    -------
    fixed_features_path : Path
        Local path where fixed data features are saved in a parquet file
    live_feature_path: Path
        Local path where live data features are saved in a parquet file
    """

    # Align data if saved aligned data not already stored
    if not data_save_path.exists():
        data = align_all_positions(
            live_dataset_name,
            fixed_dataset_name,
            resolution_level=1,
            z_slice_offsets=Z_SLICE_OFFSETS,
            savedir=model_save_path,
            alignment_method="sift",
            align_fluo=align_fluo,
        )
        # Channel used for inference is in the aligned images, which are single channel
        data["channel"] = 0
        data.to_csv(data_save_path, index=False)

    # get experiment name and run name
    experiment_name = model.cfg.experiment_name
    run_name = model.cfg.run_name

    # Evaluate model on target/moving images - set up config and run model
    target_overrides = generate_overrides_for_model_eval(
        save_path=model_save_path.as_posix(),
        data_path=data_save_path.as_posix(),
        dataset_name=live_dataset_name,
        model_manifest_name=experiment_name,
        run_name=run_name,
        num_gpus=num_gpus,
    )
    target_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "target"})
    target_overrides.update({"model.condition_key": "raw_moving"})
    model.override_config(target_overrides)
    rm_keys = ["num_workers", "cache_num", "csv_path", "dict_meta"]
    for key in rm_keys:
        model.cfg.data.predict_dataloaders.dataset.pop(key, None)
    model.predict()

    # Evaluate model on fixed data/"moving" images - set up config and run model
    moving_overrides = generate_overrides_for_model_eval(
        save_path=model_save_path.as_posix(),
        data_path=data_save_path.as_posix(),
        dataset_name=fixed_dataset_name,
        model_manifest_name=experiment_name,
        run_name=run_name,
        num_gpus=num_gpus,
    )
    moving_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "moving"})
    moving_overrides.update({"model.condition_key": "raw_moving"})
    model.override_config(moving_overrides)
    rm_keys = ["num_workers", "cache_num", "csv_path", "dict_meta"]
    for key in rm_keys:
        model.cfg.data.predict_dataloaders.dataset.pop(key, None)
    model.predict()

    # Define paths to saved features from model for both fixed and live datasets
    fixed_features_path = (
        model_save_path
        / f"predict_{fixed_dataset_name}_{experiment_name}_{run_name}_features.parquet"
    )
    live_features_path = (
        model_save_path
        / f"predict_{live_dataset_name}_{experiment_name}_{run_name}_features.parquet"
    )

    return fixed_features_path, live_features_path


def project_paired_fixed_live_data_into_ref_pc_space(
    pca: PCA,
    fixed_features_path: Path = Path("fixed_features.parquet"),
    live_features_path: Path = Path("live_features.parquet"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project features from applying fine tuned diffAE model to fixed and live data into
    reference PC space.

    Parameters
    ----------
    pca : PCA | None
        PCA model
    fixed_features_path : Path
        Path to the fixed features manifest
    live_features_path : Path
        Path to the live features manifest


    Returns
    -------
    fixed_pc_features: pd.DataFrame
        Dataframe containing PCs for fixed data
    live_pc_features: pd.DataFrame
        Dataframe containing PCs for live data
    """

    # load pc features for fixed and live data
    fixed_features = pd.read_parquet(fixed_features_path)
    live_features = pd.read_parquet(live_features_path)

    fixed_pc_features = project_features_to_pcs(fixed_features, pca)
    live_pc_features = project_features_to_pcs(live_features, pca)

    return fixed_pc_features, live_pc_features


def create_reference_timelapse_datasets(
    pca: PCA,
    reference_dataset_name: str,
    model: str = "diffae_04_10",
    time_lag: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create reference timelapse datasets to determine role of time lag in differences between fixed
    and live data. This function loads a no-flow timelapse reference dataset and gets PC values.
    It creates one copy of this data that is lagged in time by the same time gap between the live
    and fixed data snapshots. It then creates a second version which is just truncated to remove
    the rows that were shifted out by the lag.

    Parameters
    ----------
    pca : PCA
        PCA model to project features into reference PC space
    reference_dataset_name : str
        Name of the reference dataset to use for creating the timelapse datasets
    model : str
        Name of model to use for loading the reference dataset
    time_lag : int
        Number of frames to lag the reference dataset by.
        This is the same time gap between the live and fixed data snapshots.

    Returns
    -------
    df_lag : pd.DataFrame
        Dataframe containing the lagged reference features
    df_trunc : pd.DataFrame
        Dataframe containing the reference features truncated to
        remove the rows that were shifted out by the lag
    """

    # Load the PC data for reference no flow timelapse dataset
    manifest = load_dataframe_manifest(model)
    reference_features = get_dataframe_for_dynamics_workflows(reference_dataset_name, manifest, pca)

    # Create and return lagged and truncated datasets
    reference_features = reference_features.sort_values(by=ColumnName.TIMEPOINT)
    reference_features = (
        reference_features.groupby(ColumnName.CROP_INDEX)
        .apply(fill_empty_frames)
        .reset_index(drop=True)
    )

    df_lag = reference_features.groupby(ColumnName.CROP_INDEX).apply(
        create_lagged_dataset, time_lag
    )
    df_trunc = reference_features.groupby(ColumnName.CROP_INDEX).apply(
        create_truncated_dataset, time_lag
    )
    df_lag, df_trunc = dropna_both_df(df_lag, df_trunc)
    return df_lag, df_trunc


def fill_empty_frames(crop: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in any empty frames with NaNs.

    Parameters
    ----------
    crop : pd.DataFrame
        Dataframe containing the crop data for a single crop_index

    Returns
    -------
    crop : pd.DataFrame
        Dataframe with empty frames filled in with NaNs
    """
    frame_numbers = crop[ColumnName.TIMEPOINT].unique()
    all_frame_numbers = pd.DataFrame(
        {ColumnName.TIMEPOINT: np.arange(frame_numbers.min(), frame_numbers.max() + 1)}
    )
    crop = pd.merge(all_frame_numbers, crop, on=ColumnName.TIMEPOINT, how="left")
    crop[ColumnName.CROP_INDEX] = crop[ColumnName.CROP_INDEX].fillna(
        crop[ColumnName.CROP_INDEX].iloc[0]
    )
    return crop


def create_lagged_dataset(
    crop: DataFrameGroupBy,
    time_lag: int,
) -> pd.DataFrame:
    """
    Create a lagged dataset by shifting the crop data by the specified time lag.

    Parameters
    ----------
    crop : DataFrameGroupBy
        Dataframe containing the crop data for a single crop_index
    time_lag : int
        Number of frames to lag the dataset by.
        This is the same time gap between the live and fixed data snapshots.

    Returns
    -------
    crop_new : pd.DataFrame
        Dataframe with the lagged crop data
    """

    crop_new = crop.copy()
    crop_new = crop_new.shift(time_lag)
    crop_new[ColumnName.TIMEPOINT] = crop[ColumnName.TIMEPOINT]
    return crop_new.iloc[time_lag:]


def create_truncated_dataset(
    crop: pd.DataFrame,
    time_lag: int,
) -> pd.DataFrame:
    """
    Create a truncated dataset by removing the first `time_lag` rows from the crop data.

    Parameters
    ----------
    crop : pd.DataFrame
        Dataframe containing the crop data for a single crop_index
    time_lag : int
        Number of frames to truncate the dataset by.

    Returns
    -------
    crop : pd.DataFrame
        Dataframe with the truncated crop data
    """
    return crop.iloc[time_lag:]


def dropna_both_df(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows from both dataframes where PC values for either dataframe are NaN.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe containing PC values
    df2 : pd.DataFrame
        Second dataframe containing PC values

    Returns
    -------
    df1, df2 : tuple[DataFrame, DataFrame]
        Dataframes with rows dropped where all PC values are NaN in both dataframes
    """

    pc_cols = [col for col in df1.columns if "pc_" in col]
    mask = df1[pc_cols].notna().all(axis=1) & df2[pc_cols].notna().all(axis=1)
    return df1[mask], df2[mask]


def get_paired_fixed_live_validation_features(
    pc: int,
    fixed_features: pd.DataFrame,
    live_features: pd.DataFrame,
    n_std: int = 2,
) -> tuple[tuple, tuple]:
    """
    Use a confidence ellipse to define the linear model mapping between PC values for live and fixed
    data and determine the uncertainty in the fixed PC values based on this mapping. The confidence
    ellipse is oriented with its major axis along the direction of highest variation in the data; a
    line colinear with this axis is used to define a linear model for mapping between fixed and live
    data. The y-projection of the minor axis gives our uncertainty in the fixed PC value based on
    this model.

    Parameters
    ----------
    pc : int
        PC to analyze
    fixed_features : pd.DataFrame)
        Dataframe containing PCs for fixed data
    live_features : pd.DataFrame
        Dataframe containing PCs for live data
    n_std : int
        Number of standard deviations wide to make the confidence ellipse

    Returns
    -------
    x, y : tuple
        Live and fixed data respectively
    center, height, angle, slope, intercept, ellipse : tuple
        Confidence-ellipse based validation features to plot
    """

    # Format live and fixed features as needed for analysis and calculated the mean of each
    x, y = live_features[f"pc_{pc}"].values, fixed_features[f"pc_{pc}"].values
    mean_x = np.mean(np.asarray(x))
    mean_y = np.mean(np.asarray(y))
    center = (float(mean_x), float(mean_y))

    # Get covaraince matrix of live and fixed data then calculate its eigenvectors and eigenvalues
    data = [[live, fixed] for live, fixed in zip(x, y, strict=True)]
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Get indices that would sort the eigenvalues in ascending order
    sorted_indices = eigenvalues.argsort()[::-1]

    # Sort eigenvalues and eigenvectors
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # The eigenvectors define the orientation/tilt angle of a confidence ellipse and the associated
    # eigenvalues give the lengths of the ellipse axes.
    # give the lengths of the ellipse axes.
    # A linear colinear with the major axis of the ellipse is our linear model mapping between live
    # and fixed data for this PC value
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    slope = eigenvectors[1, 0] / eigenvectors[0, 0]
    intercept = mean_y - slope * mean_x
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor="magenta",
        label=rf"{n_std}$\sigma$ confidence",
    )

    return (x, y), (center, height, angle, slope, intercept, ellipse)
