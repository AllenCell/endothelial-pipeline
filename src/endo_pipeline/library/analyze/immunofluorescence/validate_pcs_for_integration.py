from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, get_output_path, load_model, upload_file_to_fms
from endo_pipeline.library.analyze.diffae_manifest import (
    get_dataframe_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from endo_pipeline.library.model import (
    generate_overrides_for_model_eval,
    upload_prediction_dataframe_to_fms,
)
from endo_pipeline.library.process.registration import align_all_positions
from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings import Z_SLICE_OFFSETS


def apply_model_paired_fixed_live(
    fixed_dataset_name: str,
    live_dataset_name: str,
    model_manifest_name: str,
    run_name: str,
    align_fluo: bool = True,
    upload_to_fms: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Align paired fixed and live data and apply a diffAE model to extract features.

    Parameters
    ----------
    fixed_dataset_name
        Dataset name of the fixed cell images.
    live_dataset_name
        Dataset name of the live cell images.
    model_manifest_name
        The name of the model manifest from which to load the model for feature extraction.
    run_name
        The name of the run corresponding to the model.
    align_fluo
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    upload_to_fms : bool
        Whether to upload dataframe of extracted features to FMS.

    Returns
    -------
    save_path: Path
        Local path to parent directory where intermediate data is saved
    fixed_features_path : Path
        Local path where fixed data features are saved in a parquet file
    live_feature_path: Path
        Local path where live data features are saved in a parquet file
    """

    # Get dataset configs
    fixed_dataset_config = load_dataset_config(fixed_dataset_name)
    live_dataset_config = load_dataset_config(live_dataset_name)

    # Load finetuned DiffAE model for extracting features
    # load model manifest
    model_manifest = load_model_manifest(model_manifest_name)

    # load model run_name from model manifest, override with eval config,
    # and make sure that "model_manifest_name" and "run_name" are stored in the config
    # as "experiment_name" and "run_name" for logging purposes
    model = load_model(model_manifest, run_name)

    # Set directory for aligned data
    save_path = get_output_path(
        "models", model_manifest_name, run_name, f"{fixed_dataset_name}_vs_{live_dataset_name}"
    )
    data_save_path = save_path / f"aligned_{fixed_dataset_name}_vs_{live_dataset_name}.csv"

    # Align data if saved aligned data not already stored
    if not data_save_path.exists():
        data = align_all_positions(
            target_dataset_name=live_dataset_name,
            moving_dataset_name=fixed_dataset_name,
            resolution_level=1,
            z_slice_offsets=Z_SLICE_OFFSETS,
            savedir=save_path,
            alignment_method="sift",
            align_fluo=align_fluo,
        )
        # Channel used for inference is in the aligned images, which are single channel
        data["channel"] = 0
        data.to_csv(data_save_path, index=False)

    # Load diffAE model and create new overrides object
    # some of these are temporary while we adjust our config infrastructure
    image_dataset_obj = "endo_pipeline.library.model.image_loading.MultiDimImageDataset"
    overrides = {
        "model.spatial_inferer.splitter.overlap": 0.9,
        "data.predict_dataloaders.dataset._target_": image_dataset_obj,
        "data.predict_dataloaders.dataset.replace_rate": 1.0,
        "data.predict_dataloaders.dataset.num_init_workers": 24,
        "data.predict_dataloaders.dataset.num_replace_workers": 24,
        "data.predict_dataloaders.batch_size": 64,
    }

    # Apply on live_dataset/"target" images - start by constructing overrides
    live_overrides = overrides.copy()  # copy to avoid overriding the original
    live_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "target"})
    prediction_live_suffix = f"{live_dataset_name}_{model_manifest_name}_{run_name}_features"
    live_overrides = generate_overrides_for_model_eval(
        live_overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        dataset_name=live_dataset_name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        prediction_filename_suffix=prediction_live_suffix,
    )

    # Apply model on target/moving images - override config and run model prediciton
    model.override_config(live_overrides)
    # the following three lines are temporary while we adjust our model/config infrastructure
    rm_keys = ["num_workers", "cache_num", "csv_path", "dict_meta"]
    for key in rm_keys:
        model.cfg.data.predict_dataloaders.dataset.pop(key, None)
    model.predict()

    # Apply on fixed dataset/"moving" images - start by constructing overrides
    fixed_overrides = overrides.copy()  # copy to avoid overriding the original
    fixed_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "moving"})
    prediction_fixed_suffix = f"{fixed_dataset_name}_{model_manifest_name}_{run_name}_features"
    fixed_overrides = generate_overrides_for_model_eval(
        fixed_overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        dataset_name=fixed_dataset_name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        prediction_filename_suffix=prediction_fixed_suffix,
    )
    # Apply on fixed dataset/"moving" images - override config and run model prediciton
    model.override_config(fixed_overrides)
    # the following two lines are temporary while we adjust our model/config infrastructure
    for key in rm_keys:
        model.cfg.data.predict_dataloaders.dataset.pop(key, None)
    model.predict()

    # Define paths to saved features from model for both fixed and live datasets
    fixed_features_path = save_path / prediction_fixed_suffix
    live_features_path = save_path / prediction_live_suffix
    if upload_to_fms:

        upload_prediction_dataframe_to_fms(
            fixed_features_path,
            fixed_dataset_config,
            model_manifest,
            run_name,
            dataframe_manifest_name=f"{model_manifest_name}_{run_name}_grid",
            workflow_name="validate_pcs_for_integration",
            workflow_parameters={},
        )

        upload_prediction_dataframe_to_fms(
            live_features_path,
            live_dataset_config,
            model_manifest,
            run_name,
            dataframe_manifest_name=f"{model_manifest_name}_{run_name}_grid",
            workflow_name="validate_pcs_for_integration",
            workflow_parameters={},
        )

    return save_path, fixed_features_path, live_features_path


def project_paired_fixed_live_data_into_ref_PC_space(
    pca: PCA,
    fixed_features_path: Path = "fixed_features.parquet",
    live_features_path: Path = "live_features.parquet",
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

    fixed_pc_features = project_manifest_to_pcs(fixed_features, pca)
    live_pc_features = project_manifest_to_pcs(live_features, pca)

    return fixed_pc_features, live_pc_features


def create_reference_timelapse_datasets(
    pca: PCA,
    reference_dataset_name: str,
    model: str = "diffae_04_10",
    time_lag: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create reference timelapse datasets to determine role of time lag in differences between fixed and live data.
    This function loads a no-flow timelapse reference dataset and gets PC values.
    It creates one copy of this data that is lagged in time by the same time gap between the live and fixed data snapshots.
    It then creates a second version which is just truncated to remove the rows that were shifted out by the lag.

    Parameters
    ----------
    pca : PCA
        PCA model to project features into reference PC space
    reference_dataset_name : str
        Name of the reference dataset to use for creating the timelapse datasets
    model : ModelConfig
        Model configuration to use for loading the reference dataset
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
    reference_features = reference_features.sort_values(by="frame_number")
    reference_features = (
        reference_features.groupby("crop_index").apply(fill_empty_frames).reset_index(drop=True)
    )
    df_lag = reference_features.groupby("crop_index").apply(create_lagged_dataset, time_lag)
    df_trunc = reference_features.groupby("crop_index").apply(create_truncated_dataset, time_lag)
    df_lag, df_trunc = dropna_both_df(df_lag, df_trunc)
    return df_lag, df_trunc


def fill_empty_frames(crop: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in any empty frames with NaNs

    Parameters
    ----------
    crop : pd.DataFrame
        Dataframe containing the crop data for a single crop_index

    Returns
    -------
    crop : pd.DataFrame
        Dataframe with empty frames filled in with NaNs
    """
    frame_numbers = crop["frame_number"].unique()
    all_frame_numbers = pd.DataFrame(
        {"frame_number": np.arange(frame_numbers.min(), frame_numbers.max() + 1)}
    )
    crop = pd.merge(all_frame_numbers, crop, on="frame_number", how="left")
    crop["crop_index"] = crop["crop_index"].fillna(crop["crop_index"].iloc[0])
    return crop


def create_lagged_dataset(
    crop: pd.DataFrame,
    time_lag: int,
) -> pd.DataFrame:
    """
    Create a lagged dataset by shifting the crop data by the specified time lag.

    Parameters
    ----------
    crop : pd.DataFrame
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
    crop_new["frame_number"] = crop["frame_number"]
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


def dropna_both_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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
    df1, df2 : pd.DataFrame
        Dataframes with rows dropped where all PC values are NaN in both dataframes
    """

    pc_cols = [col for col in df1.columns if "pc" in col]
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
    x, y = live_features[f"pc{pc}"].values, fixed_features[f"pc{pc}"].values
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    center = (float(mean_x), float(mean_y))

    # Get covaraince matrix of live and fixed data then calculate its eigenvectors and eigenvalues
    data = [[live, fixed] for live, fixed in zip(x, y)]
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Get indices that would sort the eigenvalues in ascending order
    sorted_indices = eigenvalues.argsort()[::-1]

    # Sort eigenvalues and eigenvectors
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # The eigenvectors define the orientation/tilt angle of a confidence ellipse and the associated eigenvalues
    # give the lengths of the ellipse axes.
    # A linear colinear with the major axis of the ellipse is our linear model mapping between live and fixed data
    # for this PC value
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
