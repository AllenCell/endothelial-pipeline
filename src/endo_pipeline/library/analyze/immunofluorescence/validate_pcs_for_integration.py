from pathlib import Path

import numpy as np
import pandas as pd
from cyto_dl.api import CytoDLModel
from matplotlib.patches import Ellipse
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import (
    DatasetConfig,
    ModelConfig,
    add_model_manifest,
    load_dataset_config,
    load_model_config,
    save_model_config,
)
from src.endo_pipeline.configs.model_config_utils import get_model_manifest
from src.endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.model.model_inputs import generate_overrides_for_model_eval
from src.endo_pipeline.library.process.registration import align_all_positions


def add_paired_fixed_live_data_fmsid_to_config(
    prediction_path: Path,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    model_path: Path,
) -> ModelConfig:
    """
    Upload path to FMS and add the FMS ID to the dataset config file for a dataset
    of paired fixed and live data.

    Parameters
    ----------
    prediction_path : str
        Path to the prediction file
    dataset_config : DatasetConfig
        Config file for the dataset
    model_config : ModelConfig
        Config file for the chosen model
    model_path : Path
        Path to the model directory. Used for extracting the commit hash.

    Returns
    -------
    model_config_updated : ModelConfig
        Updated model config with the FMS ID of the prediction file added to the dataset manifest
    """
    # build FMS annotations
    dataset_annotations = build_fms_annotations(
        dataset_config,
        include_timestamp=False,
        include_git_info=False,
        model=model_config,
        additional_notes=get_cytodl_commit_hash(model_config.mlflow_run_id, model_path),
    )

    # upload prediction file to FMS and get file ID
    file_id = upload_file_to_fms(
        prediction_path,
        annotations=dataset_annotations,
        file_type="parquet",
    )

    # Update the model config with the FMS ID of the prediction file
    model_config_updated = add_model_manifest(model_config, dataset_config.name, file_id)
    return model_config_updated


def apply_model_paired_fixed_live(
    fixed_dataset_name: str,
    live_dataset_name: str,
    model_name: str,
    align_fluo: bool = True,
    upload_features_to_FMS: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Align paired fixed and live data and apply a diffAE model to extract features.

    Parameters
    ----------
    fixed_dataset_name : str
        Dataset name to use as the fixed images (i.e. the reference against which the moving images are registered)
    live_dataset_name : str
        Dataset name to use as the moving images (i.e. the images to be registered to the fixed images)
    model_name : str
        The name of the model finetuned for fixation.
    align_fluo : bool
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    upload_features_to_FMS : bool
        Whether to upload validation data features to FMS. We may iteratre on analysis
        without changing features and therefore should default to not rewriting a new feature manifest every time
        this workflow is run.

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

    # Get diffAE model
    # load model config
    model_config = load_model_config(model_name)

    # Load DiffAE model
    model_path = get_output_path("models", model_name)  # new get_output_path function
    path_dict = download_model(model_config.mlflow_run_id, model_path)

    # Set directory for aligned data
    save_path = get_output_path(
        "models", model_name, f"{fixed_dataset_name}_vs_{live_dataset_name}"
    )
    data_save_path = save_path / f"aligned_{fixed_dataset_name}_vs_{live_dataset_name}.csv"

    # Align data if saved aligned data not already stored
    if not data_save_path.exists():
        data = align_all_positions(
            fixed_dataset_name,
            live_dataset_name,
            save_path,
            alignment_method="sift",
            align_fluo=align_fluo,
        )
        # Channel used for inference is in the aligned images, which are single channel
        data["channel"] = 0
        data.to_csv(data_save_path, index=False)

    # Apply on fixed images
    overrides = {"model.spatial_inferer.splitter.overlap": 0.9}
    fixed_overrides = overrides.copy()  # copy to avoid overriding the original
    fixed_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "fixed"})
    fixed_overrides = generate_overrides_for_model_eval(
        fixed_overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=fixed_dataset_name,
        model_name=model_name,
    )

    # Load diffAE model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])
    model.override_config(fixed_overrides)
    model.predict()

    # Apply on moving images
    overrides.update({"data.predict_dataloaders.dataset.img_path_column": "moving"})
    overrides = generate_overrides_for_model_eval(
        overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=live_dataset_name,
        model_name=model_name,
    )
    model.override_config(overrides)
    model.predict()

    fixed_features_path = save_path / f"predict_{fixed_dataset_name}_{model_name}_features.parquet"
    live_features_path = save_path / f"predict_{live_dataset_name}_{model_name}_features.parquet"

    if upload_features_to_FMS:
        print("Uploading fixed and live dataset feature manifests to FMS")
        model_config_updated_with_fixed = add_paired_fixed_live_data_fmsid_to_config(
            fixed_features_path,
            fixed_dataset_config,
            model_config,
            model_path,
        )
        model_config_updated_with_live = add_paired_fixed_live_data_fmsid_to_config(
            live_features_path,
            live_dataset_config,
            model_config_updated_with_fixed,
            model_path,
        )
        save_model_config(model_config_updated_with_live)

    return save_path, fixed_features_path, live_features_path


def project_paired_fixed_live_data_into_ref_PC_space(
    pca: Pipeline,
    fixed_features_path: Path = "fixed_features.parquet",
    live_features_path: Path = "live_features.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project features from applying fine tuned diffAE model to fixed and live data into
    reference PC space.

    Parameters
    ----------
    pca : Pipeline | None
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
    pca: Pipeline,
    reference_dataset_name: str,
    model: ModelConfig = "diffae_04_10",
    time_lag: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create reference timelapse datasets to determine role of time lag in differences between fixed and live data.
    This function loads a no-flow timelapse reference dataset and gets PC values.
    It creates one copy of this data that is lagged in time by the same time gap between the live and fixed data snapshots.
    It then creates a second version which is just truncated to remove the rows that were shifted out by the lag.

    Parameters
    ----------
    pca : Pipeline
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
    model_config = load_model_config(model)
    model_manifest = get_model_manifest(reference_dataset_name, model_config)
    reference_features = get_manifest_for_dynamics_workflows(model_manifest, pca)

    # Create and return lagged and truncated datasets
    reference_features = reference_features.sort_values(by="frame_number")
    df_lag = reference_features.groupby("crop_index").shift(time_lag).dropna()
    df_trunc = reference_features.groupby("crop_index").apply(
        create_truncated_dataset, time_lag=time_lag
    )
    return df_lag, df_trunc


def create_truncated_dataset(
    crop: pd.DataFrame,
    time_lag: int,
) -> pd.DataFrame:
    return crop.iloc[time_lag:]


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


def get_common_plot_range(
    fixed_features: pd.DataFrame,
    live_features: pd.DataFrame,
    lagged_live_features: pd.DataFrame,
    truncated_live_features: pd.DataFrame,
    pc: int,
) -> tuple[float, float]:
    """
    Get common plot ranges for each PC.

    Parameters
    ----------
    fixed_features : pd.DataFrame
        Dataframe containing PCs for fixed data
    live_features : pd.DataFrame
        Dataframe containing PCs for live data
    lagged_live_features : pd.DataFrame
        Dataframe containing time-lagged PC values for live data
    truncated_live_features : pd.DataFrame
        Dataframe containing original live data PC values truncated to remove the rows that were shifted out by the lag
    pc : int
        PC to analyze

    Returns
    -------
    x_min, x_max : tuple[float, float]
        Common plot ranges for fixed and live data for the specified PC
    """
    x_min = min(
        fixed_features[f"pc{pc}"].min(),
        live_features[f"pc{pc}"].min(),
        lagged_live_features[f"pc{pc}"].min(),
        truncated_live_features[f"pc{pc}"].min(),
    )
    x_max = max(
        fixed_features[f"pc{pc}"].max(),
        live_features[f"pc{pc}"].max(),
        lagged_live_features[f"pc{pc}"].max(),
        truncated_live_features[f"pc{pc}"].max(),
    )
    return x_min, x_max
