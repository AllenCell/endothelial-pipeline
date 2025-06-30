from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from cyto_dl.api import CytoDLModel
from matplotlib.patches import Ellipse
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from cellsmap.util.manifest_io import load_pca_model
from cellsmap.util.manifest_preprocessing import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import load_single_dataset_config, save_dataset_config
from src.endo_pipeline.configs.dataset_io import get_model_info
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import project_manifest_to_pcs
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.process.registration import align_all_positions
from src.endo_pipeline.workflows.apply_diffae_model import generate_overrides

"""
This workflow is used to calulcate the uncertainty and correction of PC values calculated on fixed data
for integration into analyses of live data.
The error is calculated for any single PC for any dataset by:
    1. Plotting the PC values for the two experiments against each other
    2. Fitting a 2-std dev confidence ellipse to the set of PC values from the paired experiments
    3. Projecting the height of the confidence ellipse onto the y-axis to get an error value
Reference: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
"""


def pc_error_from_confidence_ellipse(
    x: ArrayLike,
    y: ArrayLike,
    ax: plt.Axes,
    n_std: float = 3.0,
    facecolor: str = "none",
    **kwargs: Any,
) -> tuple[plt.Axes, Ellipse, float, float]:
    """
    Create a plot of the covariance confidence ellipse of *x* and *y* and caluclate the
    associated PC error (the y-axis projection of the ellipse height.
    This function is adapted from the code in the reference link in the header of this file.

    Args:
        x, y (array-like): Input data from paired experiments. These should be the PC values for
            one given PC, calculated for two paired experiments (for example, PC1 values for
            pre-fixation and post-fixation).
        ax (matplotlib.axes.Axes):The Axes object to draw the ellipse into.
        n_std (float): The number of standard deviations to determine the ellipse's radiuses.
        **kwargs: Forwarded to `~matplotlib.patches.Ellipse`

    Returns:
        ax (matplotlib.axes.Axes):The Axes object with the ellipse drawn on it.
        ellipse (matplotlib.patches.Ellipse): Ellipse object drawn on the Axes.
        yerr (float): The y-axis projection of the ellipse height, representing the PC error.
    """

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x: float = np.mean(x)

    # Calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y: float = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    slope, intercept = get_linear_model(scale_x, scale_y, mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    yerr = ellipse.height * scale_y * np.sin(45)
    ax.add_patch(ellipse)
    ellipse.set_label(rf"${n_std}\sigma$, y_err: {yerr:.3f}")
    return ax, ellipse, yerr, slope, intercept


def get_linear_fit(x: ArrayLike, y: ArrayLike):
    """
    Fits x, y data with a linear regression model.

    Args:
        x, y (array-like): Input data from paired experiments. These should be the PC values for
            one given PC, calculated for two paired experiments (for example, PC1 values for
            pre-fixation and post-fixation).

    Returns:
        LinearRegression: A linear regression model object
    """

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def get_linear_model_parameters(scale_x, scale_y, mean_x, mean_y):
    slope = scale_y / scale_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def get_linear_model(slope, intercept, x):
    return slope * x + intercept


def plot_ellipse_and_fit(
    x: ArrayLike,
    y: ArrayLike,
    pc: int,
    xlabel: str,
    ylabel: str,
    save_path: None | str | Path,
    n_std: float = 2.0,
) -> None:
    """
    Create a scatter plot of the paired *pc* data given as *x* and *y* from a given
    *dataset* with a humanreadable *name*. Then calculate and overlay covariance confidence
    ellipses of *x* and *y* for 1, 2 and 3 standard deviations out from the mean. The
    Includes a y=x line for reference.

    Args:
        x, y (array-like): Input data from paired experiments.
        pc (int): PC to analyze.
        xlabel (str):Human-readable label for experiment whose PC data is plotted on the x-axis.
        ylabel (str): Human-readable label for experiment whose PC data is plotted on the y-axis.
        save_path (None | str | Path): Path to save figs
        n_std (float): Number of standard deviations out to fit ellipse to. Defaults to fitting an
            ellipse whose axes are two standard deviations out from the mean.
    """

    # Create scatter plot of PC data from two experiments
    plt.clf()
    ax = plt.gca()
    ax.scatter(x, y, s=0.5, c="black", alpha=0.2)
    # Add a reference line for y=x where PCs for the two experiments are equal

    # Calculate and plot confidence ellipses for 1, 2 and 3 standard deviations
    # and project the height of the ellipses onto the y-axis to get an error value
    # for each ellipse which is provided in the figure legend
    _, _, _, slope, intercept = pc_error_from_confidence_ellipse(
        x, y, ax, n_std=n_std, edgecolor="magenta"
    )

    min_ = min(x.min(), y.min())
    max_ = max(x.max(), y.max())

    """
    model = get_linear_fit(x, y)
    m = model.coef_[0]
    print(f"Slope: {m:.2f}")
    b = model.intercept_
    y_pred = model.predict(np.array([min_, max_]).reshape(-1, 1))
    """
    y_model_min = get_linear_model(slope, intercept, min_)
    y_model_max = get_linear_model(slope, intercept, max_)

    plt.plot(
        [min_, max_], [y_model_min, y_model_max], "magenta", label=f"y={slope:.2f}x+{intercept:.2f}"
    )

    # Format, label, and save the figure
    plt.legend(loc="upper left")
    plt.xlabel(f"PC{pc} {xlabel}")
    plt.ylabel(f"PC{pc} {ylabel}")
    plt.title(f"PC{pc}")

    plt.plot([min_, max_], [min_, max_], "k--")
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path / f"paired_features_pc{pc}.png", dpi=300)
    print(f"Fig saved to {save_path}/paired_features_pc{pc}.png")
    plt.close()


def calc_and_plot_paired_PCS(
    fixed_features: pd.DataFrame,
    fixed_name: str,
    moving_features: pd.DataFrame,
    moving_name: str,
    save_path: Path,
    pca_dir: None | str | Path,
) -> None:
    """
    Plot the PCA features of the fixed and moving images.

    Args:
        fixed_features (pd.Dataframe): dataframe containing features from fine tuned model applied to fixed data
        fixed_name (str): name of fixed dataset
        moving_features (pd.Dataframe): dataframe containing features from fine tuned model applied to live data
        moving_name (str): name of live dataset
        save_path (Path): path to where to save output figures and data
        pca_dir (Path): path to load PC space from
    """
    pca = load_pca_model(str(pca_dir)) if pca_dir else fit_pca()

    fixed_features = project_manifest_to_pcs(fixed_features, pca, overwrite_feature_columns=False)
    moving_features = project_manifest_to_pcs(moving_features, pca, overwrite_feature_columns=False)

    n_pcs = len([c for c in fixed_features.columns if c.startswith("pc")])
    for pc in range(1, n_pcs + 1):

        x, y = moving_features[f"pc{pc}"].values, fixed_features[f"pc{pc}"].values
        plot_ellipse_and_fit(x, y, pc, moving_name, fixed_name, save_path)


def add_fmsid_to_config(
    prediction_path: str, dataset_name: str, mlflow_id: str, model_path: Path
) -> None:
    """
    Upload path to FMS and add the FMS ID to the dataset config file for the given dataset.

    Args:
        prediction_path (str): Path to the prediction file.
        dataset_name (str): Name of the dataset to update in config
        mlflow_id (str): MLflow ID of the model used for prediction.
        model_path (Path): Path to the model directory. Used for extracting the commit hash.
    """
    file_id = save_file_to_fms(
        prediction_path,
        dataset_name,
        get_cytodl_commit_hash(mlflow_id, model_path),
        misc_notes="",
        mlflow_run_id=mlflow_id,
    )

    # update dataset config with the FMS ID
    # of the prediction file
    dataset_config = load_single_dataset_config(dataset_name)
    dataset_config.diffae_manifest_fmsid = file_id
    save_dataset_config(dataset_config)


def compare_paired_features(
    model_name: str,
    fixed_dataset_name: str,
    moving_dataset_name: str,
    alignment_method: str,
    pca_dir: str | None,
    align_fluo: bool = True,
    align_only: bool = False,
    overrides: dict[str, Any] | None = None,
    upload_features_to_FMS: bool = False,
    **alignment_kwargs: dict[str, Any],
) -> None:
    """
    Compare the features of two paired datasets using a trained
    model through registration, crop extraction, and PCA.

    Args:
        model_name (str): The name of the trained model.
        fixed_dataset_name (str): Dataset name to use as the fixed images (i.e. the reference against which the moving images are registered)
        moving_dataset_name (str): Dataset name to use as the moving images (i.e. the images to be registered to the fixed images)
        alignment_method (str): The method used for alignment. Options are "sift" or "template". Input "sift" is recommended for the 20x
        pre/post fixation datasets, while "template" is recommended for the 20x/40x datasets.
        align_fluo(bool): Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
        pca_dir (str | None): Path to the PCA model directory. If None, PCA will be calculated from existing features
        align_only (bool): If True, only align the images and do not extract features or project to PCA.
        overrides Union[Dict, None], optional): Overrides for the model configuration, by default {}. One relevant override is
        `model.spatial_inferer.splitter.overlap`, which determines the percent overlap of patches extracted during sliding window inference
        and can increase the number of samples used for the dataset comparison.
        upload_features_to_FMS (bool): Whether to upload validation data features to FMS. We may iteratre on analysis
        without changing features and therefore should default to not rewriting a new feature manifest every time this workflow is run.
        **alignment_kwargs (Dict[str, Any]): Additional arguments for the alignment function.
    """

    # Get diffAE model
    mlflow_id = get_model_info(model_name)["mlflow_run_id"]
    model_path = Path(get_output_path(f"models/{model_name}"))
    path_dict = download_model(mlflow_id, model_path)

    # Set directory for aligned data
    save_path = model_path / f"{fixed_dataset_name}_vs_{moving_dataset_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    data_save_path = save_path / f"aligned_{fixed_dataset_name}_vs_{moving_dataset_name}.csv"

    # Align data if saved aligned data not already stored
    if not data_save_path.exists():
        data = align_all_positions(
            fixed_dataset_name,
            moving_dataset_name,
            save_path,
            alignment_method,
            align_fluo,
            **alignment_kwargs,
        )
        data["channel"] = (
            0  # channel used for inference is in the aligned images, which are single channel
        )
        data.to_csv(data_save_path, index=False)

    if align_only:
        print(
            f"Aligned images saved to {save_path}. Skipping feature extraction and PCA projection."
        )
        return

    # Apply on fixed images
    fixed_overrides = overrides.copy()  # copy to avoid overriding the original
    fixed_overrides.update({"data.predict_dataloaders.dataset.img_path_column": "fixed"})
    fixed_overrides = generate_overrides(
        fixed_overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=fixed_dataset_name,
        model_name=model_name,
    )

    # load diffAE model
    model = CytoDLModel()
    model.load_config_from_file(path_dict["config_path"])
    model.override_config(fixed_overrides)
    model.predict()

    # apply on moving images
    overrides.update({"data.predict_dataloaders.dataset.img_path_column": "moving"})
    overrides = generate_overrides(
        overrides,
        save_path=str(save_path),
        data_path=str(data_save_path),
        ckpt_path=path_dict["checkpoint_path"],
        dataset_name=moving_dataset_name,
        model_name=model_name,
    )
    model.override_config(overrides)
    model.predict()

    # compare paired features
    fixed_features_path = str(
        save_path / f"predict_{fixed_dataset_name}_{model_name}_features.parquet"
    )
    moving_features_path = str(
        save_path / f"predict_{moving_dataset_name}_{model_name}_features.parquet"
    )

    if upload_features_to_FMS:
        print("Uploading fixed and live dataset feature manifests to FMS")
        add_fmsid_to_config(
            fixed_features_path,
            fixed_dataset_name,
            mlflow_id,
            model_path,
        )
        add_fmsid_to_config(
            moving_features_path,
            moving_dataset_name,
            mlflow_id,
            model_path,
        )

    # load features for comparison
    fixed_features = pd.read_parquet(fixed_features_path)
    moving_features = pd.read_parquet(moving_features_path)

    calc_and_plot_paired_PCS(
        fixed_features,
        fixed_dataset_name,
        moving_features,
        moving_dataset_name,
        save_path,
        pca_dir,
    )


def main(
    pca_dir: str | None = None,
    fixed_finetuned_model_name: str = "diffae_finetuned_for_fixed",
    align_only: bool = False,
) -> None:
    """
    Compare paired features of fixed and moving images using a trained model.

    Args:
        pca_dir (str | None): Path to the PCA model directory. If None, PCA will be calculated from existing features
        fixed_finetuned_model_name (str): The name of the model finetuned for fixation.
        model_name (str): The name of the model to use for comparison.
        align_only (bool): If True, only align the images and do not extract features or project to PCA. Defaults to False.
    """

    overrides = {"model.spatial_inferer.splitter.overlap": 0.9}

    compare_paired_features(
        fixed_finetuned_model_name,
        "20250214_pairedPostFixation",
        "20250214_pairedPreFixation",
        alignment_method="sift",
        pca_dir=pca_dir,
        overrides=overrides,
        align_only=align_only,
    )


if __name__ == "__main__":
    fire.Fire(main)
