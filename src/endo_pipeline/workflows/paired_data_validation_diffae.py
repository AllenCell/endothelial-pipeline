from pathlib import Path
from typing import Any, Dict, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cyto_dl.api import CytoDLModel

from cellsmap.util.dataset_io import get_model_info, update_dataset_config
from cellsmap.util.manifest_io import load_pca_model
from cellsmap.util.manifest_preprocessing import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import project_manifest_to_pcs
from src.endo_pipeline.library.model.apply_model import get_cytodl_commit_hash
from src.endo_pipeline.library.model.mlflow import download_model
from src.endo_pipeline.library.process.registration import align_all_positions
from src.endo_pipeline.workflows.apply_diffae_model import generate_overrides


def plot_paired_features(
    fixed_features: pd.DataFrame,
    fixed_name: str,
    moving_features: pd.DataFrame,
    moving_name: str,
    save_path: Path,
    pca_dir: None | Union[str, Path],
) -> None:
    """
    Plot the PCA features of the fixed and moving images
    """
    pca = load_pca_model(str(pca_dir)) if pca_dir else fit_pca()

    fixed_features = project_manifest_to_pcs(fixed_features, pca, overwrite_feature_columns=False)
    moving_features = project_manifest_to_pcs(moving_features, pca, overwrite_feature_columns=False)

    n_pcs = len([c for c in fixed_features.columns if c.startswith("pc")])

    fig, ax = plt.subplots(1, n_pcs, figsize=(n_pcs * 4, 4))
    for i in range(n_pcs):
        r = np.corrcoef(fixed_features[f"pc{i+1}"], moving_features[f"pc{i+1}"])[0, 1]
        ax[i].scatter(fixed_features[f"pc{i+1}"], moving_features[f"pc{i+1}"], alpha=0.1, s=3)
        ax[i].set_xlabel(fixed_name)
        ax[i].set_ylabel(moving_name)
        ax[i].set_title(f"PC{i+1} r^2: {r**2:.2f}", fontsize=6)
        min_ = min(fixed_features[f"pc{i+1}"].min(), moving_features[f"pc{i+1}"].min())
        max_ = max(fixed_features[f"pc{i+1}"].max(), moving_features[f"pc{i+1}"].max())
        ax[i].plot([min_, max_], [min_, max_], "r--")
        ax[i].set_xlim(min_, max_)
        ax[i].set_ylim(min_, max_)
        ax[i].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(save_path / f"paired_features.png", dpi=300)
    fig.clf()
    plt.close(fig)


def add_fmsid_to_config(
    prediction_path: str, dataset_name: str, mlflow_id: str, model_path: Path
) -> None:
    """
    Upload path to FMS and add the FMS ID to the dataset config file for the given dataset.

    Parameters
    ----------
    prediction_path : str
        Path to the prediction file.
    dataset_name : str
        Name of the dataset to update in config
    mlflow_id : str
        MLflow ID of the model used for prediction.
    model_path : Path
        Path to the model directory. Used for extracting the commit hash.
    """
    file_id = save_file_to_fms(
        prediction_path,
        dataset_name,
        get_cytodl_commit_hash(mlflow_id, model_path),
        misc_notes="",
        mlflow_run_id=mlflow_id,
    )

    update_dataset_config(
        dataset_name,
        {"diffae_manifest_fmsid": file_id},
    )


def compare_paired_features(
    model_name: str,
    fixed_dataset_name: str,
    moving_dataset_name: str,
    alignment_method: str,
    pca_dir: str | None,
    align_fluo: bool = True,
    align_only: bool = False,
    overrides: Dict[str, Any] = {},
    **alignment_kwargs: Dict[str, Any],
) -> None:
    """
    Compare the features of two paired datasets using a trained model through registration, crop extraction, and PCA

    Parameters
    ----------
    model_name : str
        The name of the trained model.
    fixed_dataset_name : str
        Dataset name to use as the fixed images (i.e. the reference against which the moving images are registered)
    moving_dataset_name : str
        Dataset name to use as the moving images (i.e. the images to be registered to the fixed images)
    alignment_method : str
        The method used for alignment. Options are "sift" or "template". "sift" is recommended for the 20x pre/post fixation datasets, while "template" is recommended for the 20x/40x datasets.
    align_fluo : bool
        Whether to align the fluorescent channel. If False, the fluorescent channel is not aligned.
    pca_dir : str | None
        Path to the PCA model directory. If None, PCA will be calculated from existing features
    overrides : Union[str, Dict], optional
        Overrides for the model configuration, by default {}. One relevant override is `model.spatial_inferer.splitter.overlap`, which determines the percent overlap of patches extracted during sliding window inference and can increase the number of samples used for the dataset comparison.
    **alignment_kwargs : Dict[str, Any]
        Additional arguments for the alignment function.
    """
    mlflow_id = get_model_info(model_name)["mlflow_run_id"]
    model_path = Path(get_output_path(f"models/{model_name}"))
    path_dict = download_model(mlflow_id, model_path)

    save_path = model_path / f"{fixed_dataset_name}_vs_{moving_dataset_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    data_save_path = save_path / f"aligned_{fixed_dataset_name}_vs_{moving_dataset_name}.csv"

    if not data_save_path.exists():
        data = align_all_positions(
            fixed_dataset_name,
            moving_dataset_name,
            save_path,
            alignment_method,
            align_fluo,
            **alignment_kwargs,
        )
        # channel used for inference is in the aligned images, which are single channel
        data["channel"] = 0
        data.to_csv(data_save_path, index=False)

    if align_only:
        print(
            f"Aligned images saved to {save_path}. Skipping feature extraction and PCA projection."
        )
        return

    # apply on fixed images
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

    # load model
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
    add_fmsid_to_config(
        fixed_features_path,
        fixed_dataset_name,
        mlflow_id,
        model_path,
    )
    moving_features_path = str(
        save_path / f"predict_{moving_dataset_name}_{model_name}_features.parquet"
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

    plot_paired_features(
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
    model_name: str = "diffae_04_10",
    align_only: bool = False,
) -> None:
    """ "
    Main function to compare paired features of fixed and moving images using a trained model.
    Parameters
    ----------
    pca_dir : str | None
        Path to the PCA model directory. If None, PCA will be calculated from existing features"
    """
    overrides = {"model.spatial_inferer.splitter.overlap": 0.9}

    datasets_live_fixed = {
        "fixed": [
            "20250214_pairedPreFixation",
        ],
        "moving": [
            "20250214_pairedPostFixation",
        ],
    }
    for fixed, moving in zip(datasets_live_fixed["fixed"], datasets_live_fixed["moving"]):
        compare_paired_features(
            # use model finetuned for fixation
            fixed_finetuned_model_name,
            fixed,
            moving,
            alignment_method="sift",
            pca_dir=pca_dir,
            overrides=overrides,
            align_only=align_only,
        )

    datasets_20x_40x = {
        "fixed": ["20250110_paired20X", "20250227_paired20X", "20250228_paired20X"],
        "moving": ["20250110_paired40X", "20250227_paired40X", "20250228_paired40X"],
    }
    for fixed, moving in zip(datasets_20x_40x["fixed"], datasets_20x_40x["moving"]):
        compare_paired_features(
            model_name,
            fixed,
            moving,
            alignment_method="template",
            pca_dir=pca_dir,
            overrides=overrides,
            align_only=align_only,
        )


if __name__ == "__main__":
    fire.Fire(main)
