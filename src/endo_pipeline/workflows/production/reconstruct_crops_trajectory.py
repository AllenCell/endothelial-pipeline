from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["diffae_image_generation", "diffae_features"]


def main(
    csv_path: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
) -> None:
    """
    Reconstruct crops from PC space coordinates stored in a given CSV file.

    The reconstructed crops are saved as PNG files in a local directory.

    Parameters
    ----------
    csv_path
        Path to a CSV file containing latent space coordinates along trajectories.
    model_manifest_name
        Name of the model manifest containing the specific run to load features from.
    run_name
        Run name corresponding to features to load and the model to use for image reconstruction.
    """
    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.io import get_output_path, load_model, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.model import generate_from_coords_batch
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_model_manifest,
    )

    logger = logging.getLogger(__name__)

    # convert csv_path to Path object
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        logger.error("CSV file [ %s ] does not exist.", csv_path)
        raise FileNotFoundError(f"CSV file [ {csv_path} ] does not exist.")

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)

    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )

    # Get fit (3D) PCA object from manifest
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    # Directory to save reconstructed crops
    csv_file_name = csv_path_obj.stem
    crop_savedir = get_output_path(
        "reconstructed_crops", model_manifest_name, run_name_, csv_file_name
    )

    # load coordinates from csv
    pc_coords = np.loadtxt(csv_path_obj, delimiter=",")

    # make sure that coords is a 2D array with shape (num_points, num_dimensions)
    pc_coords = np.atleast_2d(pc_coords)
    if pc_coords.shape[1] != 3:
        logger.debug("Transposing input coordinates array for correct shape.")
        pc_coords = pc_coords.T

    # transform interpolated points to full latent space
    latent_coords = pca.inverse_transform(pc_coords)

    walk_imgs = generate_from_coords_batch(model, latent_coords, num_gpus=NUM_GPUS)

    for i, img in enumerate(walk_imgs):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        save_plot_to_path(fig, crop_savedir, f"coordinate_row_{i:03d}")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
