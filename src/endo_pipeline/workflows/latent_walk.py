# MODIFIED FROM https://github.com/AllenCellModeling/cyto-dl/blob/08c6aadb5da54ef7d186d82b71bf8473c5e0e814/cyto_dl/callbacks/latent_walk_diffae.py#L16
from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from bioio.writers import OmeTiffWriter
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import get_pca_reference_model_manifests
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest.diffae_manifest_utils import (
    get_feature_column_names,
    get_pc_column_names,
)
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.model.diffae.generate_image import generate_from_coords


def write_text(img: np.ndarray, text: str) -> np.ndarray:
    """Write text on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = tuple([img.max()] * 3)
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - 3  # 3 pixels from the right edge
    text_y = text_size[1] + 3  # 3 pixels from the top edge
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img


def write_pc_vals(walk_img: np.ndarray, ranges: list) -> np.ndarray:
    """Write dimension index and value on image."""
    idx = 0
    for i, range_ in enumerate(ranges):
        for val in range_:
            walk_img[idx] = write_text(walk_img[idx], f"{i+1}:{val:.1f}")
            idx += 1
    return walk_img


def get_walk(data: np.ndarray, n_dims: int, sigma: float, n_steps: int) -> tuple[np.ndarray, list]:
    """
    Generate a latent walk based on standard deviation
    or min/max of each dimension.

    Parameters
    ----------
    data: np.ndarray
        Numpy array containing the data to be traversed.
    n_dims: int
        Number of dimensions for the latent walk.
    sigma: float
        Range of values for the latent walk.
    n_steps: int
        Number of steps in the latent walk.
    """
    walk = []
    ranges = []
    for dim in range(n_dims):
        if sigma is None:
            data_min = data[:, dim].min()
            data_max = data[:, dim].max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            std = data[:, dim].std()
            range_ = np.arange(-sigma, sigma + 0.01) * std
        print(f"Dim {dim} range: {range_}")
        dim_traversal = np.stack([data.mean(axis=0)] * range_.shape[0])
        dim_traversal[:, dim] = range_
        walk.append(dim_traversal)
        ranges.append(range_)
    walk = np.concatenate(walk).squeeze()
    return walk, ranges


def get_pca_coords(
    data: np.ndarray, pca: Pipeline, num_pcs: int, sigma: float, n_steps: int
) -> tuple[np.ndarray, list]:
    """
    Generate PCA coordinates and corresponding PC values for a latent walk.

    Parameters
    ----------
    data: np.ndarray
        Numpy array containing the data to be transformed.
    pca: PCA
        PCA object fitted to the data.
    num_pcs: int
        Number of principal components to use for the latent walk.
    sigma: float
        Range of values for the latent walk.
    n_steps: int
        Number of steps in the latent walk.
    """
    pca_data = pca.transform(data)
    walk, ranges = get_walk(pca_data, num_pcs, sigma, n_steps)
    walk = pca.inverse_transform(walk)
    return walk, ranges


def get_latent_coords(data: np.ndarray, sigma: float, n_steps: int) -> tuple[np.ndarray, list]:
    """
    Generate latent coordinates and corresponding values for a latent walk.

    Parameters
    ----------
    data: np.ndarray
        Numpy array containing the data to be transformed.
    sigma: float
        Range of values for the latent walk.
    n_steps: int
        Number of steps in the latent walk.
    """
    n_dims = data.shape[1]
    walk, ranges = get_walk(data, n_dims, sigma, n_steps)
    return walk, ranges


def main(
    model_name: str,
    num_pcs: int = 3,
    sigma: float = 3.0,
    n_steps: int = 10,
    use_pcs: bool = True,
    show_coords: bool = True,
    n_noise_samples: int = 1,
) -> None:
    """
    Create latent walk for a given model using PCA or model features.

    Example usage:
    ```
    uv run src/endo_pipeline/workflows/latent_walk.py
        --model_name diffae_04_10 --num_pcs 3 --sigma 3.0
        --n_steps 10 --use_pcs True --show_coords True
    ```

    Parameters
    ----------
    model_name: str
        Name of the model to use for generating the latent walk.
    num_pcs: int, optional
        Number of principal components to use for the
        latent walk. Default is 3.
    sigma: float, optional
        Number of standard deviations from the mean to traverse
        for the latent walk. Default is 3.0. If passing `sigma=None`,
        the min and max of the range are used as endpoints for the walk.
    n_steps: int, optional
        Number of steps in the latent walk. Default is 10.
    use_pcs: bool, optional
        Whether to use PCA for generating the latent walk.
        If False, the raw latent dimensions are used. Default is True.
    show_coords: bool, optional
        Whether to show the dimension value to generate a
        given image. Default is True.
    n_noise_samples: int, optional
        Number of noise samples to use for generating images.
        Default is 1.
    """
    save_dir = get_output_path("models", model_name, include_timestamp=False)

    pca = fit_pca()

    reference_dataset_model_manifests = get_pca_reference_model_manifests(model_name)

    if use_pcs:
        # perform latent walk along the principal components
        pca = fit_pca()
        manifest_dataframe = pd.concat(
            [
                get_manifest_for_dynamics_workflows(model_manifest, pca)
                for model_manifest in reference_dataset_model_manifests
            ]
        )
        pc_column_names = get_pc_column_names(manifest_dataframe, pc_axes=list(range(num_pcs)))
        data_for_walk = manifest_dataframe[pc_column_names].values
        walk, ranges = get_pca_coords(data_for_walk, pca, num_pcs, sigma, n_steps)
    else:
        # perform latent walk along the raw latent dimensions
        manifest_dataframe = pd.concat(
            [
                get_manifest_for_dynamics_workflows(model_manifest, pca=None)
                for model_manifest in reference_dataset_model_manifests
            ]
        )
        feature_column_names = get_feature_column_names(manifest_dataframe)
        data_for_walk = manifest_dataframe[feature_column_names].values
        walk, ranges = get_latent_coords(data_for_walk, sigma, n_steps)

    # generate images from the latent walk
    walk_img = generate_from_coords(model_name, walk, n_noise_samples=n_noise_samples)

    # vertically stack multi-channel generations
    walk_img = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])
    if show_coords:
        walk_img = write_pc_vals(walk_img, ranges)

    save_path = Path(save_dir) / f"latent_walk_sigma_{sigma}_use_pcs_{use_pcs}.tif"
    OmeTiffWriter.save(
        uri=save_path,
        data=walk_img,
    )


if __name__ == "__main__":
    fire.Fire(main)
