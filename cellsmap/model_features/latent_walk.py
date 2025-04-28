# MODIFIED FROM https://github.com/AllenCellModeling/cyto-dl/blob/08c6aadb5da54ef7d186d82b71bf8473c5e0e814/cyto_dl/callbacks/latent_walk_diffae.py#L16
import fire
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from bioio.writers import OmeTiffWriter
from typing import Optional, Tuple, List

from cellsmap.util.dataset_io import get_reference_datasets
from cellsmap.util.manifest_io import get_diffae_manifest, load_pca_model, save_pca_model
from cellsmap.util.manifest_pca import get_feature_cols, fit_pca
from cellsmap.model_features.generate_image import generate_from_coords
from cellsmap.util.set_output import get_output_path


def write_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = tuple([img.max()] * 3)
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - 3  # 3 pixels from the right edge
    text_y = text_size[1] + 3  # 3 pixels from the top edge
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img

def write_pc_vals(walk_img, ranges):
    """Write PC index and value on image."""
    idx = 0
    for i, range_ in enumerate(ranges):
        for val in range_:
            walk_img[idx] = write_text(walk_img[idx], f"PC{i+1}:{val:.1f}")
            idx += 1
    return walk_img

def get_walk(data, n_dims, sigma, n_steps):
    """
    Generate a latent walk based on standard deviation or min/max of each dimension
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the data to be transformed.
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
        std = data[:, dim].std()
        if sigma is None:
            min = data[:, dim].min() / std
            max = data[:, dim].max() / std
            range_ = np.linspace(min, max, n_steps)
        else:
            range_ = np.arange(-sigma, sigma + 0.01)
        print(f"PC{dim} range: {range_}")
        for i in range_:
            array = np.zeros(n_dims)
            array[dim] = i * std
            walk.append(array)
        ranges.append(range_)
    walk = np.stack(walk).squeeze()
    return walk, ranges

def get_pca_coords(data, pca, num_pcs, sigma, n_steps) -> Tuple[List, List]:
    """
    Generate PCA coordinates and corresponding PC values for a latent walk.
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the data to be transformed.
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

def get_latent_coords(data, sigma, n_steps) -> Tuple[List, List]:
    """
    Generate latent coordinates and corresponding values for a latent walk.
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the data to be transformed.
    sigma: float
        Range of values for the latent walk.
    n_steps: int
        Number of steps in the latent walk.
    """
    n_dims = data.shape[1]
    walk, ranges = get_walk(data, n_dims, sigma, n_steps)
    return walk, ranges

def generate_latent_walk(model_name:str, pca_dir: Optional[str] = None, num_pcs: int = 3, sigma: float = 3.0, n_steps: int = 10, use_pcs: bool = True, show_coords: bool = True):
    """
    Create latent walk for a given model using PCA or model features

    Parameters
    ----------
    model_name: str
        Name of the model to use for generating the latent walk.
    pca_dir: str, optional
        Directory to load the PCA model from. If not provided, a new PCA model will be fitted.
    num_pcs: int, optional
        Number of principal components to use for the latent walk. Default is 3.
    sigma: float, optional
        Number of standard deviations from the mean to traverse for the latent walk. Default is 3.0.
    n_steps: int, optional
        Number of steps in the latent walk. Default is 10.
    use_pcs: bool, optional
        Whether to use PCA for generating the latent walk. If False, the raw latent dimensions are used. Default is True.
    show_coords: bool, optional
        Whether to show the dimension value to generate a given image. Default is True.    
    """
    save_dir = get_output_path(f"models/{model_name}")

    reference_manifests = pd.concat([get_diffae_manifest(name) for name in get_reference_datasets()])
    feature_cols = get_feature_cols(reference_manifests)
    data = reference_manifests[feature_cols].values
    if use_pcs: 
        # use fitted PCA if path to one is passed, otherwise fit a new one on the reference dataset
        if pca_dir is None:
            pca = fit_pca(reference_manifests, num_pcs=num_pcs)
            save_pca_model(pca, save_dir)
        else:
            pca = load_pca_model(pca_dir)
        walk, ranges = get_pca_coords(data, pca, num_pcs, sigma, n_steps)
    else:
        walk, ranges = get_latent_coords(data, sigma, n_steps)

    walk_img = generate_from_coords(model_name, walk)

    # vertically stack multi-channel generations
    walk_img = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])
    if show_coords:
        walk_img = write_pc_vals(walk_img, ranges)
    
    OmeTiffWriter.save(uri=Path(save_dir)/f'latent_walk_sigma_{sigma}_use_pcs_{use_pcs}.tif', data=walk_img)


if  __name__ == '__main__':
    fire.Fire(generate_latent_walk)