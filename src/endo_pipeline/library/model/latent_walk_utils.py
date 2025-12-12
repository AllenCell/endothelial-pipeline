from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.library.model.diffae import DiffusionAutoEncoder, generate_from_coords

if TYPE_CHECKING:
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import ColumnName


def get_latent_walk(
    data: np.ndarray, n_dims: int, sigma: float | None, n_steps: int
) -> tuple[np.ndarray, list]:
    """
    Generate a latent walk based on standard deviation or min/max of each
    dimension.

    Parameters
    ----------
    data
        Numpy array containing the data to be traversed.
    n_dims
        Number of dimensions for the latent walk.
    sigma
        Range of values for the latent walk. If None, use min/max of dimension.
    n_steps
        Number of steps in the latent walk.
    """

    walks: list[np.ndarray] = []
    ranges: list[np.ndarray] = []

    for dim in range(n_dims):
        if sigma is None:
            data_min = data[:, dim].min()
            data_max = data[:, dim].max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            std = data[:, dim].std()
            range_ = np.arange(-sigma, sigma + 0.01) * std

        dim_traversal = np.stack([data.mean(axis=0)] * range_.shape[0])
        dim_traversal[:, dim] = range_
        walks.append(dim_traversal)
        ranges.append(range_)

    walk = np.concatenate(walks).squeeze()

    return walk, ranges


def build_data_for_pca_latent_walk(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    include_cell_piling: bool,
    crop_pattern: Literal["grid", "tracked"],
) -> np.ndarray:
    """
    Build data array for latent walk on data projected onto PCA axes.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in array.
    dataframe_manifest
        Manifest for dataframes containing feature data
    pca
        PCA model to fit to feature data.
    include_cell_piling
        True keep timepoints annotated as cell piling, False otherwise.
    crop_pattern
        Crop pattern used to generate the feature dataframe.

    Returns
    -------
    :
        Combined data array projected onto PCA axes.
    """

    column_names = [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(pca.n_components_)]
    dataframe = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                dataset_name,
                dataframe_manifest,
                pca,
                include_cell_piling=include_cell_piling,
                crop_pattern=crop_pattern,
            )
            for dataset_name in dataset_names
        ]
    )

    return dataframe[column_names].values


def build_data_for_raw_latent_walk(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    model,
    include_cell_piling: bool,
    crop_pattern: Literal["grid", "tracked"],
) -> np.ndarray:
    """
    Build data array for latent walk on raw feature data.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in array.
    dataframe_manifest
        Manifest for dataframes containing feature data
    include_cell_piling
        True keep timepoints annotated as cell piling, False otherwise.
    crop_pattern
        Crop pattern used to generate the feature dataframe.

    Returns
    -------
    :
        Combined data array.
    """

    num_latent_dims = model.semantic_encoder.base_encoder.num_classes
    column_names = [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(num_latent_dims)]

    dataframe = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                dataset_name,
                dataframe_manifest,
                pca=None,
                include_cell_piling=include_cell_piling,
                crop_pattern=crop_pattern,
            )
            for dataset_name in dataset_names
        ]
    )

    return dataframe[column_names].values


def get_pca_latent_walk(
    pca_data: np.ndarray, pca: PCA, sigma: float | None, n_steps: int
) -> tuple[np.ndarray, list]:
    """
    Generate PCA coordinates and corresponding PC values for a latent walk.

    Parameters
    ----------
    pca_data
        Array containing the data projected onto PCA axes for the latent walk.
    pca
        PCA model fit to the data.
    sigma
        Range of values for the latent walk. If None, use min/max of dimension.
    n_steps
        Number of steps in the latent walk.
    """

    n_dims = pca.n_components_
    walk, ranges = get_latent_walk(pca_data, n_dims, sigma, n_steps)
    walk = pca.inverse_transform(walk)
    return walk, ranges


def get_raw_latent_walk(
    data: np.ndarray, sigma: float | None, n_steps: int
) -> tuple[np.ndarray, list]:
    """
    Generate latent coordinates and corresponding values for a latent walk.

    Parameters
    ----------
    data
        Array containing the data for the latent walk.
    sigma
        Range of values for the latent walk. If None, use min/max of dimension.
    n_steps
        Number of steps in the latent walk.
    """

    n_dims = data.shape[1]
    walk, ranges = get_latent_walk(data, n_dims, sigma, n_steps)
    return walk, ranges


def generate_latent_walk_images(
    model: "DiffusionAutoEncoder",
    walk: np.ndarray,
    ranges: list,
    n_noise_samples: int = 1,
    num_gpus: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    # Ggenerate images from the latent walk
    walk_img = generate_from_coords(
        model, walk, n_noise_samples=n_noise_samples, num_gpus=num_gpus, random_seed=random_seed
    )

    # Reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = len(ranges)
    n_steps_actual = ranges[0].shape[0]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]

    return walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)
