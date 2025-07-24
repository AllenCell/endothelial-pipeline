# MODIFIED FROM https://github.com/AllenCellModeling/cyto-dl/blob/08c6aadb5da54ef7d186d82b71bf8473c5e0e814/cyto_dl/callbacks/latent_walk_diffae.py#L16
from pathlib import Path

import pandas as pd
from bioio.writers import OmeTiffWriter

from src.endo_pipeline.configs import get_pca_reference_model_manifests, load_model_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_feature_column_names,
    get_manifest_for_dynamics_workflows,
    get_pc_column_names,
)
from src.endo_pipeline.library.model import (
    generate_from_coords,
    get_latent_coords,
    get_pca_coords,
    write_pc_vals,
)


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
    # set up output directory
    save_dir = get_output_path("models", model_name, include_timestamp=False)

    # load model configuration and reference dataset manifests
    model_config = load_model_config(model_name)
    reference_dataset_model_manifests = get_pca_reference_model_manifests(model_config)

    if use_pcs:
        # perform latent walk along the principal components
        pca = fit_pca(model_name=model_name, num_pcs=num_pcs)
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
