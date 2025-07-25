TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_name: str = "diffae_04_10",
    num_pcs: int = 3,
    sigma: float = 3.0,
    n_steps: int = 10,
    use_pcs: bool = True,
    show_coords: bool = True,
    n_noise_samples: int = 1,
) -> None:
    """
    Create latent walk for a given model using PC axes or the original latent space axes.

    Parameters
    ----------
    model_name
        Name of the model to use for generating the latent walk.
    num_pcs
        Number of principal components to use for the
        latent walk.
    sigma
        Number of standard deviations from the mean to traverse
        for the latent walk.
    n_steps
        Number of steps in the latent walk. Default is 10.
    use_pcs
        Whether to use PCA for generating the latent walk.
    show_coords
        Whether to show the dimension value to generate a
        given image.
    n_noise_samples
        Number of noise samples to use for generating images.

    Returns
    -------
    :
        Saves the latent walk images to the output directory.
        The images are saved as a multi-channel TIFF file.
    """
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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
