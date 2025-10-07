TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
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
    model_manifest_name
        Name of the model manifest containing the specific run to load.
    run_name
        Run name corresponding to the model to load. If None, uses the most recent run.
    num_pcs
        Number of principal components to use for the
        latent walk.
    sigma
        Number of standard deviations from the mean to traverse
        for the latent walk.
    n_steps
        Number of steps in the latent walk. Default is 10.
    use_pcs
        True to use principal component axes, False to use original latent space axes.
    show_coords
        True to write the coordinate value used generate a given image, False to not.
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

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_feature_column_names,
        get_pc_column_names,
    )
    from endo_pipeline.library.model import (
        generate_from_coords,
        get_latent_coords,
        get_pca_coords,
        write_pc_vals,
    )
    from endo_pipeline.manifests import (
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_])

    # set up output directory
    save_dir = get_output_path("models", model_manifest_name, run_name_)

    # load model configuration and reference dataset manifests
    if model_manifest_name == "diffae_04_10":
        dataframe_manifest_name = "diffae_04_10"
    else:
        dataframe_manifest_name = f"{model_manifest_name}_{run_name_}_grid"
    manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection("pca_reference")

    if use_pcs:
        # perform latent walk along the principal components
        pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=num_pcs)
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(dataset_name, manifest, pca)
                for dataset_name in dataset_names
            ]
        )
        pc_column_names = get_pc_column_names(dataframe, pc_axes=list(range(num_pcs)))
        data_for_walk = dataframe[pc_column_names].values
        walk, ranges = get_pca_coords(data_for_walk, pca, num_pcs, sigma, n_steps)
    else:
        # perform latent walk along the raw latent dimensions
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(dataset_name, manifest, pca=None)
                for dataset_name in dataset_names
            ]
        )
        feature_column_names = get_feature_column_names(dataframe)
        data_for_walk = dataframe[feature_column_names].values
        walk, ranges = get_latent_coords(data_for_walk, sigma, n_steps)

    # generate images from the latent walk
    walk_img = generate_from_coords(model, walk, n_noise_samples=n_noise_samples, num_gpus=NUM_GPUS)

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
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
