from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    NUM_PCS_TO_ANALYZE,
)

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
    sigma: float = 3.0,
    n_steps: int = 7,
    use_pcs: bool = True,
    n_noise_samples: int = 1,
) -> None:
    """
    Create latent walk for a given model using PC axes or original axes.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest containing the specific run to load.
    run_name
        Run name corresponding to the model to load. If None, uses the most recent run.
    crop_pattern
        Crop pattern used to generate the feature dataframe. Either 'grid' or 'tracked'.
    include_cell_piling
        True to include timepoints with cell piling to fit the PCA model, False to exclude them.
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
    n_noise_samples
        Number of noise samples to use for generating images.
    """

    import logging

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.model import (
        build_data_for_pca_latent_walk,
        build_data_for_raw_latent_walk,
        generate_latent_walk_images,
        get_pca_latent_walk,
        get_raw_latent_walk,
    )
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    logger = logging.getLogger(__name__)

    if crop_pattern not in ["tracked", "grid"]:
        logger.error("Crop pattern must be 'tracked' or 'grid', got [ %s ]", crop_pattern)
        raise ValueError("Input crop_pattern must be 'grid' or 'tracked'")

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)
    assert isinstance(model, DiffusionAutoEncoder)

    # set up output directory
    save_path = get_output_path(
        "latent_walks",
        model_manifest_name,
        run_name_,
        crop_pattern,
        dataset_collection,
        "include_cell_piling" if include_cell_piling else "exclude_cell_piling",
    )

    # load model configuration and reference dataset manifests
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection(dataset_collection)

    if use_pcs:
        # perform latent walk along the principal components
        pca = fit_pca(
            dataset_collection_name=dataset_collection,
            dataframe_manifest_name=dataframe_manifest_name,
            include_cell_piling=include_cell_piling,
            num_pcs=num_pcs,
        )
        data_for_walk = build_data_for_pca_latent_walk(
            dataset_names, dataframe_manifest, pca, include_cell_piling, crop_pattern
        )
        walk, ranges = get_pca_latent_walk(data_for_walk, pca, sigma, n_steps)
    else:
        # perform latent walk along the raw latent dimensions
        data_for_walk = build_data_for_raw_latent_walk(
            dataset_names, dataframe_manifest, model, include_cell_piling, crop_pattern
        )
        walk, ranges = get_raw_latent_walk(data_for_walk, sigma, n_steps)

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(model, walk, ranges, n_noise_samples, NUM_GPUS)

    # save generated latent walk as grid
    axis_suffix = "_along_pcs" if use_pcs else "_along_latent"
    file_name = f"latent_walk_{int(sigma)}sigma{axis_suffix}"
    plot_latent_walk_as_grid(walk_img_grid, ranges, save_path, file_name, use_pcs)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
