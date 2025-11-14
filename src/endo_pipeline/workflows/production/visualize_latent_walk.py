from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    list_of_axes: list[int] | None = None,
    sigma: float = 3.0,
    n_steps: int = 10,
    use_pcs: bool = True,
    show_coords: bool = False,
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
    import pandas as pd
    from bioio.writers import OmeTiffWriter

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.model import generate_from_coords, get_walk, write_pc_vals
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import NUM_PCS_TO_ANALYZE, ColumnName

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)

    # set up output directory
    save_path = get_output_path(
        "latent_walks",
        model_manifest_name,
        run_name_,
        "include_cell_piling" if include_cell_piling else "exclude_cell_piling",
    )

    # load model configuration and reference dataset manifests
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection("pca_reference")

    if list_of_axes is None:
        list_of_axes = list(range(NUM_PCS_TO_ANALYZE))

    if use_pcs:
        # Perform latent walk along the principal component axes.
        # Because we have to do the inverse transform later, we need to fit PCA on all PCs
        # up to the max PC index specified, perform the walk, then inverse transform back.
        # Then we can select only the specified PCs from the walk for image generation
        max_num_pcs = max(list_of_axes) + 1
        pca = fit_pca(
            dataframe_manifest_name=dataframe_manifest_name,
            include_cell_piling=include_cell_piling,
            num_pcs=max_num_pcs,
        )
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca)
                for dataset_name in dataset_names
            ]
        )
        pc_column_names = [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(max_num_pcs)]
        data_for_walk = dataframe[pc_column_names].values
        walk, ranges = get_walk(data_for_walk, sigma, n_steps, list_of_axes=list_of_axes)
        # inverse transform to original latent space
        # need this for image generation
        walk = pca.inverse_transform(walk)
    else:
        # Perform latent walk along the raw latent dimensions.
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(
                    dataset_name,
                    dataframe_manifest,
                    pca=None,
                    include_cell_piling=include_cell_piling,
                )
                for dataset_name in dataset_names
            ]
        )
        feature_column_names = [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in list_of_axes]
        data_for_walk = dataframe[feature_column_names].values
        walk, ranges = get_walk(data_for_walk, sigma, n_steps, list_of_axes=list_of_axes)

    # generate images from the latent walk
    walk_img = generate_from_coords(model, walk, n_noise_samples=n_noise_samples, num_gpus=NUM_GPUS)

    # vertically stack multi-channel generations
    walk_img_stack = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])
    if show_coords:
        walk_img_stack = write_pc_vals(walk_img_stack, ranges)

    axis_label = "_pcs" if use_pcs else "_dims"
    axes_str = "_".join([str(ax + 1) for ax in list_of_axes])
    file_name = f"latent_walk_sigma_{int(sigma)}_along{axis_label}_{axes_str}"
    OmeTiffWriter.save(
        uri=save_path / f"{file_name}.tif",
        data=walk_img_stack,
    )

    # also plot the latent walk as a grid and save
    # reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = len(ranges)
    n_steps_actual = ranges[0].shape[0]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]
    walk_img_grid = walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)

    plot_latent_walk_as_grid(walk_img_grid, ranges, list_of_axes, use_pcs, save_path, file_name)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
