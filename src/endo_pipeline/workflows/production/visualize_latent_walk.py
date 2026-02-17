from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, OptionalFloatList, StrList
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
    crop_pattern: CropPattern = "grid",
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    n_dims: int = NUM_PCS_TO_ANALYZE,
    columns: StrList | None = None,
    replace_mean_with_value: OptionalFloatList | None = None,
    sigma: float = 3.0,
    n_steps: int = 7,
    use_pcs: bool = True,
    use_polar: bool = False,
    n_noise_samples: int = 1,
) -> None:
    """
    Create latent walk for a given model using PC axes or original axes.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest containing the specific run to load.
    run_name
        Run name corresponding to the model to load. If None, uses the most
        recent run.
    crop_pattern
        Crop pattern used to generate the feature dataframe. Either 'grid' or
        'tracked'.
    include_cell_piling
        True to include timepoints with cell piling to fit the PCA model, False
        to exclude them (``--exclude-cell-piling``).
    n_dims
        Number of axes to use for the latent walk (either PCs or original latent
        dimensions, depending on use_pcs).
    columns
        List of column names to use for the latent walk. If None, defaults to
        using all PCs or all latent dimensions as specified by n_dims.
    replace_mean_with_value
        List of values to replace the mean with for each dimension. If None,
        uses the mean of the data.
    sigma
        Number of standard deviations from the mean to traverse for the latent
        walk.
    n_steps
        Number of steps in the latent walk. Default is 10.
    use_pcs
        True to use principal component axes, False to use original latent space
        axes.
    n_noise_samples
        Number of noise samples to use for generating images.
    """
    import logging

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca, polar_to_pcs
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.latent_walk_utils import (
        generate_latent_walk_images,
        get_dataframe_for_latent_walk,
        get_latent_walk,
    )
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import ColumnName

    logger = logging.getLogger(__name__)

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)
    if not isinstance(model, DiffusionAutoEncoder):
        raise ValueError(f"Expected model of type DiffusionAutoEncoder, got {type(model)}.")

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
            num_pcs=n_dims,
        )
        if use_polar:
            column_names = (
                [f"{ColumnName.POLAR_ANGLE}", f"{ColumnName.POLAR_RADIUS}"]
                if columns is None
                else columns
            )
            compute_polar = True
            n_dims = len(column_names)
        else:
            column_names = (
                [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(n_dims)]
                if columns is None
                else columns
            )
            compute_polar = False

    else:
        # perform latent walk along the raw latent dimensions
        pca = None
        compute_polar = False
        column_names = (
            [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(n_dims)]
            if columns is None
            else columns
        )

    # get dataframe for getting the standard dev. based latent walk
    try:
        data_for_walk = get_dataframe_for_latent_walk(
            dataset_names,
            dataframe_manifest,
            pca,
            include_cell_piling,
            crop_pattern,
            column_names,
            compute_polar=compute_polar,
        )
    except KeyError:
        logger.error(
            "Passed in an invalid column name for latent walk dataframe; column names specified: [ %s ].",
            ", ".join(column_names),
        )
        raise

    # get latent walk
    walk, ranges = get_latent_walk(data_for_walk, n_dims, sigma, n_steps, replace_mean_with_value)
    if use_polar:
        # have to manually convert from polar to cartesian coordinates before
        # applying inverse PCA transformation
        walk_polar = walk.copy()
        polar_angle_idx = column_names.index(f"{ColumnName.POLAR_ANGLE}")
        polar_radius_idx = column_names.index(f"{ColumnName.POLAR_RADIUS}")
        pcs_from_polar = polar_to_pcs(
            walk_polar[:, polar_radius_idx], walk_polar[:, polar_angle_idx]
        )
        walk[:, polar_angle_idx] = pcs_from_polar[:, 0]
        walk[:, polar_radius_idx] = pcs_from_polar[:, 1]
    if use_pcs:
        # if using PCs, inverse transform the walk to get back to latent space
        # coordinates (for passing to the model to generate images)
        walk = pca.inverse_transform(walk)

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(model, walk, ranges, n_noise_samples, NUM_GPUS)

    # save generated latent walk as grid
    axis_suffix = "_along_pcs" if use_pcs else "_along_latent"
    file_name = f"latent_walk_{int(sigma)}sigma{axis_suffix}"
    if replace_mean_with_value is not None:
        replace_str = "_".join(
            [
                f"{column_names[i]}setto{str(val).replace('.', 'p')}"
                for i, val in enumerate(replace_mean_with_value)
                if val is not None
            ]
        )
        file_name += f"_replace_{replace_str}"
    plot_latent_walk_as_grid(walk_img_grid, ranges, save_path, file_name, use_pcs)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
