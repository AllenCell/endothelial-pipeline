from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, StrList
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    columns: StrList | None = None,
    sigma: float = 3.0,
    n_steps: int = 7,
    use_pcs: bool = True,
    n_noise_samples: int = 1,
    replace_mean_with_pc_value: list[float | None] | None = None,
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
        to exclude them.
    columns
        List of column names corresponding to the dimensions to perform the
        latent walk along. ∑If None, defaults to polar angle, polar radius, and
        flipped PC3.
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
    replace_mean_with_pc_value
        List of PC values to replace the mean with for each PC dimension. Must
        be of length num_pcs. If None, uses the mean of the data.
    """
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        polar_to_pcs,
    )
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.latent_walk_utils import (
        generate_latent_walk_images,
        get_latent_walk,
        get_num_dims_from_column_names,
    )
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

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

    # default column names if none provided
    column_names = (
        [ColumnName.POLAR_ANGLE.value, ColumnName.POLAR_RADIUS.value, ColumnName.PC3_FLIPPED.value]
        if columns is None
        else columns
    )

    # get number of dimensions for latent walk based on column names e.g., if
    # "pc_11" is in the column names, then the fit pca object needs to be fit
    # with at least 11 pcs, and the latent walk needs to be performed in at
    # least 11 dimensions
    n_dims = get_num_dims_from_column_names(column_names)

    # initialize pca variable to None in case use_pcs is False, so that it can
    # be passed to get_dataframe_for_dynamics_workflows without error
    pca = None
    if use_pcs:
        # get fit pca object and data for latent walk
        pca = fit_pca(
            dataset_collection_name=dataset_collection,
            dataframe_manifest_name=dataframe_manifest_name,
            include_cell_piling=include_cell_piling,
            num_pcs=n_dims,
        )

    dataframe_all_datasets = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                dataset_name,
                dataframe_manifest,
                pca=pca,
                include_cell_piling=include_cell_piling,
                crop_pattern=crop_pattern,
            )
            for dataset_name in dataset_names
        ]
    )
    data_for_walk = dataframe_all_datasets[column_names]

    # get coordinate values for latent walk and the ranges of the walk for each
    # dimension
    walk, ranges = get_latent_walk(
        data_for_walk,
        column_names,
        sigma if sigma > 0 else None,
        n_steps,
        replace_mean_with_pc_value,
    )
    # if polar angle and radius are included in the column names, convert them
    # to PC1 and PC2 coordinates for image generation (inverse PCA
    # transformation cannot be performed with polar coordinates)
    if use_pcs:
        if (
            ColumnName.POLAR_ANGLE.value in column_names
            and ColumnName.POLAR_RADIUS.value in column_names
        ):
            pc1_column_name = f"{ColumnName.PCA_FEATURE_PREFIX}1"
            pc2_column_name = f"{ColumnName.PCA_FEATURE_PREFIX}2"
            angle = walk[ColumnName.POLAR_ANGLE.value].to_numpy()
            radius = walk[ColumnName.POLAR_RADIUS.value].to_numpy()
            pc1_values, pc2_values = polar_to_pcs(angle, radius)
            walk[pc1_column_name] = pc1_values
            walk[pc2_column_name] = pc2_values

        # if flipped pc3 is included in the column names, convert it to regular pc3
        # before performing inverse PCA transformation for image generation (inverse PCA
        if ColumnName.PC3_FLIPPED.value in column_names:
            pc3_column_name = f"{ColumnName.PCA_FEATURE_PREFIX}3"
            walk[pc3_column_name] = -walk[ColumnName.PC3_FLIPPED.value].to_numpy()

        pc_column_names = [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(n_dims)]
        walk = pca.inverse_transform(walk[pc_column_names].to_numpy())
    else:
        walk = walk.to_numpy()
    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(model, walk, ranges, n_noise_samples, NUM_GPUS)

    # save generated latent walk as grid
    axis_suffix = "_along_pcs" if use_pcs else "_along_latent"
    file_name = f"latent_walk_{int(sigma)}sigma{axis_suffix}"
    if replace_mean_with_pc_value is not None:
        replace_str = "_".join(
            [
                f"{column_name}_setto_{val}"
                for column_name, val in zip(column_names, replace_mean_with_pc_value, strict=True)
                if val is not None
            ]
        )
        file_name += f"_replace_{replace_str}"
    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        column_names,
        save_path,
        file_name,
        label_sigmas=True if sigma > 0 else False,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
