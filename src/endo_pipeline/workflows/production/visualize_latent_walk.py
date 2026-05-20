from typing import Annotated

from cyclopts import Parameter


def main(
    walk_on_columns: Annotated[
        list[str] | None, Parameter(name="--along", consume_multiple=True, negative_iterable=[])
    ] = None,
    set_column_value: Annotated[
        dict[str, float] | None, Parameter(name="--with", negative="")
    ] = None,
    sigma: float | None = None,
    n_steps: int = 7,
    n_noise_samples: int = 1,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Create latent walk for a given model using PC axes.

    #diffae_image_generation #pc_interpretation

    **Workflow defaults**

    This workflow performs a latent walk along features in the latent space of a
    Diffusion Autoencoder (DiffAE) model. By default, the workflow uses the
    model and data as defined by the global constants
    `DEFAULT_MODEL_MANIFEST_NAME`, `DEFAULT_MODEL_RUN_NAME`, and
    `DEFAULT_PCA_DATASET_COLLECTION_NAME`, which can be set in
    `endo_pipeline.settings.workflow_defaults`.

    The model manifest name and run name are used to load the model (for image
    generation) and the dataframe manifest of feature data (for getting the data
    to perform the walk on). The dataset collection name is used to get the list
    of datasets to load and concatenate to get the data for performing the walk.

    **Input columns**

    The columns to perform the latent walk along can be specified with the
    `walk_on_columns` argument. If not specified, defaults to polar angle,
    polar radius, and flipped PC3, which are the most interpretable dimensions
    in the latent space. The CLI flag for this argument is `--along`, e.g.,

    .. code-block:: bash
        endopipe visualize-latent-walk --along polar_theta

    If using PCs, any of the PCs can be selected by specifying the corresponding
    column name (e.g. "pc_1", "pc_2", etc.). If using original axes, any of the
    original latent space dimensions can be selected by specifying the
    corresponding column name (e.g. "feat_1", "feat_2", etc.).

    **Setting values of other columns when generating the latent walk**

    By default, when generating the latent walk, the values of all columns other
    than the current column being traversed are set to the mean value of those
    columns in the data. However, there may be cases where you want to set the
    values of certain columns to specific values instead of the mean when
    generating the latent walk.

    The option `set_column_value` allows you to do this by providing a
    dictionary where the keys are the column names and the values are the
    specific values you want to set for those columns when generating the latent
    walk.

    For example, you may want to set the polar radius to be equal to 1.0 when
    traversing along the polar angle dimension to see how changing the polar
    angle affects the images at that specific radius. To do this, you would set
    `set_column_value` to `{"polar_r": 1.0}` where "polar_r" is the name of
    the column corresponding to the polar radius in your data. This is done via
    the command line flag `--with` as follows:

    .. code-block:: bash
        endopipe visualize-latent-walk --along polar_theta --with.polar_r 1.0

    **Latent walk ranges**

    The range of the latent walk can be specified with the `sigma` argument,
    which indicates the number of standard deviations from the mean to traverse
    for the latent walk. For example, if sigma=2, the latent walk will traverse
    -2 to 2 standard deviations along the selected dimensions with `n_steps` steps. If not
    specified, the latent walk will traverse the full range of the data in each
    dimension.

    Parameters
    ----------
    walk_on_columns
        List of column names corresponding to the dimensions to perform the
        latent walk along.
    set_column_value
        Optional, dictionary mapping column names to values to set for those
        columns when generating the latent walk.
    sigma
        Optional, number of standard deviations from the mean to traverse for
        the latent walk.
    n_steps
        Number of steps in the latent walk.
    n_noise_samples
        Number of noise samples to use for generating images.
    figsize
        Optional, tuple specifying the figure size in inches (width, height). If
        not provided, defaults to (6.5, num_rows) where num_rows is the number of
        dimensions in the latent walk.
    """

    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, load_model
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.diffae.generate_image import generate_latent_walk_images
    from endo_pipeline.library.model.latent_walk_utils import (
        add_pc_coordinates_to_dataframe,
        get_column_names_for_latent_walk_dataframe,
        get_feature_coordinates_as_string,
        get_latent_walk,
        get_num_pcs_from_column_names,
    )
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    )

    # load model manifest, get run name, and load model
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)
    model = load_model(model_manifest.locations[run_name], instantiate=True)
    if not isinstance(model, DiffusionAutoEncoder):
        raise ValueError(
            f"Model loaded from {model_manifest_name} with run name {run_name} is not a DiffusionAutoEncoder."
        )

    # set up output directory
    save_path = get_output_path(__file__)

    # load model configuration and reference dataset manifests
    dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # default column names if none provided
    # default column names for walk if none provided
    walk_column_names = (
        [
            Column.DiffAEData.POLAR_ANGLE.value,
            Column.DiffAEData.POLAR_RADIUS.value,
            Column.DiffAEData.PC3_FLIPPED.value,
        ]
        if walk_on_columns is None
        else walk_on_columns
    )

    # get column names not used for walk but to be set to specific values when generating the walk
    set_column_names = list(set_column_value.keys()) if set_column_value is not None else []

    # columns to keep in dataframe are the union of the walk column names and the set column names
    input_column_names = list(set(walk_column_names + set_column_names))

    # process input column names and add any additional ones needed for the walk / image generation
    column_names = get_column_names_for_latent_walk_dataframe(input_column_names)

    # get minimum number of pcs needed for the fit pca object based on the
    # column names provided; for example, if "pc_11" is in the column names,
    # then the fit pca object needs to be fit with at least 11 pcs
    num_pcs = get_num_pcs_from_column_names(column_names)
    if num_pcs == 0:
        raise ValueError(f"No PC-related column names found in {column_names}.")

    # get fit pca object and data for latent walk
    pca = fit_pca(num_pcs=num_pcs)

    dataframe_all_datasets = pd.concat(
        [
            load_dataframe(get_dataframe_location_for_dataset(dataframe_manifest, dataset_name))
            for dataset_name in dataset_names
        ]
    )

    data_for_walk = dataframe_all_datasets[column_names]

    # get coordinate values for latent walk and the ranges of the walk for each
    # dimension
    walk, ranges = get_latent_walk(
        data_for_walk,
        walk_column_names,
        sigma,
        n_steps,
        set_column_value,
    )

    # re-transform coordinates if they are in polar format (angle and radius) or
    # if they include flipped pc3
    walk = add_pc_coordinates_to_dataframe(walk, column_names)

    pc_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    walk = pca.inverse_transform(walk[pc_column_names].to_numpy())

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(model, walk, ranges, n_noise_samples, NUM_GPUS)

    # save generated latent walk as grid
    axis_suffix = "_along_" + "_".join(walk_column_names)
    sigma_suffix = f"_{int(sigma)}sigma_" if sigma is not None else ""
    file_name = f"latent_walk{sigma_suffix}{axis_suffix}"
    if set_column_value is not None:
        cols_setto = [f"{col}_setto" for col in set_column_names]
        replace_str = get_feature_coordinates_as_string(cols_setto, list(set_column_value.values()))
        file_name += f"_with_{replace_str}"

    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        walk_column_names,
        save_path,
        file_name,
        label_sigmas=False if sigma is None else True,
        figsize=figsize,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
