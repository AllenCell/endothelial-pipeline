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
) -> None:
    """
    Visualize latent walk across select features.

    #diffae #pca #visualization #test-ready

    This workflow performs a latent walk along features in the latent space of a
    Diffusion Autoencoder (DiffAE) model.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-latent-walk -d
    ```

    To run the full workflow:

    ```bash
    uv run endopipe visualize-latent-walk
    ```

    ## Selecting latent walk columns

    The columns to use for the latent walk may be specified with the `--along`
    option. By default, this workflow will run on the "polar_theta", "polar_r",
    and "rho" columns. PCs can be selected using "pc_1", "pc_2", etc. Original
    latent space dimensions can be selected using "feat_"1", "feat_2", etc.

    ```bash
    uv run endopipe visualize-latent-walk --along COLUMN_NAME COLUMN_NAME
    ```

    ## Setting values of other columns when generating the latent walk

    By default, when generating the latent walk, the values of all columns other
    than the current column being traversed are set to the mean value of those
    columns in the data. However, there may be cases where you want to set the
    values of certain columns to specific values instead of the mean when
    generating the latent walk using the `--with` option.

    ```bash
    uv run endopipe visualize-latent-walk --with.COLUMN_NAME=COLUMN_VALUE
    ```

    ## Latent walk ranges

    The range of the latent walk can be specified with the `sigma` argument,
    which indicates the number of standard deviations from the mean to traverse
    for the latent walk. For example, if sigma=2, the latent walk will traverse
    -2 to 2 standard deviations along the selected dimensions with `n_steps`
    steps. If not specified, the latent walk will traverse the full range of the
    data in each dimension with `n_steps` steps.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize the
    latent walk for a single column with 3 steps and 1 noise sample.

    Parameters
    ----------
    walk_on_columns
        Columns to perform latent walk along.
    set_column_value
        Columns to set to specific values when generating latent walk.
    sigma
        Number of standard deviations from the mean to traverse for latent walk.
    n_steps
        Number of steps in the latent walk.
    n_noise_samples
        Number of noise samples to use for generating images.
    """

    import logging

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
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
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    # load model manifest, get run name, and load model
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)
    model = load_model(model_manifest.locations[run_name], instantiate=True)

    if not isinstance(model, DiffusionAutoEncoder):
        logger.error(
            "Model loaded from '%s' with run name '%s' is not a DiffusionAutoEncoder",
            model_manifest_name,
            run_name,
        )
        return

    # load model configuration and reference dataset manifests
    dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # default column names if not provided
    walk_column_names = walk_on_columns or list(DYNAMICS_COLUMN_NAMES)

    # get column names not used for walk but to be set to specific values when generating the walk
    set_column_names = list(set_column_value.keys()) if set_column_value is not None else []

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to single column with 3 steps and 1 noise sample")
        walk_column_names = walk_column_names[:1]
        n_steps = 3
        n_noise_samples = 1

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
        output_path,
        file_name,
        label_sigmas=False if sigma is None else True,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
