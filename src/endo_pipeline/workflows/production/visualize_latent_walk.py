from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern
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
    walk_on_columns: Annotated[
        list[str] | None, Parameter(name="--along", consume_multiple=True, negative_iterable=[])
    ] = None,
    sigma: float | None = None,
    n_steps: int = 7,
    use_pcs: bool = True,
    n_noise_samples: int = 1,
    set_column_value: Annotated[
        dict[str, float] | None, Parameter(name="--with", negative="")
    ] = None,
) -> None:
    """
    Create latent walk for a given model using PC axes or original axes.

    **Input columns**

    The columns to perform the latent walk along can be specified with the
    ``walk_on_columns`` argument. If not specified, defaults to polar angle, polar
    radius, and flipped PC3, which are the most interpretable dimensions in the
    latent space. The CLI flag for this argument is ``--along``, e.g.,

    .. code-block:: bash
        endopipe visualize-latent-walk --along polar_theta

    If using PCs, any of the PCs can be selected by specifying the corresponding
    column name (e.g. "pc_1", "pc_2", etc.). If using original axes, any of the
    original latent space dimensions can be selected by specifying the
    corresponding column name (e.g. "feat_1", "feat_2", etc.).

    **Latent walk ranges**

    The range of the latent walk can be specified with the ``sigma`` argument,
    which indicates the number of standard deviations from the mean to traverse
    for the latent walk. For example, if sigma=2, the latent walk will traverse
    (-2, -1, 0, 1, 2) standard devations along the selected dimensions. If not
    specified, the latent walk will traverse the full range of the data in each
    dimension.

    **Setting values of other columns when generating the latent walk**

    By default, when generating the latent walk, the values of all columns other
    than the current column being traversed are set to the mean value of those
    columns in the data. However, there may be cases where you want to set the
    values of certain columns to specific values instead of the mean when
    generating the latent walk.

    The option ``set_column_value`` allows you to do this by providing a
    dictionary where the keys are the column names and the values are the
    specific values you want to set for those columns when generating the latent
    walk.

    For example, you may want to set the polar radius to be equal to 1.0 when
    traversing along the polar angle dimension to see how changing the polar
    angle affects the images at that specific radius. To do this, you would set
    ``set_column_value`` to ``{"polar_r": 1.0}`` where "polar_r" is the name of
    the column corresponding to the polar radius in your data. This is done via
    the command line flag ``--with`` as follows:

    .. code-block:: bash
        endopipe visualize-latent-walk --along polar_theta --with.polar_r 1.0

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
    walk_on_columns
        List of column names corresponding to the dimensions to perform the
        latent walk along.
    sigma
        Optional, number of standard deviations from the mean to traverse for
        the latent walk.
    n_steps
        Number of steps in the latent walk.
    use_pcs
        True to use principal component axes, False to use original latent space
        axes.
    n_noise_samples
        Number of noise samples to use for generating images.
    set_column_value
        Optional, dictionary mapping column names to values to set for those
        columns when generating the latent walk.
    """
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_pc_column_names,
        polar_to_pcs,
    )
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.latent_walk_utils import (
        generate_latent_walk_images,
        get_column_names_for_latent_walk_dataframe,
        get_latent_walk,
        get_num_pcs_from_column_names,
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
    if not isinstance(model, DiffusionAutoEncoder):
        raise ValueError(
            f"Model loaded from {model_manifest_name} with run name {run_name_} is not a DiffusionAutoEncoder."
        )

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
    # default column names for walk if none provided
    walk_column_names = (
        [ColumnName.POLAR_ANGLE.value, ColumnName.POLAR_RADIUS.value, ColumnName.PC3_FLIPPED.value]
        if walk_on_columns is None
        else walk_on_columns
    )

    # get column names not used for walk but to be set to specific values when generating the walk
    set_column_names = list(set_column_value.keys()) if set_column_value is not None else []

    # columns to keep in dataframe are the union of the walk column names and the set column names
    input_column_names = list(set(walk_column_names + set_column_names))

    # process input column names and add any additional ones needed for the walk / image generation
    column_names = get_column_names_for_latent_walk_dataframe(input_column_names)

    compute_polar = False
    if (
        ColumnName.POLAR_ANGLE.value in column_names
        or ColumnName.POLAR_RADIUS.value in column_names
    ):
        compute_polar = True

    flip_pc3_sign = False
    if ColumnName.PC3_FLIPPED.value in column_names:
        flip_pc3_sign = True

    # initialize pca variable to None in case use_pcs is False, so that it can
    # be passed to get_dataframe_for_dynamics_workflows without error
    pca = None
    if use_pcs:
        # get minimum number of pcs needed for the fit pca object based on the
        # column names provided; for example, if "pc_11" is in the column names,
        # then the fit pca object needs to be fit with at least 11 pcs
        num_pcs = get_num_pcs_from_column_names(column_names)
        if num_pcs == 0:
            raise ValueError(
                f"Column names indicate use_pcs=True but no PC-related column names found in {column_names}."
            )
        # get fit pca object and data for latent walk
        pca = fit_pca(
            dataset_collection_name=dataset_collection,
            dataframe_manifest_name=dataframe_manifest_name,
            include_cell_piling=include_cell_piling,
            num_pcs=num_pcs,
        )

    dataframe_all_datasets = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                dataset_name,
                dataframe_manifest,
                pca=pca,
                include_cell_piling=include_cell_piling,
                crop_pattern=crop_pattern,
                compute_polar=compute_polar,
                flip_pc3_sign=flip_pc3_sign,
            )
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

        pc_column_names = get_pc_column_names(num_pcs)
        walk = pca.inverse_transform(walk[pc_column_names].to_numpy())
    else:
        walk = walk.to_numpy()
    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(model, walk, ranges, n_noise_samples, NUM_GPUS)

    # save generated latent walk as grid
    axis_suffix = "_along_" + "_".join(walk_column_names)
    sigma_suffix = f"_{int(sigma)}sigma_" if sigma is not None else ""
    file_name = f"latent_walk{sigma_suffix}{axis_suffix}"
    if set_column_value is not None:
        replace_str = "_".join(
            [
                f"{column_name}_setto_{str(val).replace('.', 'p')}"
                for column_name, val in set_column_value.items()
            ]
        )
        file_name += f"_{replace_str}"
    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        walk_column_names,
        save_path,
        file_name,
        label_sigmas=False if sigma is None else True,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
