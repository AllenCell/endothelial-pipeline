from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    path: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
    column_names: list[str] | None = None,
    dataset_labels: bool = False,
) -> None:
    """
    Reconstruct crops from feature space coordinates stored in a given dataframe.

    The reconstructed crops are saved as PNG files in a local directory.

    #diffae_features #image_generation #pca

    **Dataframe file format**:

    The dataframe file (.csv, .parquet, etc.) should contain rows of latent
    space coordinates, with each row representing a point in the PCA-transformed
    space.

    The column names for the features can either be specified via
    the ``column_names`` parameter or will default to the standard naming
    convention defined in ``DIFFAE_PC_COLUMN_NAMES``.

    ** Dataset labels**:

    If the dataframe contains metadata for dataset labels corresponding to each
    point, the column name for the dataset is specified by
    ``ColumnName.DATASET`` in
    ``endo_pipeline.settings.diffae_feature_dataframes``. If the user input
    parameter ``dataset_labels`` is set to True, the dataset label will be
    prefixed to the saved file names. Else, the saved file names will only
    contain the feature coordinate values.

    Parameters
    ----------
    path
        Path to a dataframe file containing PC space coordinates.
    model_manifest_name
        Name of the model manifest containing the specific run to load features
        from.
    run_name
        Run name corresponding to features to load and the model to use for
        image reconstruction.
    num_pcs
        Number of principal components used in the PCA transformation.
    column_names
        List of column names in the dataframe corresponding to the feature
        coordinates.
    dataset_labels
        If true, the dataset label from the dataframe will be prefixed to the
        saved file names.
    """
    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, load_dataframe, load_model, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.model import generate_from_coords_batch
    from endo_pipeline.manifests import (
        build_dataframe_location_from_path,
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES

    logger = logging.getLogger(__name__)

    # convert csv_path to Path object
    dataframe_path = Path(path).resolve()
    dataframe_location = build_dataframe_location_from_path(path)
    dataframe = load_dataframe(dataframe_location)

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)

    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )

    # Get fit (3D) PCA object from manifest
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=num_pcs)

    # Directory to save reconstructed crops
    dataframe_file_name = dataframe_path.stem
    crop_savedir = get_output_path(
        "reconstructed_crops", model_manifest_name, run_name_, dataframe_file_name
    )

    # get coordinates as array from dataframe, using specified feature column names or default names
    feature_column_names = (
        DIFFAE_PC_COLUMN_NAMES[:num_pcs] if column_names is None else column_names
    )
    required_columns = (
        [Column.DATASET, *feature_column_names] if dataset_labels else feature_column_names
    )
    check_required_columns_in_dataframe(dataframe, required_columns)  # validate required columns
    feature_coords = dataframe[feature_column_names].values  # extract coordinate values

    # make sure that coords is a 2D array with shape (num_points, num_dimensions)
    feature_coords = np.atleast_2d(feature_coords)
    num_points, num_dims = feature_coords.shape
    logger.debug(
        "Loaded [ %d ] points with [ %d ] dimensions from dataframe file [ %s ].",
        num_points,
        num_dims,
        dataframe_file_name,
    )
    if num_dims != num_pcs:
        logger.error(
            "Expected coordinates of [ %d ] dimensions from loaded dataframe, but got [ %d ] dimensions.",
            num_pcs,
            num_dims,
        )
        raise ValueError(
            f"Expected coordinates of [ {num_pcs} ] dimensions from loaded dataframe, but got [ {num_dims} ] dimensions."
        )

    # transform interpolated points to full latent space
    latent_coords = pca.inverse_transform(feature_coords)

    walk_imgs = generate_from_coords_batch(model, latent_coords, num_gpus=NUM_GPUS)

    for i, img in enumerate(walk_imgs):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        file_name = "feature_coordinate_"
        if dataset_labels:
            dataset_name = dataframe.iloc[i][Column.DATASET]
            file_name = f"{dataset_name}_{file_name}"
        feature_coord_as_str = "_".join([f"{coord:.2f}" for coord in feature_coords[i]])
        save_plot_to_path(fig, crop_savedir, f"{file_name}{feature_coord_as_str}.png")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
