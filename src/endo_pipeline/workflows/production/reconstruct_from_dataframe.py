from typing import Literal


def main(
    fmsid: str | None,
    s3uri: str | None,
    path: str | None,
    columns: list[str] | None = None,
    dataset_labels: bool = False,
    output_dir: str | None = None,
    file_format: Literal[".png", ".svg", ".pdf"] = ".png",
) -> None:
    """
    Reconstruct crops from feature coordinates stored in a given dataframe.

    #diffae-features #image-generation #pca

    **Dataframe location**

    The dataframe containing the feature coordinates can be specified via one of
    three mutually exclusive parameters: `path`, `fmsid`, or `s3uri`. The
    function will attempt to load the dataframe from the specified location. If
    multiple location parameters are provided, the function will load the
    dataframe based on the following priority order: `path` > `fmsid` > `s3uri`.
    If none of the location parameters are provided, the function will raise a
    ValueError.

    **Dataframe file format**

    The dataframe file (.csv, .parquet, etc.) should contain rows of
    coordinates, with each row representing a point in the feature space.

    The column names for the features can be specified via the `columns`
    parameter. If not provided, they will default to the features defined in the
    `DYNAMICS_COLUMN_NAMES` setting in
    `endo_pipeline.settings.dynamics_workflows`.

    The column names should correspond to the feature coordinates used for image
    reconstruction. For example, if the features are PCA coordinates, the column
    names might be "pc_1", "pc_2", etc.

    **Feature variable transformations**

    If the dataframe contains polar coordinates (angle and radius) instead of
    Cartesian coordinates (PC1 and PC2), the function will automatically convert
    the polar coordinates to Cartesian coordinates before performing the inverse
    PCA transformation for image reconstruction.

    If the dataframe contains a flipped version of PC3 (named "pc3_flipped"),
    the function will automatically convert it to regular PC3 by negating the
    values before performing the inverse PCA transformation for image
    reconstruction.

    This workflow does not currently support dataframes with other types of
    feature variable transformations, nor does it support use of the original
    latent coordinates as features for image reconstruction without PCA
    transformation. If the dataframe contains features that do not correspond to
    the expected coordinate formats, the function may raise an error or produce
    incorrect reconstructions.

    **Dataset labels**

    If the dataframe contains metadata for dataset labels corresponding to each
    point, the column name for the dataset is specified by `ColumnName.DATASET`
    in `endo_pipeline.settings.diffae_feature_dataframes`. If the user input
    parameter ``dataset_labels`` is set to True, the dataset label will be
    prefixed to the saved file names. Else, the saved file names will only
    contain the feature coordinate values.

    **Workflow output**

    The reconstructed crops will be saved to the specified output directory in
    the specified file format. If no output directory is provided, a default
    output directory will be used.

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
    columns
        List of column names in the dataframe corresponding to the feature
        coordinates.
    dataset_labels
        If true, the dataset label from the dataframe will be prefixed to the
        saved file names.
    output_dir
        Optional output directory to save reconstructed crops. If not provided,
        a default output directory will be used.
    file_format
        Optional file format for the saved plots.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, load_dataframe, load_model, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.analyze.polar_coords import polar_to_pcs
    from endo_pipeline.library.model import generate_from_coords_batch
    from endo_pipeline.library.model.latent_walk_utils import get_num_pcs_from_column_names
    from endo_pipeline.manifests import (
        DataframeLocation,
        build_dataframe_location_from_path,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # convert input location to DataframeLocation and load dataframe
    if path is not None:
        dataframe_location = build_dataframe_location_from_path(path)
    elif fmsid is not None:
        dataframe_location = DataframeLocation(fmsid=fmsid)
    elif s3uri is not None:
        dataframe_location = DataframeLocation(s3uri=s3uri)
    else:
        raise ValueError(
            "One of path, fmsid, or s3uri must be provided to specify dataframe location."
        )
    dataframe = load_dataframe(dataframe_location)

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)

    # Directory to save reconstructed crops
    crop_savedir = get_output_path(output_dir or "reconstructed_crops")

    # get minimum number of pcs needed for the fit pca object based on the
    # column names provided; for example, if "pc_11" is in the column names,
    # then the fit pca object needs to be fit with at least 11 pcs
    column_names = DYNAMICS_COLUMN_NAMES or columns
    num_pcs = get_num_pcs_from_column_names(column_names)
    if num_pcs == 0:
        raise ValueError(f"No PC-related column names found in {column_names}.")

    # Get fit (3D) PCA object from manifest
    pca = fit_pca(num_pcs=num_pcs)

    # get coordinates as array from dataframe
    required_columns = [Column.DATASET, *column_names] if dataset_labels else column_names
    check_required_columns_in_dataframe(dataframe, required_columns)
    feature_coords = dataframe[column_names].to_numpy()

    # make sure that coords is a 2D array with shape (num_points, num_dimensions)
    feature_coords = np.atleast_2d(feature_coords)

    # if polar angle and radius are included in the column names, convert them
    # to PC1 and PC2 coordinates for image generation (inverse PCA
    # transformation cannot be performed with polar coordinates)
    if (
        Column.DiffAEData.POLAR_ANGLE.value in column_names
        and Column.DiffAEData.POLAR_RADIUS.value in column_names
    ):
        pc1_column_name = f"{Column.DiffAEData.PCA_FEATURE_PREFIX}1"
        pc2_column_name = f"{Column.DiffAEData.PCA_FEATURE_PREFIX}2"
        angle = dataframe[Column.DiffAEData.POLAR_ANGLE].to_numpy()
        radius = dataframe[Column.DiffAEData.POLAR_RADIUS].to_numpy()
        pc1_values, pc2_values = polar_to_pcs(angle, radius)
        dataframe[pc1_column_name] = pc1_values
        dataframe[pc2_column_name] = pc2_values

    # if flipped pc3 is included in the column names, convert it to regular pc3
    # before performing inverse PCA transformation for image generation
    if Column.DiffAEData.PC3_FLIPPED.value in column_names:
        pc3_column_name = f"{Column.DiffAEData.PCA_FEATURE_PREFIX}3"
        dataframe[pc3_column_name] = -dataframe[Column.DiffAEData.PC3_FLIPPED.value].to_numpy()

    # get latent coordinates by performing inverse PCA transformation on the PC
    # coordinates from the dataframe; only use the PC columns needed for the
    # inverse transformation based on the number of PCs determined earlier
    pc_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    latent_coords = pca.inverse_transform(dataframe[pc_column_names].to_numpy())

    reconstructed_imgs = generate_from_coords_batch(model, latent_coords, num_gpus=NUM_GPUS)

    for i, img in enumerate(reconstructed_imgs):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        file_name = "feature_coordinate_"
        if dataset_labels:
            dataset_name = dataframe.iloc[i][Column.DATASET]
            file_name = f"{dataset_name}_{file_name}"
        feature_coord_as_str = "_".join([f"{coord:.2f}" for coord in feature_coords[i]])
        save_plot_to_path(
            fig, crop_savedir, f"{file_name}{feature_coord_as_str}", file_format=file_format
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
