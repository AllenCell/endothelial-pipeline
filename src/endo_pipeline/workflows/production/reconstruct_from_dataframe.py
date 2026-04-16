from typing import Literal


def main(
    fmsid: str | None = None,
    s3uri: str | None = None,
    path: str | None = None,
    columns: list[str] | None = None,
    dataset_labels: bool = True,
    output_dir: str | None = None,
    file_format: Literal[".png", ".svg", ".pdf"] = ".png",
    figsize: tuple[float, float] = (2, 2),
    random_seed: int | None = None,
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
    output directory will be used. The saved file names will include the feature
    coordinate values, and if dataset labels are included and `dataset_labels`
    is set to True, the dataset label will also be included in the file name.

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
    figsize
        Optional figure size for the saved plots.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, load_dataframe, load_model, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
    from endo_pipeline.library.model.latent_walk_utils import (
        add_pc_coordinates_to_dataframe,
        get_feature_coordinates_as_string,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        build_dataframe_location_from_path,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
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

    # default column names if none provided
    column_names = columns or list(DYNAMICS_COLUMN_NAMES)

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)

    # Directory to save reconstructed crops
    crop_savedir = get_output_path(output_dir or "reconstructed_crops")

    # get coordinates as array from dataframe
    required_columns = [Column.DATASET, *column_names] if dataset_labels else column_names
    check_required_columns_in_dataframe(dataframe, required_columns)
    feature_coords = np.atleast_2d(dataframe[column_names].to_numpy())

    # re-transform coordinates if they are in polar format (angle and radius) or
    # if they include flipped pc3
    dataframe = add_pc_coordinates_to_dataframe(dataframe, column_names)

    reconstructed_imgs = generate_from_dataframe(
        dataframe,
        column_names,
        model,
        num_gpus=NUM_GPUS,
        random_seed=random_seed,
        n_noise_samples=1,
    )
    print(reconstructed_imgs.shape)
    img_list = [reconstructed_imgs[i] for i in range(len(reconstructed_imgs))]

    for i, image in enumerate(img_list):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        file_name = "crop_"
        if dataset_labels:
            dataset_name = dataframe.iloc[i][Column.DATASET]
            file_name = f"{dataset_name}_{file_name}"
        feature_coord_as_str = get_feature_coordinates_as_string(column_names, feature_coords[i])
        save_plot_to_path(
            fig, crop_savedir, f"{file_name}{feature_coord_as_str}", file_format=file_format
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
