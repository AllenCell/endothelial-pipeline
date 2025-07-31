TAGS = ["diffae_model_training"]


def main(zarr_resolution: int = 1) -> None:
    """
    Generate .csv files with paths to zarr files for training a DiffAE model.

    This script collects zarr file paths from multiple datasets, splits them into
    training and validation sets, and saves them as .csv files in a specified directory.
    It also includes metadata such as channel and resolution level.

    The datasets are defined in the `diffae_model_training` dataset collection configuration.

    These csv files are accessed by the data loader in the DiffAE model training script.

    Parameters
    ----------
    zarr_resolution
        The resolution level of the zarr files to be used for training. Default is 1,
        which corresponds to downsampling by half.

    Returns
    -------
    :
        Uploads the training and validation dataframes to FMS and saves a DataframeManifest
        with DatasetLocation objects containing the FMS IDs of the uploaded files.
    """

    import datetime

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline.configs import (
        get_available_zarr_files,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from src.endo_pipeline.io import (
        build_fms_annotations_for_model_training_inputs,
        get_output_path,
        upload_file_to_fms,
    )
    from src.endo_pipeline.manifests import (
        DataframeLocation,
        DataframeManifest,
        save_dataframe_manifest,
    )

    output_savedir = get_output_path("manifests", include_timestamp=False)

    # load data config
    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_file_paths = []
    for dataset_config in dataset_config_list:
        available_zarr_files = get_available_zarr_files(dataset_config)
        zarr_file_paths.extend(
            [str(zarr_file) for zarr_file in available_zarr_files]  # convert Path to str
        )

    zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
    zarr_path_df["channel"] = "0,1"  # cdh5, brightfield
    zarr_path_df["resolution"] = zarr_resolution  # downsampling factor

    train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)

    # save the dataframes to csv files locally as intermediates
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_output_path = output_savedir / f"train_resolution_{zarr_resolution}_{timestamp}.csv"
    val_output_path = output_savedir / f"val_resolution_{zarr_resolution}_{timestamp}.csv"
    train.to_csv(train_output_path, index=False)
    val.to_csv(val_output_path, index=False)

    # upload dataframes to fms
    train_annotations = build_fms_annotations_for_model_training_inputs(
        "training",
        dataset_name_list,
        zarr_resolution,
        include_timestamp=False,
        include_git_info=False,
    )
    val_annotations = build_fms_annotations_for_model_training_inputs(
        "validation",
        dataset_name_list,
        zarr_resolution,
        include_timestamp=False,
        include_git_info=False,
    )

    train_fmsid = upload_file_to_fms(
        train_output_path,
        annotations=train_annotations,
        file_type="csv",
    )
    val_fmsid = upload_file_to_fms(
        val_output_path,
        annotations=val_annotations,
        file_type="csv",
    )

    # build and save out DataframeManifest object with FMS IDs
    dataframe_manifest = DataframeManifest(
        name=f"diffae_training_csv_resolution_{zarr_resolution}_{timestamp}",
        workflow="generate_diffae_training_csv",
        parameters={"zarr_resolution": zarr_resolution},
        locations={
            "train": DataframeLocation(fmsid=train_fmsid, s3uri=None),
            "val": DataframeLocation(fmsid=val_fmsid, s3uri=None),
        },
    )

    save_dataframe_manifest(dataframe_manifest)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
