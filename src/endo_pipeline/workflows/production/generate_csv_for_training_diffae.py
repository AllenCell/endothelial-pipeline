TAGS = ["production", "diffae_model_training"]


def main() -> None:
    """
    Generate .csv files with paths to zarr files for training a DiffAE model.

    This script collects zarr file paths from multiple datasets, splits them into
    training and validation sets, and saves them as .csv files in a specified directory.
    It also includes metadata such as channel and resolution level.

    The datasets are defined in the `diffae_model_training` dataset collection configuration.

    These csv files are accessed by the data loader in the DiffAE model training script.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves the training and validation dataframes to `train.csv` and `val.csv`
        in the specified output directory.
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline.configs import (
        get_available_zarr_files,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from src.endo_pipeline.io import get_output_path

    output_savedir = get_output_path("manifests", include_timestamp=False)

    # load data config
    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_file_paths = [
        get_available_zarr_files(dataset_config) for dataset_config in dataset_config_list
    ]

    zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
    zarr_path_df["channel"] = "0,1"  # cdh5, brightfield
    zarr_path_df["resolution"] = 1  # downsample by half

    train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)

    # save the dataframes to csv files
    train.to_csv(output_savedir / "train.csv", index=False)
    val.to_csv(output_savedir / "val.csv", index=False)


if __name__ == "__main__":
    main()
