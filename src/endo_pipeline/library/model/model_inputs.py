import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.endo_pipeline.configs import DatasetConfig, load_dataset_collection_config
from src.endo_pipeline.io import get_output_path, load_dataframe_from_fms


def get_dataset_names_used_for_training(
    train_csv_path: Path, val_csv_path: Path, dataset_collection_name: str
) -> list[str]:
    """
    Pull list of dataset names used for model training
    from train.csv and val.csv files that are passed
    into the model training script.
    """
    # load train.csv and val.csv files as dataframes
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # get date part of dataset name from zarr path
    # note: this might be something that
    # gets turned into a zarr method in a future PR
    for df in [train_df, val_df]:
        df["dataset_date"] = df["path"].apply(lambda s: Path(s).stem.split("_")[0])

    # get unique dataset dates used in training from dataset_date
    # by combining the unique dates from both train and val datasets
    training_dataset_dates = list(
        set(train_df["dataset_date"].unique().tolist() + val_df["dataset_date"].unique().tolist())
    )

    # get unique dataset names by looping over
    # the provided dataset collection name,
    # which should be a superset of the datasets used for training

    training_dataset_superset = load_dataset_collection_config(dataset_collection_name)
    training_dataset_names = []
    for dataset_name in training_dataset_superset.datasets:
        for date in training_dataset_dates:
            if date in dataset_name:
                training_dataset_names.append(dataset_name)

    return sorted(training_dataset_names)
