"""Methods for validating dataframe contents before further analysis."""

import logging

import pandas as pd

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.settings.column_names import ColumnName as Column

logger = logging.getLogger(__name__)


def check_required_columns_in_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Check that required columns are present in a given dataframe.

    Parameters
    ----------
    df
        DataFrame to check.
    required_columns
        List of required column names to check for.
    """

    for col in required_columns:
        if col not in df.columns:
            logger.error("DataFrame must contain column [ %s ]", col)
            raise ValueError(f"DataFrame must contain column [ {col} ]")


def check_dataframe_has_single_dataset(dataframe: pd.DataFrame) -> None:
    """
    Check that a given dataframe is restricted to a single dataset.

    This is done by checking that the column Column.DATASET contains only one
    unique value. If there are multiple unique values in the Column.DATASET
    column, a ValueError is raised.

    Parameters
    ----------
    dataframe
        DataFrame to check.
    """
    # first check that Column.DATASET is present in the dataframe
    check_required_columns_in_dataframe(dataframe, [Column.DATASET])

    # then check that there is only one unique dataset in the dataframe
    if dataframe[Column.DATASET].nunique() != 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")


def check_dataframe_dataset_matches_dataset_config(
    dataframe: pd.DataFrame,
    dataset_config: DatasetConfig,
) -> None:
    """
    Check that the dataset name in a given dataframe matches the dataset name in
    a given dataset config.

    Parameters
    ----------
    dataframe
        DataFrame to check.
    dataset_config
        Dataset config to check against.
    """
    # check that required columns are present in dataframe
    # first check that Column.DATASET is present in the dataframe
    check_required_columns_in_dataframe(dataframe, [Column.DATASET])

    # then check that there is only one unique dataset in the dataframe
    check_dataframe_has_single_dataset(dataframe)

    # then check that the dataset name in the dataframe matches the dataset name
    # in the dataset config
    if dataframe[Column.DATASET].unique()[0] != dataset_config.name:
        logger.error("Dataset name in dataframe does not match dataset name in dataset config.")
        raise ValueError("Dataset name in dataframe does not match dataset name in dataset config.")
