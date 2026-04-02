"""Methods for validating dataframe contents before further analysis."""

import logging

import pandas as pd

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


def check_dataframe_has_single_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Check that a given dataframe is restricted to a single dataset.

    This is done by checking that the column Column.DATASET contains only one
    unique value. If there are multiple unique values in the Column.DATASET
    column, a ValueError is raised.

    Parameters
    ----------
    df
        DataFrame to check.
    """
    # first check that Column.DATASET is present in the dataframe
    check_required_columns_in_dataframe(df, [Column.DATASET])

    # then check that there is only one unique dataset in the dataframe
    if df[Column.DATASET].nunique() != 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")
