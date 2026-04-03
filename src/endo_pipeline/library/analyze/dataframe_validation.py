"""Methods for validating dataframe contents before further analysis."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def check_required_columns_in_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Check that required columns are present in a given dataframe.

    Raises a ValueError if any required column is missing.

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
