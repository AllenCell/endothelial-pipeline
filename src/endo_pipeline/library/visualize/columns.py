"""Utility methods for plotting column data."""

import logging

from endo_pipeline.settings import column_metadata
from endo_pipeline.settings.column_names import ColumnNameType

logger = logging.getLogger(__name__)


def get_label_for_column(
    column_name: str,
    column_metadata: dict[
        ColumnNameType, column_metadata.ColumnMetadata
    ] = column_metadata.COLUMN_METADATA,
    single_line: bool = False,
) -> str:
    """
    Convert column name into label using column metadata.

    Parameters
    ----------
    column_name
        Column name to convert.
    column_metadata
        Mapping of column names to column metadata.

    Returns
    -------
    :
        Label for the column name.
    """

    if column_name not in column_metadata:
        logger.warning(
            "Column '%s' does not have associated metadata. "
            "Returning given column name as the label.",
            column_name,
        )
        return column_name

    label = column_metadata[column_name].label or column_name
    if single_line:
        label = label.replace("\n", " ")
    return label
