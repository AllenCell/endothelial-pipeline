"""Utility methods for plotting feature data."""

from typing import Any

from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_metadata import COLUMN_METADATA_DICT
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


def get_label_for_column(
    column_name: str,
    mapping_dict: dict[str, dict[str, Any]] | None = None,
    capitalize: bool = False,
) -> str:
    """Convert dataframe column names to human-readable labels.

    For example, "feat_0" becomes "Feature 0", and "pc_1" becomes "PC 1".

    Parameters
    ----------
    column_name
        Column name to convert.
    mapping_dict
        Optional dictionary mapping column names to human-readable labels.
    capitalize
        Capitalize the first letter of the label if True, otherwise leave as is.

    Returns
    -------
    :
        Human-readable label for the column name.

    """
    # check for other specific patterns, overriding default label
    label = None

    if column_name.startswith(f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}"):
        feature_number = column_name.split("_")[1]
        label = f"feature {feature_number}"
    elif column_name.startswith(f"{Column.DiffAEData.PCA_FEATURE_PREFIX}"):
        pc_number = column_name.split("_")[1]
        label = f"PC {pc_number}"
    elif column_name == Column.DiffAEData.POLAR_RADIUS:
        label = "r"
    elif column_name == Column.DiffAEData.POLAR_ANGLE:
        label = f"{Unicode.THETA}"
    elif column_name == Column.DiffAEData.PC3_FLIPPED:
        label = f"{Unicode.RHO}"
    elif column_name == Column.OpticalFlow.UNIT_VECTOR_MEAN:
        label = "Migration Coherence"
    elif column_name == Column.OpticalFlow.SPEED_MEAN:
        label = "Mean Speed"
    elif column_name == Column.OpticalFlow.ANGLE_MEAN:
        label = "Optical Flow Mean Angle"

    # check mapping dict for label override
    if mapping_dict is None:
        mapping_dict = COLUMN_METADATA_DICT
    if column_name in mapping_dict:
        label = mapping_dict[column_name]["label"]

    # if no label found, return column name as is
    if label is None:
        return column_name

    # else, take label found and capitalize if specified
    if capitalize:
        label = label.capitalize()

    return label
