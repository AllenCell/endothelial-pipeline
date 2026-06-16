from endo_pipeline.cli import UniqueStrList


def main(manifest_names: UniqueStrList | None = None, reload_columns: bool = False) -> None:
    """
    Update column descriptions in manifests using column name docstrings.

    #internal #manifests

    This script iterates through all dataframe manifest to get column names
    and updates the column description using the column name docstrings by
    parsing the module.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe update-column-descriptions -vd
    ```

    To run the workflow for a single manifest:

    ```bash
    uv run endopipe update-column-descriptions MANIFEST_NAME
    ```

    Parameters
    ----------
    manifest_names
        List of dataframe manifest names to update. None to update all.
    reload_columns
        True to reload column names from dataframe, False to use names in manifest.
    """

    import ast
    import inspect
    import logging
    from itertools import pairwise

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import (
        get_available_dataframe_manifests,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column

    logger = logging.getLogger(__name__)

    column_descriptions = {}

    # Method for finding attribute assignment followed by docstring
    def _extract_docstrings_from_body(body) -> None:
        for a, b in pairwise(body):
            a_is_assign = isinstance(a, ast.Assign)
            b_is_string = (
                isinstance(b, ast.Expr)
                and isinstance(b.value, ast.Constant)
                and isinstance(b.value.value, str)
            )

            if not a_is_assign or not b_is_string:
                continue

            column_descriptions[a.value.value] = b.value.value.strip(".")

    # Parse source into abstract syntax tree
    source = ast.parse(inspect.getsource(Column)).body[0]

    # Extract docstrings from top level attributes
    _extract_docstrings_from_body(source.body)

    # Extract docstrings from each nested enum class
    for node in source.body:
        if isinstance(node, ast.ClassDef):
            _extract_docstrings_from_body(node.body)

    # Get list of manifest names to iterate through
    manifest_names = manifest_names or get_available_dataframe_manifests()

    for manifest_name in manifest_names:
        manifest = load_dataframe_manifest(manifest_name)

        # If the manifest does not have a columns entry, or if requesting to
        # reload the columns, read the columns from the first dataframe in the
        # manifest and assign empty descriptions.
        if not manifest.columns or reload_columns:
            logger.info("Reloading column names by reading columns from dataframe")
            location = manifest.locations[list(manifest.locations.keys())[0]]
            columns = load_dataframe(location, delay=True).columns
            manifest.columns = dict.fromkeys(columns, "")

        # Iterate through all the columns and assign a description, if available
        # in the column descriptions map.
        for column_name in manifest.columns:
            if column_name in column_descriptions:
                manifest.columns[column_name] = column_descriptions[column_name]

        save_dataframe_manifest(manifest)
