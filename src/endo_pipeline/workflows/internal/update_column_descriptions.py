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
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.workflow_defaults import DEFAULT_NUM_LATENT_DIMENSIONS

    logger = logging.getLogger(__name__)

    template_expansions = {
        ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP: [0, 1, 2, 3, 4, 5, 6],
        ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP_CENTROID_X: [0, 1, 2, 3, 4, 5, 6],
        ColumnTemplate.NUCLEI_WITH_MOST_OVERLAP_CENTROID_Y: [0, 1, 2, 3, 4, 5, 6],
        ColumnTemplate.LATENT_FEATURE: range(DEFAULT_NUM_LATENT_DIMENSIONS),
        ColumnTemplate.PCA_FEATURE: range(1, DEFAULT_NUM_LATENT_DIMENSIONS + 1),
        ColumnTemplate.FIXED_POINT: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.DRIFT_COEFFICIENT: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.MESH_GRID: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.BASELINE_FIXED_POINT: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.BOOTSTRAP_CLUSTER_MEAN: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.BOOTSTRAP_CI_LOWER: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
        ColumnTemplate.BOOTSTRAP_CI_UPPER: ["rho", ("polar_theta", "theta"), ("polar_r", "r")],
    }

    column_descriptions = {}

    def _extract_docstrings_from_body(body) -> None:
        for a, b in pairwise(body):
            # Finding attribute assignment followed by docstring
            a_is_assign = isinstance(a, ast.Assign)
            b_is_string = (
                isinstance(b, ast.Expr)
                and isinstance(b.value, ast.Constant)
                and isinstance(b.value.value, str)
            )

            if not a_is_assign or not b_is_string:
                continue

            # Get base column name and description (which may be templated)
            name = a.value.value
            description = b.value.value

            # Remove whitespace
            description = " ".join([d.strip() for d in description.split("\n")]).strip(" .")

            # Expand templated descriptions
            if name in template_expansions:
                description = description.replace("Column name template: ", "")
                for option in template_expansions[name]:
                    if isinstance(option, tuple):
                        column_descriptions[name % option[0]] = description % option[1]
                    else:
                        column_descriptions[name % option] = description % option
            else:
                column_descriptions[name] = description

    # Parse source into abstract syntax tree
    source = ast.parse(inspect.getsource(Column)).body[0]

    # Extract docstrings from top level attributes
    _extract_docstrings_from_body(source.body)

    # Extract docstrings from each nested enum class
    for node in source.body:
        if isinstance(node, ast.ClassDef):
            _extract_docstrings_from_body(node.body)

    # Extract and expand docstrings from templates
    prefix_source = ast.parse(inspect.getsource(ColumnTemplate)).body[0]
    _extract_docstrings_from_body(prefix_source.body)

    # Get list of manifest names to iterate through
    manifest_names = manifest_names or get_available_dataframe_manifests()

    for manifest_name in manifest_names:
        manifest = load_dataframe_manifest(manifest_name)

        # If the manifest does not have a columns entry, or if requesting to
        # reload the columns, read the columns from all dataframes in the
        # manifest and assign empty descriptions.
        if not manifest.columns or reload_columns:
            logger.info("Reloading column names by reading columns from dataframe")
            columns = []
            for location in manifest.locations.values():
                dataframe_columns = load_dataframe(location, delay=True).columns
                columns.extend([col for col in dataframe_columns if col not in columns])
            manifest.columns = dict.fromkeys(columns, "")

        # Iterate through all the columns and assign a description, if available
        # in the column descriptions map.
        for column_name in manifest.columns:
            if column_name in column_descriptions:
                manifest.columns[column_name] = column_descriptions[column_name]

        save_dataframe_manifest(manifest)
