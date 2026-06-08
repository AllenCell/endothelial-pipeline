def main() -> None:
    """
    Sync column descriptions in manifests with column settings docstrings.

    #internal #manifests
    """

    import ast
    import inspect
    from itertools import pairwise

    from endo_pipeline.manifests import (
        get_available_dataframe_manifests,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import NUM_LATENT_FEATURES

    COLUMN_DESCRIPTIONS = {}

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

            COLUMN_DESCRIPTIONS[a.value.value] = b.value.value.strip(".")

    # Parse source into abstract syntax tree
    source = ast.parse(inspect.getsource(Column)).body[0]

    # Extract docstrings from top level attributes
    _extract_docstrings_from_body(source.body)

    # Extract docstrings from each nested enum class
    for node in source.body:
        if isinstance(node, ast.ClassDef):
            if node.name != "SegData" and node.name != "DiffAEData":
                continue
            _extract_docstrings_from_body(node.body)

    # Expand prefix features
    for prefix in [Column.DiffAEData.LATENT_FEATURE_PREFIX, Column.DiffAEData.PCA_FEATURE_PREFIX]:
        for i in range(NUM_LATENT_FEATURES + 1):
            COLUMN_DESCRIPTIONS[f"{prefix}{i}"] = COLUMN_DESCRIPTIONS[prefix].replace("[i]", str(i))

    # Add custom column descriptions
    COLUMN_DESCRIPTIONS["dataset_name"] = COLUMN_DESCRIPTIONS[Column.DATASET]
    COLUMN_DESCRIPTIONS["area"] = COLUMN_DESCRIPTIONS[Column.SegData.AREA_PX_SQ]
    COLUMN_DESCRIPTIONS["perimeter"] = COLUMN_DESCRIPTIONS[Column.SegData.PERIMETER_PX]

    for manifest_name in get_available_dataframe_manifests():
        manifest = load_dataframe_manifest(manifest_name)

        for column_name in manifest.columns:
            # TODO: remove this manual skip
            if column_name in ("duration_minutes"):
                continue

            if not (
                manifest_name.startswith("cell_centered_features")
                or manifest_name.startswith("diffae_pca_features")
                or manifest_name.startswith("grid_based_features")
            ) and (
                column_name == Column.DiffAEData.POLAR_ANGLE
                or column_name == Column.DiffAEData.POLAR_RADIUS
                or column_name == Column.DiffAEData.PC3_FLIPPED
            ):
                continue

            if column_name in COLUMN_DESCRIPTIONS:
                manifest.columns[column_name] = COLUMN_DESCRIPTIONS[column_name]

        save_dataframe_manifest(manifest)
