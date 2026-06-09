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

    # Expand prefix features
    for prefix in [Column.DiffAEData.LATENT_FEATURE_PREFIX, Column.DiffAEData.PCA_FEATURE_PREFIX]:
        for i in range(NUM_LATENT_FEATURES + 1):
            column_descriptions[f"{prefix}{i}"] = column_descriptions[prefix].replace("[i]", str(i))

    # Add custom column descriptions
    column_descriptions["dataset_name"] = column_descriptions[Column.DATASET]
    column_descriptions["area"] = column_descriptions[Column.SegData.AREA_PX_SQ]
    column_descriptions["perimeter"] = column_descriptions[Column.SegData.PERIMETER_PX]
    column_descriptions["edge_fluorescences (a.u.)"] = column_descriptions[
        Column.SegData.EDGE_FLUOR
    ]
    column_descriptions["node_fluorescences (a.u.)"] = column_descriptions[
        Column.SegData.NODE_FLUOR
    ]
    column_descriptions["cell_fluorescence_max (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_MAX
    ]
    column_descriptions["cell_fluorescence_mean (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_MEAN
    ]
    column_descriptions["cell_fluorescence_median (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_MEDIAN
    ]
    column_descriptions["cell_fluorescence_min (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_MIN
    ]
    column_descriptions["cell_fluorescence_pct25 (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_PCT25
    ]
    column_descriptions["cell_fluorescence_pct75 (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_PCT75
    ]
    column_descriptions["cell_fluorescence_std (a.u.)"] = column_descriptions[
        Column.SegData.CELL_FLUOR_STD
    ]
    column_descriptions["centroid_X"] = column_descriptions[Column.SegData.CENTROID_X]
    column_descriptions["centroid_Y"] = column_descriptions[Column.SegData.CENTROID_Y]

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

            if column_name in column_descriptions:
                manifest.columns[column_name] = column_descriptions[column_name]

        save_dataframe_manifest(manifest)
