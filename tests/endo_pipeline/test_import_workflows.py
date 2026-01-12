import endo_pipeline.__main__ as main


def test_import_workflows():
    # The following line is equivalent to `uv run endopipe`. This will import
    # all the workflow modules registered to the CLI, which can error if some
    # are misconfigured: e.g., incorrect top-level imports.
    main.build_pipeline_app()
