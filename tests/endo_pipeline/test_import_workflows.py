import pytest

from endo_pipeline.cli import apps


def test_import_workflows():
    # The following lines are equivalent to `uv run endopipe`. This will import
    # all the workflow modules registered to the CLI, which can error if some
    # are misconfigured: e.g., incorrect top-level imports.
    with pytest.raises(SystemExit):
        apps.build_command_groups()
        apps.pipeline_app([])
