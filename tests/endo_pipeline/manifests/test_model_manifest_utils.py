import json
from contextlib import nullcontext

import pytest

from endo_pipeline.manifests.model_manifest import ModelManifest
from endo_pipeline.manifests.model_manifest_io import get_model_manifest_dir
from endo_pipeline.manifests.model_manifest_utils import get_model_manifest_with_parameters


@pytest.mark.parametrize(
    "workflow,parameters,expected",
    [
        ("workflow_a", None, pytest.raises(ValueError)),
        ("workflow_b", None, nullcontext("B")),
        ("workflow_c", None, pytest.raises(LookupError)),
        ("workflow_a", {}, pytest.raises(ValueError)),
        ("workflow_b", {}, nullcontext("B")),
        ("workflow_a", {"param1": "A"}, nullcontext("A")),
        ("workflow_a", {"param1": "B"}, nullcontext("C")),
        ("workflow_a", {"param2": 2}, pytest.raises(ValueError)),
        ("workflow_a", {"param2": 3}, pytest.raises(LookupError)),
        ("workflow_a", {"param1": "A", "param2": 2}, nullcontext("A")),
        ("workflow_a", {"param1": "B", "param2": 2}, nullcontext("C")),
        ("workflow_b", {"param1": "B"}, nullcontext("B")),
    ],
)
def test_get_model_manifest_with_parameters(fs, workflow, parameters, expected):
    manifest_dir = get_model_manifest_dir()

    manifests = {
        "A": {"name": "A", "workflow": "workflow_a", "parameters": {"param1": "A", "param2": 2}},
        "B": {"name": "B", "workflow": "workflow_b", "parameters": {"param1": "B"}},
        "C": {"name": "C", "workflow": "workflow_a", "parameters": {"param1": "B", "param2": 2}},
    }

    fs.create_file(manifest_dir / "a.yaml", contents=json.dumps(manifests["A"]))
    fs.create_file(manifest_dir / "b.yaml", contents=json.dumps(manifests["B"]))
    fs.create_file(manifest_dir / "c.yaml", contents=json.dumps(manifests["C"]))

    with expected as e:
        manifest = get_model_manifest_with_parameters(workflow, parameters)
        assert manifest == ModelManifest(**manifests[e])
