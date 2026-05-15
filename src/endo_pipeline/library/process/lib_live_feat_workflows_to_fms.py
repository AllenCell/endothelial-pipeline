import logging

from endo_pipeline.manifests import load_model_manifest

logger = logging.getLogger(__name__)


def get_model_annotations_for_upload() -> dict:
    """Return dictionary of label-free nuclei Cellpose model info for FMS upload annotations."""

    model_name = "nuc_pred_labelfree"
    run_name = "finetuned_20250419"
    return {"model_manifest": load_model_manifest(model_name), "run_name": run_name}
