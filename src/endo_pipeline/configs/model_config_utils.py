import logging

logger = logging.getLogger(__name__)


def get_labelfree_nuclei_prediction_model_name() -> str:
    """Get the name of the label-free nuclei prediction model."""

    return "nuc_pred_labelfree_finetuned_20250419"
