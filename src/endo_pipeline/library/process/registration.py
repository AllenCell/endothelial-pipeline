import logging
from typing import Literal

from endo_pipeline.configs import get_datasets_in_collection

FLUOR_CHANNEL = 0
BF_CHANNEL = 1

logger = logging.getLogger(__name__)


def get_paired_dataset_dict(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
) -> dict[str, list[str]]:
    """
    Get a dictionary of paired datasets for alignment with correct
    'target' and 'moving' labels.

    Parameters
    ----------
    dataset_pair_type
        The type of dataset pair to align, either "live_fixed" or "20X_40X".

    Returns
    -------
    :
        Dictionary with keys "target" and "moving" containing lists of dataset names.
    """

    if dataset_pair_type not in ["live_fixed", "20X_40X"]:
        logger.error(
            "Invalid dataset pair type: [ %s ]. Choose 'live_fixed' or '20X_40X'.",
            dataset_pair_type,
        )
        raise ValueError("Invalid dataset pair type. Choose 'live_fixed' or '20X_40X'.")

    # Get the list of datasets of the specified pair type.
    dataset_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")

    # Set dataset name flags for setting
    # "target" and "moving" images for alignment.
    if dataset_pair_type == "live_fixed":
        # for live/fixed pairs, the "target" image
        # for alignment is the pre-fixation (live) image
        # and the "moving" image is the post-fixation (fixed) image.
        target_flag = "PreFixation"
        moving_flag = "PostFixation"
    else:
        # for 20X/40X pairs, the "target" image is the 20X image
        # and the "moving" image is the 40x image.
        target_flag = "20X"
        moving_flag = "40X"
    dataset_pairs = {
        "target": [dataset_name for dataset_name in dataset_list if target_flag in dataset_name],
        "moving": [dataset_name for dataset_name in dataset_list if moving_flag in dataset_name],
    }
    return dataset_pairs
