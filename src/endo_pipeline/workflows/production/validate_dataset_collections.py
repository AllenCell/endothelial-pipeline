TAGS = ["validation", "datasets"]


def main() -> None:
    """Validate datasets in dataset collections."""

    import logging

    from endo_pipeline.configs import (
        validate_3d_flow_field_dataset_collection,
        validate_filtered_dataset_collection,
    )

    logger = logging.getLogger(__name__)

    logger.info("Starting validation of dataset collection configs")

    validate_filtered_dataset_collection("live", "20X", "3i")

    validate_3d_flow_field_dataset_collection()

    logger.info("Finished validation of dataset collection configs")
