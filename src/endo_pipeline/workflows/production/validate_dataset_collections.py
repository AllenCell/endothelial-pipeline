TAGS = ["validation", "datasets"]


def main() -> None:
    """Validate datasets in dataset collections."""

    import logging

    from src.endo_pipeline.configs import validate_filtered_dataset_collection

    logger = logging.getLogger(__name__)

    logger.info("Starting validation of dataset configs")

    validate_filtered_dataset_collection("live", "20X", "3i")

    logger.info("Finished validation of dataset configs")
