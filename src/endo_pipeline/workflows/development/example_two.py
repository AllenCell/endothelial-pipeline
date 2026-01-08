import time

print("IMPORTING WORKFLOW 2")
time.sleep(10)


def main() -> None:
    """Short description of the workflow."""

    import logging

    logger = logging.getLogger(__name__)
    logger.info("Running workflow 2")
