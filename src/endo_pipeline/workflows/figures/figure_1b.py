import logging

logger = logging.getLogger(__name__)

TAGS = ["fig1", "tag2", "tag3"]


def main() -> None:
    """
    Workflow to produce Figure 1B.
    """

    logger.debug(f"debug message")
    logger.info(f"info message")
    logger.warning(f"warn message")
    logger.error(f"error message")
    logger.critical(f"critical message")
