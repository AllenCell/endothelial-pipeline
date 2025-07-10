import logging

logger = logging.getLogger(__name__)

TAGS = ["fig1", "a", "b"]


def main() -> None:
    """
    Workflow to produce Figure 1B. Tags: fig1, a, b
    """

    logger.debug(f"debug message")
    logger.info(f"info message")
    logger.warning(f"warn message")
    logger.error(f"error message")
    logger.critical(f"critical message")
