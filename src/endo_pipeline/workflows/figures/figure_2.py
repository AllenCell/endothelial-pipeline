import logging

logger = logging.getLogger(__name__)

TAGS = ["fig2", "d"]


def main() -> None:
    """
    Workflow to produce Figure 2. Tags: fig1, d
    """

    logger.debug(f"debug message")
    logger.info(f"info message")
    logger.warning(f"warn message")
    logger.error(f"error message")
    logger.critical(f"critical message")
