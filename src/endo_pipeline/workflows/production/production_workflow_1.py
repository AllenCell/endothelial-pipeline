import logging

logger = logging.getLogger(__name__)

TAGS = ["a", "d"]


def main() -> None:
    """
    Description of production workflow 1. Example using no params. Tags: a, d

    Parameters
    ----------
    param1
        Description for param 1.
    param2
        Description for param 2.
    """

    logger.debug(f"debug message")
    logger.info(f"info message")
    logger.warning(f"warn message")
    logger.error(f"error message")
    logger.critical(f"critical message")
