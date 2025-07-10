import logging

logger = logging.getLogger(__name__)

TAGS = ["tag1"]


def main(param1: str, param2: int) -> None:
    """
    Description of archived workflow 3.

    Parameters
    ----------
    param1
        Description for param 1.
    param2
        Description for param 2.
    """

    logger.debug(f"debug message: {param1} {param2}")
    logger.info(f"info message: {param1} {param2}")
    logger.warning(f"warn message: {param1} {param2}")
    logger.error(f"error message: {param1} {param2}")
    logger.critical(f"critical message: {param1} {param2}")
