import logging

logger = logging.getLogger(__name__)

TAGS = ["tag1", "tag2", "tag4"]


def main(param1: str, param2: int, param3: bool, param4: str = "DEFAULT") -> None:
    """
    Description of production workflow 3.

    Parameters
    ----------
    param1
        Description for param 1.
    param2
        Description for param 2.
    param3
        Description for param 3
    param4
        Description for param 4.
    """

    logger.debug(f"debug message: {param1} {param2} {param3} {param4}")
    logger.info(f"info message: {param1} {param2} {param3} {param4}")
    logger.warning(f"warn message: {param1} {param2} {param3} {param4}")
    logger.error(f"error message: {param1} {param2} {param3} {param4}")
    logger.critical(f"critical message: {param1} {param2} {param3} {param4}")
