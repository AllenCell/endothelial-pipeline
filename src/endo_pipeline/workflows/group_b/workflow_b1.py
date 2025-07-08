import logging

logger = logging.getLogger(__name__)


def main(arg1: str, arg2: int) -> None:
    """
    Description of workflow B1.

    Parameters
    ----------
    arg1
        Description for arg 1.
    arg2 : int
        Description for arg 2.
    """

    logger.debug(f"debug message: {arg1} {arg2}")
    logger.info(f"info message: {arg1} {arg2}")
    logger.warning(f"warn message: {arg1} {arg2}")
    logger.error(f"error message: {arg1} {arg2}")
    logger.critical(f"critical message: {arg1} {arg2}")
