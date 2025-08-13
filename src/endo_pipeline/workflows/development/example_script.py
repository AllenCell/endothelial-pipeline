TAGS = ["tag1", "tag2", "example"]


def main() -> None:
    """Short description of the script-style workflow."""

    import logging

    logger = logging.getLogger(__name__)

    logger.debug(f"debug message from script")
    logger.info(f"info message from script")
    logger.warning(f"warn message from script")
    logger.error(f"error message from script")
    logger.critical(f"critical message from script")
