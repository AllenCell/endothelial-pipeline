DESCRIPTION = "Short description of the workflow."
TAGS = ["tag1", "tag2", "notebook", "example"]

import logging

logger = logging.getLogger(__name__)

logger.debug(f"debug message from notebook")
logger.info(f"info message from notebook")
logger.warning(f"warn message from notebook")
logger.error(f"error message from notebook")
logger.critical(f"critical message from notebook")
