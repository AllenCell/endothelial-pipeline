import os

DEMO_MODE = False
"""True if workflows should be run in demo mode, False otherwise."""

USE_STAGING = False
"""True to use staging environments, False otherwise."""

NUM_GPUS: int | None = None
"""Number of GPUs available to use. None if no GPUs are available."""

IS_MAIN_PROCESS: bool = int(os.environ.get("LOCAL_RANK", "0")) == 0
"""True if the current process is the main process, False otherwise."""

# RUN_NAME: str = os.environ.get("RUN_NAME", "default_run_name")

# CONFIG_PATH: str = os.environ.get("CONFIG_PATH", "deafault_config_path")
