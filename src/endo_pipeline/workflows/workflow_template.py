# ============================= IMPORT STATEMENTS ==============================
# Workflows are registered to the CLI by automatically importing the module.
# Because all workflows are registered each time the CLI is called and imports
# can be slow, it is recommended that you place import statements under the
# `main` method, rather than at the top of the module.
# ==============================================================================

# =============================== WORKFLOW TAGS ================================
# Workflows may optionally include a list of tags to categorize the workflow
# and group related workflows. These tags are automatically pulled from TAGS
# when registering workflows to the CLI. Users can then use the `--show-tags`
# flag to include these tags in the workflow descriptions or `--filter-tag=TAG`
# to filter and show only workflows with a specific tag `TAG`.
#
# For example, these tags would be displayed in the following help message:
#
#   ╭─ Workflow Category ────────────────────────────────────────────────────╮
#   │ workflow-template  tag1 | tag2 | Short description of the workflow.    │
#   ╰────────────────────────────────────────────────────────────────────────╯
#
# Recommendations: all lowercase, separate tags containing multiple words with
# hyphen (-), use no more than three words per tag, avoid more than three tags
# per workflow.
# ==============================================================================
TAGS = ["tag1", "tag2"]


# ============================== MAIN ENTRYPOINT ===============================
# Workflows must have a method called main in order to be registered by the CLI.
# The arguments, defaults, and docstring of this method are all automatically
# parsed to build the CLI command for calling the workflow.
#
# For example, this method will be parsed to the following help message:
#
#   Usage: endopipe workflow-template [ARGS] [OPTIONS]
#
#   Short description of the workflow.
#
#   Additional description for the workflow here. This text will be displayed as
#   part of the --help message.
#
#   ╭─ Parameters ───────────────────────────────────────────────────────────╮
#   │ *  PARAM1 --param1              Description for param 1. [required]    │
#   │ *  PARAM2 --param2              Description for param 2. [required]    │
#   │ *  PARAM3 --param3 --no-param3  Description for param 3. [required]    │
#   │    PARAM4 --param4              Description for param 4. [default: X]  │
#   ╰────────────────────────────────────────────────────────────────────────╯
#
# You can use the following patterns to indicate keyword-only or positional-only
# parameters:
#
#   def positional_or_keyword(a, b):
#      pass
#
#   def positional_only(a, b, /):
#      pass
#
#   def keyword_only(*, a, b=2):
#      pass
#
#   def mixture(a, /, b, *, c=3):
#      pass
#
# By default, parameters as treated as positional or keyword, which means the
# parameters can be passed in the CLI by position or by keyword:
#
#   endopipe workflow-template X 10 false Y
#   endopipe workflow-template --param1=X --param2=10 --no-param3 --param4=Y
# ==============================================================================
def main(param1: str, param2: int, param3: bool, param4: str = "X") -> None:
    """
    Short description of the workflow.

    Additional description for the workflow here. This text will be displayed as
    part of the `--help` message.

    Parameters
    ----------
    param1
        Description for param 1.
    param2
        Description for param 2.
    param3
        Description for param 3.
    param4
        Description for param 4.
    """

    # Place imports here instead of at the top of the module. This ensures that
    # imports are only imported when the main method is called.
    import logging

    from endo_pipeline import DEMO_MODE, NUM_GPUS

    logger = logging.getLogger(__name__)

    # Call workflow methods here. All methods should be located in the library,
    # config, or io packages. No methods should be defined in the workflow.
    logger.debug(f"debug message: {param1} {param2} {param3} {param4}")
    logger.info(f"info message: {param1} {param2} {param3} {param4}")
    logger.warning(f"warn message: {param1} {param2} {param3} {param4}")
    logger.error(f"error message: {param1} {param2} {param3} {param4}")
    logger.critical(f"critical message: {param1} {param2} {param3} {param4}")

    if DEMO_MODE:
        logger.info("Running in demo mode.")

    logger.info(f"Number of GPUs available: {NUM_GPUS}")


# =============================== WORKFLOW CLI =================================
# The following code enables the workflow to be run outside of the main pipeline
# CLI with some of the same settings, such as logging level.
#
#   uv run path/to/workflow.py (if using uv)
#   python path/to/workflow.py (if in activated virtual environment)
#
# You may drop this if you want the workflow to only be runnable from the main
# pipeline CLI.
# ==============================================================================
if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
