# ============================= IMPORT STATEMENTS ==============================
# Workflows are registered to the CLI by automatically importing the module.
# Because all workflows are registered each time the CLI is called and imports
# can be slow, it is recommended that you place import statements under the
# `main` method, rather than at the top of the module.
# ==============================================================================


# ================================== LOGGING ===================================
# We recommend using logging instead of print statements in almost all cases.
# Logging allows you to specify the "severity" of the message, which can be
# controlled using the `--verbose` and `--debug` flags. These logs are also
# saved to the `logs` folder.
#
# By default, workflows show logs at the WARNING, ERROR, and SEVERE levels. With
# the `--verbose` flag, workflows will also show logs at the INFO level. With
# the `--debug` flag, workflows will additionally show logs at the DEBUG level.
#
#   endopipe workflow-template X 10 false Y
#   endopipe workflow-template X 10 false Y -v
#   endopipe workflow-template X 10 false Y -vv
# ==============================================================================


# =============================== WORKFLOW TAGS ================================
# Workflows may optionally include a list of tags to categorize the workflow and
# group related workflows. These tags are automatically found by parsing the
# docstring for any text that matches the pattern #[a-z0-9\-].
#
# Users can then use the `--show-tags` flag to include these tags in the
# workflow descriptions or `--filter-tag=TAG` to filter and show only workflows
# with a specific tag `TAG`.
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
#   #tag1 #tag2
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

    #tag1 #tag2

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

    # The DEMO_MODE variable is False by default, and set to True when the user
    # passes `--demo-mode` or `-d` to the CLI. Note that this only works if the
    # import is inside the main method.
    from endo_pipeline.cli import DEMO_MODE

    logger = logging.getLogger(__name__)

    # Call workflow methods here. All methods should be located in the library,
    # config, manifest, or io packages. Avoid defining methods in the workflow.
    logger.debug(f"debug message: {param1} {param2} {param3} {param4}")
    logger.info(f"info message: {param1} {param2} {param3} {param4}")
    logger.warning(f"warn message: {param1} {param2} {param3} {param4}")
    logger.error(f"error message: {param1} {param2} {param3} {param4}")
    logger.critical(f"critical message: {param1} {param2} {param3} {param4}")

    # To support review and testing, you can use the DEMO_MODE flag to alter
    # workflow behavior so that it runs faster. For example, if you are
    # iterating through and processing a list of datasets, consider exiting the
    # loop early when running in demo mode.
    if DEMO_MODE:
        logger.info("Running in demo mode.")


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
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
