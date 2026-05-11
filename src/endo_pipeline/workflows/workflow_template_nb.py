# ============================ WORKFLOW DESCRIPTION ============================
# Notebooks should include top level "docstring" that can be automatically
# parsed to populate the help message. It must be located at the beginning of
# the file.
#
# For example, this description would be displayed in the help message as:
#
#   ╭─ Workflow Category ────────────────────────────────────────────────────╮
#   │ workflow-template  Short description of the workflow.                  │
#   ╰────────────────────────────────────────────────────────────────────────╯
#
# ==============================================================================

"""
Short description of the workflow.

Additional description for the workflow here. This text will be displayed as
part of the `--help` message.

#tag1 #tag2
"""

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

# %%
import logging

from endo_pipeline.cli import DEMO_MODE

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
#   endopipe workflow-template
#   endopipe workflow-template -v
#   endopipe workflow-template -vv
# ==============================================================================
logger = logging.getLogger(__name__)

# %%

# ============================== MAIN ENTRYPOINT ===============================
# Notebook workflows are run by importing the module, which runs everything
# outside of an import guard. Parameters cannot be set from the CLI.
# ==============================================================================

# Call workflow methods here. All methods should be located in the library,
# config, or io packages. No methods should be defined in the workflow.
logger.debug("debug message")
logger.info("info message")
logger.warning("warn message")
logger.error("error message")
logger.critical("critical message")

# To support review and testing, you can use the DEMO_MODE flag to alter
# workflow behavior so that it runs faster. For example, if you are iterating
# through and processing a list of datasets, consider exiting the loop early
# when running in demo mode.
if DEMO_MODE:
    logger.info("Running in demo mode.")

# %%
