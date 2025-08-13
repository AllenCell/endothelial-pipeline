# ============================ WORKFLOW DESCRIPTION ============================
# Notebooks do not have a "docstring" that can be automatically parsed to
# populate the help message. Instead, these workflows may include a description
# string used to populate the help message of the CLI.
#
# For example, this description would be displayed in the help message as:
#
#   ╭─ Workflow Category ────────────────────────────────────────────────────╮
#   │ workflow-template  Short description of the workflow.                  │
#   ╰────────────────────────────────────────────────────────────────────────╯
#
# If a description is not provided, the workflow will default to the following:
#
#   ╭─ Workflow Category ────────────────────────────────────────────────────╮
#   │ workflow-template  Run notebook workflow_template_nb.py                │
#   ╰────────────────────────────────────────────────────────────────────────╯
#
# ==============================================================================
DESCRIPTION = "Short description of the workflow."

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
# Notebook workflows are run by importing the module, which runs everything
# outside of an import guard. Parameters cannot be set from the CLI.
# ==============================================================================

import logging

logger = logging.getLogger(__name__)

# Call workflow methods here. All methods should be located in the library,
# config, or io packages. No methods should be defined in the workflow.
logger.debug(f"debug message")
logger.info(f"info message")
logger.warning(f"warn message")
logger.error(f"error message")
logger.critical(f"critical message")
