import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from cyclopts import App, Group, Parameter
from rich.console import Console

from endo_pipeline.cli.commands import build_command_group
from endo_pipeline.cli.gpu import setup_gpu
from endo_pipeline.cli.logs import setup_logging, silence_external_loggers
from endo_pipeline.cli.options import InternalOptions, PipelineOptions, WorkflowOptions
from endo_pipeline.cli.tags import check_workflow_tags, get_app_tags

IS_MAIN_PROCESS: bool = int(os.environ.get("LOCAL_RANK", "0")) == 0
"""True if the current process is the main process, False otherwise."""

IS_INTERNAL: bool = Path("//allen/aics").exists()
"""True if internal storage is reachable (proxy for internal use), False otherwise."""

logger = logging.getLogger("")

pipeline_app = App(
    help="Endothelial pipeline CLI",
    version_flags=[],
    console=Console(),
)

workflow_app = App(
    version_flags=[],
    usage="",
)

WORKFLOW_OPTIONS = WorkflowOptions()
PIPELINE_OPTIONS = PipelineOptions()
INTERNAL_OPTIONS = InternalOptions()

FIGURE_WORKFLOWS = Group("Figure Workflows", sort_key=0)
PRODUCTION_WORKFLOWS = Group("Production Workflows", sort_key=1)
DEVELOPMENT_WORKFLOWS = Group("Development Workflows", sort_key=2)
TESTING_WORKFLOWS = Group("Testing Workflows", sort_key=3)
INTERNAL_WORKFLOWS = Group("Internal Workflows", sort_key=4)


def build_command_groups() -> None:
    """Build command groups."""

    build_command_group(pipeline_app, FIGURE_WORKFLOWS, "figures", True)
    build_command_group(pipeline_app, PRODUCTION_WORKFLOWS, "production", True)

    if IS_INTERNAL:
        build_command_group(pipeline_app, DEVELOPMENT_WORKFLOWS, "development", True)
        build_command_group(pipeline_app, TESTING_WORKFLOWS, "testing", True)
        build_command_group(pipeline_app, INTERNAL_WORKFLOWS, "internal", True)


def pipeline_cli() -> None:
    """Pipeline CLI."""

    pipeline_app["--help"].group = "Options"

    build_command_groups()

    if IS_INTERNAL:
        pipeline_app.meta.default(pipeline_entrypoint_internal)
    else:
        pipeline_app.meta.default(pipeline_entrypoint_external)

    pipeline_app.meta()


def workflow_cli(workflow: Callable) -> None:
    """Workflow CLI."""

    import sys

    if hasattr(sys, "ps1"):
        # The ps1 string is only defined in interactive mode, so using it to
        # check for an interactive session. If detected, the workflow is called
        # directly, rather than passing through the CLI. Note that this approach
        # only works if the workflow does NOT require any arguments (i.e. the
        # workflow requires no arguments or all arguments have default values).
        logger.debug("Detected running in interactive shell")
        workflow()
    else:
        workflow_app["--help"].group = "Options"

        workflow_app.default(workflow)

        if IS_INTERNAL:
            workflow_app.meta.default(workflow_entrypoint_internal)
        else:
            workflow_app.meta.default(workflow_entrypoint_external)

        workflow_app.meta()


def pipeline_entrypoint_external(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
    pipeline_options: PipelineOptions = PIPELINE_OPTIONS,
) -> None:
    """External pipeline CLI entrypoint."""

    pipeline_entrypoint(
        *tokens,
        workflow_options=workflow_options,
        pipeline_options=pipeline_options,
        internal_options=None,
    )


def pipeline_entrypoint_internal(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
    pipeline_options: PipelineOptions = PIPELINE_OPTIONS,
    internal_options: InternalOptions = INTERNAL_OPTIONS,
) -> None:
    """Internal pipeline CLI entrypoint."""

    pipeline_entrypoint(
        *tokens,
        workflow_options=workflow_options,
        pipeline_options=pipeline_options,
        internal_options=internal_options,
    )


def pipeline_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions | None,
    pipeline_options: PipelineOptions | None,
    internal_options: InternalOptions | None,
) -> None:
    """Pipeline CLI entrypoint."""

    # If pipeline CLI is called with a workflow, check for known workflow errors
    if tokens:
        check_workflow_tags(pipeline_app[tokens[0]])

    # If workflow options are provided, apply them
    if workflow_options is not None:
        apply_workflow_options(workflow_options)

    # If internal options are provided, apply them
    if internal_options is not None:
        apply_internal_options(internal_options)

    # Only apply pipeline options if running the pipeline CLI without any
    # tokens. Otherwise, this call will cause all modules to be resolved (and
    # therefore trigger imports)
    if not tokens and pipeline_options is not None:
        apply_pipeline_options(pipeline_app._registered_commands, pipeline_options)

    pipeline_app(tokens)


def workflow_entrypoint_external(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
) -> None:
    """External workflow CLI entrypoint."""

    workflow_entrypoint(*tokens, workflow_options=workflow_options)


def workflow_entrypoint_internal(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
    internal_options: InternalOptions = INTERNAL_OPTIONS,
) -> None:
    """Internal workflow CLI entrypoint."""

    workflow_entrypoint(
        *tokens,
        workflow_options=workflow_options,
        internal_options=internal_options,
    )


def workflow_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions | None,
    internal_options: InternalOptions | None,
) -> None:
    """Workflow CLI entrypoint."""

    check_workflow_tags(workflow_app)

    # If workflow options are provided, apply them
    if workflow_options is not None:
        apply_workflow_options(workflow_options)

    # If internal options are provided, apply them
    if internal_options is not None:
        apply_internal_options(internal_options)

    workflow_app(tokens)


def apply_workflow_options(options: WorkflowOptions):
    """Apply options for running workflows."""

    import endo_pipeline.cli

    if options.debug:
        setup_logging(logging.DEBUG)
    elif options.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    if not options.show_external_logs:
        silence_external_loggers()

    if IS_MAIN_PROCESS:
        if options.num_gpus is not None and options.num_gpus > 0:
            endo_pipeline.cli.NUM_GPUS = setup_gpu(options.num_gpus)
        else:
            logger.info("Workflow running on CPU")
            endo_pipeline.cli.NUM_GPUS = None
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if options.demo_mode:
        logger.info("Running workflow in demo mode")
        endo_pipeline.cli.DEMO_MODE = True


def apply_internal_options(options: InternalOptions):
    """Apply internal options for running workflows."""

    import endo_pipeline.cli

    if options.upload_to_fms:
        logger.info("Uploading outputs to FMS (if applicable)")
        endo_pipeline.cli.UPLOAD_TO_FMS = True

    if options.use_staging:
        logger.info("Using staging environments")
        endo_pipeline.cli.FMS_ENVIRONMENT = "stg"
    else:
        endo_pipeline.cli.FMS_ENVIRONMENT = "prod"


def apply_pipeline_options(apps: dict[str, App], options: PipelineOptions) -> None:
    """Apply options for running pipeline."""

    for app in apps.values():
        tags = get_app_tags(app)

        if tags and options.show_tags:
            app.help = f"| {' | '.join(tags)} | {app.help}"

        if options.filter_tag:
            app.show = options.filter_tag in tags and app.show
