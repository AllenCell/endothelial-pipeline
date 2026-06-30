from dataclasses import dataclass
from typing import Annotated

from cyclopts import Group, Parameter, validators

FLAGS = Parameter(negative="", show_default=False)
OPTIONS = Group("Options", sort_key=100, default_parameter=FLAGS)
LOGGING_OPTIONS = (OPTIONS, Group(validator=validators.MutuallyExclusive()))


@Parameter(name="*")
@dataclass
class WorkflowOptions:
    """CLI options for workflows."""

    verbose: Annotated[bool, Parameter(alias="-v", group=LOGGING_OPTIONS)] = False
    """Show verbose logging."""

    debug: Annotated[bool, Parameter(alias="-vv", group=LOGGING_OPTIONS)] = False
    """Show debug logging."""

    show_external_logs: Annotated[bool, Parameter(alias="-s", group=OPTIONS)] = False
    """Show logging outputs from external libraries."""

    num_workers: Annotated[int | None, Parameter(alias="-n", group=OPTIONS)] = None
    """Number of workers to use for workflow execution."""

    num_gpus: Annotated[int | None, Parameter(alias="-g", group=OPTIONS)] = None
    """Number of GPUs to use for workflow execution. Use CPU if not provided."""

    demo_mode: Annotated[bool, Parameter(alias="-d", group=OPTIONS)] = False
    """Run workflows in demo mode."""


@Parameter(name="*")
@dataclass
class InternalOptions:
    """CLI options for workflows (for internal use only)."""

    upload_to_fms: Annotated[bool, Parameter(group=OPTIONS)] = False
    """Upload workflow outputs to FMS."""

    use_staging: Annotated[bool, Parameter(alias="-u", group=OPTIONS)] = False
    """Use staging FMS environment."""


@Parameter(name="*")
@dataclass
class PipelineOptions:
    """CLI options for pipeline."""

    show_tags: Annotated[bool, Parameter(alias="-t", group=OPTIONS)] = False
    """Show all available workflow tags."""

    filter_tag: Annotated[str | None, Parameter(alias="-f", group=OPTIONS)] = None
    """Filter workflows by given tag."""
