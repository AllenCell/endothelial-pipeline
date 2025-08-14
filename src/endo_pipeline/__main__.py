import importlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import App, Group, Parameter, validators
from rich.console import Console

logger = logging.getLogger("")

pipeline_app = App(
    help="Endothelial pipeline CLI",
    version_flags=[],
    default_parameter=Parameter(negative=()),
    console=Console(),
)

workflow_app = App(
    version_flags=[],
    usage="",
)

tags: dict[str, list[str]] = {}

EXTERNAL_LOGGERS = {
    "aicsfiles.client.http.http_client": logging.WARNING,
    "cyto_dl": logging.ERROR,
    "fsspec.local": logging.WARNING,
    "git.cmd": logging.WARNING,
    "h5py._conv": logging.WARNING,
    "lightning.pytorch": logging.WARNING,
    "lightning.pytorch.accelerators.cuda": logging.WARNING,
    "lightning.pytorch.utilities.rank_zero": logging.WARNING,
    "lightning.fabric.utilities": logging.WARNING,
    "numcodecs": logging.WARNING,
    "matplotlib": logging.WARNING,
    "torch": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
}

FIGURE_WORKFLOWS = Group("Figure Workflows", sort_key=0)
PRODUCTION_WORKFLOWS = Group("Production Workflows", sort_key=1)
DEVELOPMENT_WORKFLOWS = Group("Development Workflows", sort_key=2)
ARCHIVED_WORKFLOWS = Group("Archived Workflows", sort_key=3)

SETTINGS = Group("Settings", sort_key=100)
FLAGS = Parameter(negative="", show_default=False)
LOGGING = (SETTINGS, Group(validator=validators.MutuallyExclusive(), default_parameter=FLAGS))
OPTIONS = (SETTINGS, Group(default_parameter=FLAGS))


def pipeline_cli() -> None:
    """Pipeline CLI."""

    pipeline_app["--help"].group = SETTINGS
    pipeline_app.meta.group_parameters = SETTINGS

    build_cli_group(FIGURE_WORKFLOWS, "figures", True)
    build_cli_group(PRODUCTION_WORKFLOWS, "production", True)
    build_cli_group(DEVELOPMENT_WORKFLOWS, "development", True)
    build_cli_group(ARCHIVED_WORKFLOWS, "archive", False)

    pipeline_app.meta.default(pipeline_entrypoint)
    pipeline_app.meta()


def workflow_cli(workflow: Callable) -> None:
    """Workflow CLI."""

    workflow_app["--help"].group = SETTINGS
    workflow_app.meta.group_parameters = SETTINGS

    workflow_app.default(workflow)

    workflow_app.meta.default(workflow_entrypoint)
    workflow_app.meta()


def pipeline_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(alias="-v", group=LOGGING)] = False,
    debug: Annotated[bool, Parameter(alias="-vv", group=LOGGING)] = False,
    show_archive: Annotated[bool, Parameter(alias="-a", group=OPTIONS)] = False,
    show_tags: Annotated[bool, Parameter(alias="-t", group=OPTIONS)] = False,
    filter_tag: Annotated[str | None, Parameter(alias="-f")] = None,
    config: Annotated[Path, Parameter(alias="-c")] = Path("config.yaml"),
    run_with_gpu: Annotated[bool, Parameter(alias="-g", group=OPTIONS)] = False,
    show_external_logs: Annotated[bool, Parameter(alias="-s", group=OPTIONS)] = False,
    testing_mode: Annotated[bool, Parameter(alias="-x", group=OPTIONS)] = False,
) -> None:
    """
    Parameters
    ----------
    *tokens
        Workflow and workflow arguments.
    verbose
        Show verbose logging.
    debug
        Show debug logging.
    show_archive
        Show available archived workflows.
    show_tags
        Show all available workflow tags.
    filter_tag
        Filter workflows by given tag.
    config
        Path to user configuration file.
    run_with_gpu
        Run workflow with GPU settings.
    show_external_logs
        Show logging outputs from external libraries.
    testing_mode
        Run workflows in testing mode.
    """

    apply_entrypoint_settings(verbose, debug, run_with_gpu, show_external_logs, testing_mode)

    if config.read_text() != "":
        pipeline_app.config = cyclopts.config.Yaml(config)  # type: ignore[assignment]

    for app in pipeline_app.meta.subapps:
        if show_archive and app.group and app.group[0].name == ARCHIVED_WORKFLOWS.name:
            app.show = True

        if app.name[0] in tags:
            if show_tags:
                app.help = f"| {' | '.join(tags[app.name[0]])} | {app.help}"

            if filter_tag:
                app.show = filter_tag in tags[app.name[0]] and app.show

    pipeline_app(tokens)


def workflow_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(alias="-v", group=LOGGING)] = False,
    debug: Annotated[bool, Parameter(alias="-vv", group=LOGGING)] = False,
    run_with_gpu: Annotated[bool, Parameter(alias="-g", group=OPTIONS)] = False,
    show_external_logs: Annotated[bool, Parameter(alias="-s", group=OPTIONS)] = False,
    testing_mode: Annotated[bool, Parameter(alias="-x", group=OPTIONS)] = False,
) -> None:
    """
    Parameters
    ----------
    *tokens
        Workflow and workflow arguments.
    verbose
        Show verbose logging.
    debug
        Show debug logging.
    run_with_gpu
        Run workflow with GPU settings.
    show_external_logs
        Show logging outputs from external libraries.
    testing_mode
        Run workflows in testing mode.
    """

    apply_entrypoint_settings(verbose, debug, run_with_gpu, show_external_logs, testing_mode)

    workflow_app(tokens)


def apply_entrypoint_settings(
    verbose: bool = False,
    debug: bool = False,
    run_with_gpu: bool = False,
    show_external_logs: bool = False,
    testing_mode: bool = False,
):
    """
    Apply settings shared between pipeline and workflow entrypoints.

    Parameters
    ----------
    verbose
        Show verbose logging.
    run_with_gpu
        Run workflow with GPU settings.
    show_external_logs
        Show logging outputs from external libraries.
    testing_mode
        Run workflows in testing mode.
    """

    if debug:
        setup_logging(logging.DEBUG)
    elif verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    if run_with_gpu:
        setup_gpu()

    if not show_external_logs:
        silence_external_loggers(EXTERNAL_LOGGERS)

    if testing_mode:
        import src.endo_pipeline

        logger.info("Running workflows in testing mode")
        src.endo_pipeline.TESTING_MODE = True


def build_cli_group(group: Group, directory: str, show: bool) -> None:
    """Create CLI command group by automatically importing modules from directory."""

    workflows_path = Path(__file__).resolve().parent / "workflows" / directory

    for module_path in workflows_path.glob("*py"):
        relative_path = module_path.relative_to(Path(__file__).resolve().parents[2])
        name = relative_path.stem.replace("_", "-")
        module = importlib.import_module(".".join(relative_path.with_suffix("").parts))
        tags[name] = module.TAGS if hasattr(module, "TAGS") else []
        pipeline_app.command(name=name, group=group, show=show)(module.main)


class CustomStreamLoggingFormatter(logging.Formatter):
    """Custom class for formatting stream logging with colored levels."""

    def __init__(self) -> None:
        super().__init__()
        self.format_template = (
            "%(asctime)s - %(name)s - \033[COLORm%(levelname)s\033[0m - %(message)s"
        )
        self.formats = {
            logging.DEBUG: self.format_template.replace("COLOR", "37;1"),
            logging.INFO: self.format_template.replace("COLOR", "34;1"),
            logging.WARNING: self.format_template.replace("COLOR", "33;1"),
            logging.ERROR: self.format_template.replace("COLOR", "31;1"),
            logging.CRITICAL: self.format_template.replace("COLOR", "31;1;4"),
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format logging record with colored levels."""

        log_format = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level: int) -> None:
    """Set up logging handlers and assign logging levels."""

    logger.setLevel(logging.DEBUG)

    log_path = Path(__file__).resolve().parents[2] / "logs"
    log_path.mkdir(exist_ok=True)
    file_name = log_path / f"endo_pipeline_{datetime.now(tz=UTC).strftime('%Y%m%d')}.log"

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_formatter = CustomStreamLoggingFormatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def silence_external_loggers(external_loggers: dict) -> None:
    """
    Set external logger to a specific logging level to avoid excessive logging outputs.

    Parameters
    ----------
    external_loggers
        Dictionary of external loggers and their respective logging levels.
    """
    for logger_name, logging_level in external_loggers.items():
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(logging_level)


def setup_gpu() -> None:
    """Set up GPU environmental variables."""

    logger.info("Setting up environment to run workflow using GPU")

    import os
    import re
    import subprocess

    # Query to get free memory of available GPU devices
    command = ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,noheader,nounits"]
    gpu_memory_free = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode().strip()

    # If unable to access the driver, report error and exit
    if "failed" in gpu_memory_free:
        logger.error("Workflow is unable to communicate with the NVIDIA driver")
        raise EnvironmentError(gpu_memory_free)

    # Select device number with the maximum free memory
    gpu_options = [(int(free), gpu) for free, gpu in re.findall(r"(\d+), (\d+)", gpu_memory_free)]
    _, gpu_with_max_free = sorted(gpu_options, reverse=True)[0]

    # Set the CUDA_VISIBLE_DEVICES environment variable to selected GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_with_max_free
    logger.info("Setting CUDA_VISIBLE_DEVICES to [ %s ]", gpu_with_max_free)


if __name__ == "__main__":
    pipeline_cli()
