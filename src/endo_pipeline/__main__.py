import importlib
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from cyclopts import App, Group, Parameter, validators
from rich.console import Console

from endo_pipeline import IS_MAIN_PROCESS
from endo_pipeline.cli.options import PipelineOptions, WorkflowOptions

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
    "matplotlib": logging.ERROR,
    "torch": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
}

FIGURE_WORKFLOWS = Group("Figure Workflows", sort_key=0)
PRODUCTION_WORKFLOWS = Group("Production Workflows", sort_key=1)
DEVELOPMENT_WORKFLOWS = Group("Development Workflows", sort_key=2)
ARCHIVED_WORKFLOWS = Group("Archived Workflows", sort_key=3)

WORKFLOW_OPTIONS = WorkflowOptions()
PIPELINE_OPTIONS = PipelineOptions()


def pipeline_cli() -> None:
    """Pipeline CLI."""

    pipeline_app["--help"].group = "Options"

    build_cli_group(FIGURE_WORKFLOWS, "figures", True)
    build_cli_group(PRODUCTION_WORKFLOWS, "production", True)
    build_cli_group(DEVELOPMENT_WORKFLOWS, "development", True)
    build_cli_group(ARCHIVED_WORKFLOWS, "archive", False)

    pipeline_app.meta.default(pipeline_entrypoint)
    pipeline_app.meta()


def workflow_cli(workflow: Callable) -> None:
    """Workflow CLI."""

    workflow_app["--help"].group = "Options"

    workflow_app.default(workflow)

    workflow_app.meta.default(workflow_entrypoint)
    workflow_app.meta()


def pipeline_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
    pipeline_options: PipelineOptions = PIPELINE_OPTIONS,
) -> None:
    """Pipeline CLI entrypoint."""

    apply_workflow_options(workflow_options)

    for app in pipeline_app.meta.subapps:
        if (
            pipeline_options.show_archive
            and app.group
            and app.group[0].name == ARCHIVED_WORKFLOWS.name
        ):
            app.show = True

        if app.name[0] in tags:
            if pipeline_options.show_tags:
                app.help = f"| {' | '.join(tags[app.name[0]])} | {app.help}"

            if pipeline_options.filter_tag:
                app.show = pipeline_options.filter_tag in tags[app.name[0]] and app.show

    pipeline_app(tokens)


def workflow_entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    workflow_options: WorkflowOptions = WORKFLOW_OPTIONS,
) -> None:
    """Workflow CLI entrypoint."""

    apply_workflow_options(workflow_options)

    workflow_app(tokens)


def apply_workflow_options(options: WorkflowOptions):
    """Apply options for running workflows."""

    import endo_pipeline

    if options.debug:
        setup_logging(logging.DEBUG)
    elif options.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    if not options.show_external_logs:
        silence_external_loggers(EXTERNAL_LOGGERS)

    if IS_MAIN_PROCESS:
        if options.num_gpus is not None and options.num_gpus > 0:
            endo_pipeline.NUM_GPUS = setup_gpu(options.num_gpus)
        else:
            logger.info("Workflow running on CPU")
            endo_pipeline.NUM_GPUS = None
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if options.demo_mode:
        logger.info("Running workflow in demo mode")
        endo_pipeline.DEMO_MODE = True

    if options.use_staging:
        logger.info("Using staging environments")
        endo_pipeline.USE_STAGING = True


def build_cli_group(group: Group, directory: str, show: bool) -> None:
    """Create CLI command group by automatically importing modules from directory."""

    workflows_path = Path(__file__).resolve().parent / "workflows" / directory

    for module_path in workflows_path.glob("*py"):
        relative_path = module_path.relative_to(Path(__file__).resolve().parents[2])
        workflow_name = relative_path.stem.replace("_", "-")
        module_name = ".".join(relative_path.with_suffix("").parts[1:])

        if workflow_name.endswith("-nb"):
            register_notebook_to_cli(workflow_name, group, show, module_name, relative_path)
        else:
            register_script_to_cli(workflow_name, group, show, module_name)


def register_notebook_to_cli(name: str, group: Group, show: bool, module: str, path: Path) -> None:
    """Register a notebook-style module to the pipeline CLI."""

    # Rename workflow to remove the "nb" suffix
    name = name.replace("-nb", "")

    # Create wrapper around import (which "runs" the workflow when called)
    def module_wrapper():
        importlib.import_module(module)

    # Set help message based on DESCRIPTION variable (if it exists)
    description_match = re.findall(r'DESCRIPTION = "([\w\.\s+]+)"', path.read_text())
    default_doc = f"Run notebook ``{path.name}``"
    module_wrapper.__doc__ = description_match[0] if description_match else default_doc

    # Set tags based on TAGS variable (if it exists)
    tag_match = re.findall(r'TAGS = \[([\w\-, "]+)\]', path.read_text())
    tags[name] = re.findall(r'"([\w\-]+)"', tag_match[0]) if tag_match else []

    # Add workflow command to pipeline
    pipeline_app.command(name=name, group=group, show=show)(module_wrapper)


def register_script_to_cli(name: str, group: Group, show: bool, module: str):
    """Register a script-style module to the pipeline CLI."""

    # Dynamically import the module to access the main function
    module_import = importlib.import_module(module)

    # Set workflow tags based on TAGS variable (if it exists)
    tags[name] = module_import.TAGS if hasattr(module_import, "TAGS") else []

    # Add workflow command to pipeline
    pipeline_app.command(name=name, group=group, show=show)(module_import.main)


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

    # Only rank 0 should create log directory and file handler
    if IS_MAIN_PROCESS:

        log_path = Path(__file__).resolve().parents[2] / "logs"
        log_path.mkdir(exist_ok=True)
        file_name = log_path / f"endo_pipeline_{datetime.now(tz=UTC).strftime('%Y%m%d')}.log"

        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    stream_formatter = CustomStreamLoggingFormatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)


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


def setup_gpu(num_gpus: int | None) -> int | None:
    """
    Set up the GPU environment for workflow.

    Picks the GPUs with most free memory or sets CUDA_VISIBLE_DEVICES for multi-GPU & MIG,
    using the number of GPUs specified by the user.

    Parameters
    ----------
    num_gpus
        Number of GPUs to use with the workflow.
    """
    import os
    import re
    import subprocess
    import time

    logger.info("Setting up environment to run workflow using %d GPU(s)", num_gpus)

    # Detect MIG devices
    mig_output = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE).stdout.decode()
    is_mig = "MIG" in mig_output
    mig_uuids = re.findall(r"UUID: (MIG-[a-f0-9-]+)", mig_output)

    if is_mig:
        logger.info("MIG detected.")
        if num_gpus > 1:
            logger.error("Cannot use DDP with MIG devices. Only one MIG device can be used.")
            raise RuntimeError("Cannot use DDP with MIG devices.")
        if not mig_uuids:
            logger.error("MIG partitioning detected, but no UUIDs seen! No MIG UUIDs found.")
            raise RuntimeError("No MIG UUIDs found, but MIG is enabled.")
        selected_uuid = mig_uuids[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_uuid
        logger.info("Using MIG UUID: %s", selected_uuid)
        logger.info("Set CUDA_VISIBLE_DEVICES to [ %s ]", selected_uuid)

        return 1

    # Not MIG: Pick by available GPUs and free memory
    mem_info = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
        )
        .stdout.decode()
        .strip()
    )
    gpu_avail = re.findall(r"(\d+), (\d+)", mem_info)  # (memory_free, gpu_index)
    if not gpu_avail:
        logger.error("No GPUs available (nvidia-smi did not return any GPU info).")
        raise RuntimeError("No GPUs available for training.")

    # Sort by free memory, descending, get the indices
    gpu_avail_sorted = sorted(gpu_avail, key=lambda x: int(x[0]), reverse=True)
    chosen_gpus = [g[1] for g in gpu_avail_sorted[:num_gpus]]
    available_indices = [int(g[1]) for g in gpu_avail]

    logger.info("Available GPU indices: %s", available_indices)
    logger.info("Selecting %d GPU(s): %s", num_gpus, chosen_gpus)

    if num_gpus > len(available_indices):
        logger.warning(
            "Requested %d devices, but only %d available. Using all available.",
            num_gpus,
            len(available_indices),
        )
        chosen_gpus = [g[1] for g in gpu_avail_sorted]

    devs_str = ",".join(chosen_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = devs_str
    logger.info("Set CUDA_VISIBLE_DEVICES to [ %s ]", devs_str)

    return len(chosen_gpus)


if __name__ == "__main__":
    pipeline_cli()
