import importlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import App, Group, Parameter, validators

app = App(
    help="Endothelial pipeline CLI",
    version_flags=[],
    default_parameter=Parameter(negative=()),
)

FIGURE_WORKFLOWS = Group("Figure Workflows", sort_key=0)
PRODUCTION_WORKFLOWS = Group("Production Workflows", sort_key=1)
DEVELOPMENT_WORKFLOWS = Group("Development Workflows", sort_key=2)
ARCHIVED_WORKFLOWS = Group("Archived Workflows", sort_key=3)

SETTINGS = Group("Settings", sort_key=100)
LOGGING = (SETTINGS, Group(validator=validators.MutuallyExclusive()))


def cli() -> None:
    """Entrypoint CLI."""

    app["--help"].group = SETTINGS
    app.meta.group_parameters = SETTINGS

    build_cli_group(FIGURE_WORKFLOWS, "figures", True)
    build_cli_group(PRODUCTION_WORKFLOWS, "production", True)
    build_cli_group(DEVELOPMENT_WORKFLOWS, "development", True)
    build_cli_group(ARCHIVED_WORKFLOWS, "archive", False)

    app.meta()


@app.meta.default
def entrypoint(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(alias="-v", group=LOGGING, show_default=False)] = False,
    debug: Annotated[bool, Parameter(alias="-vv", group=LOGGING, show_default=False)] = False,
    show_archive: Annotated[bool, Parameter(show_default=False)] = False,
    config: Path = Path("config.yaml"),
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
    config
        Path to user configuration file.
    """

    if debug:
        setup_logging(logging.DEBUG)
    elif verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    app.config = cyclopts.config.Yaml(config)  # type: ignore[assignment]

    for subapp in app.meta.subapps:
        if show_archive and subapp.group and subapp.group[0].name == ARCHIVED_WORKFLOWS.name:
            subapp.show = True

    app(tokens)


def build_cli_group(group: Group, directory: str, show: bool) -> None:
    """Create CLI command group by automatically importing modules from directory."""

    workflows_path = Path(__file__).resolve().parent / "workflows" / directory

    for module_path in workflows_path.glob("*py"):
        relative_path = module_path.relative_to(Path(__file__).resolve().parents[2])
        name = relative_path.stem.replace("_", "-")
        module = importlib.import_module(".".join(relative_path.with_suffix("").parts))
        app.command(name=name, group=group, show=show)(module.main)


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

    logger = logging.getLogger("src.endo_pipeline")
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


if __name__ == "__main__":
    cli()
