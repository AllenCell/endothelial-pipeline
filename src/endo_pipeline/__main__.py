import importlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import App, Group, Parameter

app = App(help="Endothelial pipeline CLI", default_parameter=Parameter(negative=()))


def cli():
    command_group = Group("Settings", sort_key=100)
    app["--help"].group = command_group
    app["--version"].group = command_group
    app.meta.group_parameters = command_group

    build_cli_group("Group A", "group_a", 0, True)
    build_cli_group("Group B", "group_b", 1, False)
    build_cli_group("Group C", "group_c", 2, False)

    app.meta()


@app.meta.default
def entry(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(alias="-v", show_default=False)] = False,
    show_b: Annotated[bool, Parameter(show_default=False)] = False,
    show_c: Annotated[bool, Parameter(show_default=False)] = False,
    config: Path = Path("config.yaml"),
):
    """Help string for this demo application.

    Parameters
    ----------
    verbose
        Show additional logging
    show_b
        Show available Group B workflows.
    show_c
        Show available Group C workflows.
    config
        Path to user configuration file.
    """

    if verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.WARNING)

    app.config = cyclopts.config.Yaml(config)

    for subapp in app.meta.subapps:
        if show_b and subapp.group and subapp.group[0].name == "Group B":
            subapp.show = True
        if show_c and subapp.group and subapp.group[0].name == "Group C":
            subapp.show = True

    app(tokens)


def build_cli_group(group_name: str, workflows_name: str, sort_key: int, show: bool):
    group = Group(group_name, sort_key=sort_key)
    workflows_path = Path(__file__).resolve().parent / "workflows" / workflows_name

    for module_path in workflows_path.glob("*py"):
        relative_path = module_path.relative_to(Path(__file__).resolve().parents[2])
        name = relative_path.stem.replace("_", "-")
        module = importlib.import_module(".".join(relative_path.with_suffix("").parts))
        app.command(name=name, group=group, show=show)(module.main)


class CustomStreamLoggingFormatter(logging.Formatter):
    def __init__(self):
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

    def format(self, record):
        log_format = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level):
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

    return logger


if __name__ == "__main__":
    cli()
