import re

from cyclopts import App
from termcolor import colored


def get_app_tags(app: App) -> list[str]:
    """Extract tags from app help message (defaults to docstring)."""

    return re.findall(r"#([a-z0-9\-]+)", app.help)


def print_colored_message(color: str, main_message: str, sub_message: str | None = None) -> None:
    """Print colored message."""

    print(colored(f"\N{BULLET} {main_message}", color, attrs=["bold"]))

    if sub_message is not None:
        print(colored(f"  {sub_message}", color))


def print_error_message(main_message: str, sub_message: str | None = None) -> None:
    """Print red error message."""

    print_colored_message("red", main_message, sub_message)


def print_warning_message(main_message: str, sub_message: str | None = None) -> None:
    """Print yellow warning message."""

    print_colored_message("yellow", main_message, sub_message)


def print_info_message(main_message: str, sub_message: str | None = None) -> None:
    """Print blue info message."""

    print_colored_message("blue", main_message, sub_message)


def check_workflow_tags(app):
    """Check for known warnings and errors based on workflow tags."""

    import os
    from pathlib import Path

    tags = get_app_tags(app)
    exit_on_error = False

    if "internal" in tags:
        warning = "This workflow should only be run internally"
        print_warning_message(warning)

    if "fms" in tags:
        try:
            from aicsfiles import FileManagementSystem  # noqa: F401
        except Exception:
            error = "Required dependencies for workflow could not be imported"
            solution = "Use 'uv sync --extra internal' to install the dependencies"
            print_error_message(error, solution)
            exit_on_error = True

    if "vast" in tags:
        if not Path("//allen/aics").exists():
            error = "Workflow requires access to the Vast mounted at /allen/aics"
            solution = "Make sure you are on the Allen Institute network and have the Vast mounted"
            print_error_message(error, solution)
            exit_on_error = True

    if "fms" in tags and os.name == "nt":
        warning = "FMS uploads do not work on a Windows machine"
        solution = "If this workflows needs to upload to FMS, run on a Linux or MacOS machine"
        print_warning_message(warning, solution)

    if "gpu" in tags:
        info = "This workflow should be run on an NVIDIA GPU"
        solution = "Append '-g NUM_GPUS' when running the workflow to make sure GPUs are visible"
        print_info_message(info, solution)

    if exit_on_error:
        exit(1)
