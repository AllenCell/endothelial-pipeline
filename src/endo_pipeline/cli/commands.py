import ast
import importlib
from pathlib import Path

from cyclopts import App, Group


def build_command_group(app: App, group: Group, directory: str, show: bool) -> None:
    """Create CLI command group by finding modules in a directory."""

    workflows_path = Path(__file__).resolve().parents[1] / "workflows" / directory

    for module_path in workflows_path.glob("*py"):
        relative_path = module_path.relative_to(Path(__file__).resolve().parents[3])
        workflow_name = relative_path.stem.replace("_", "-")
        module_name = ".".join(relative_path.with_suffix("").parts[1:])

        if workflow_name.endswith("-nb"):
            register_notebook_command(app, workflow_name, group, show, module_name, relative_path)
        else:
            register_script_command(app, workflow_name, group, show, module_name, relative_path)


def register_notebook_command(
    app: App, name: str, group: Group, show: bool, module: str, path: Path
) -> None:
    """Register a notebook-style module as command to the CLI."""

    # Rename workflow to remove the "nb" suffix
    name = name.replace("-nb", "")

    # Create wrapper around import (which "runs" the workflow when called)
    def module_wrapper() -> None:
        importlib.import_module(module)

    # Set help message based on docstring (if it exists)
    docstring = ast.get_docstring(ast.parse(path.read_text())) or ""
    module_wrapper.__doc__ = docstring

    # Add workflow command to pipeline
    app.command(module_wrapper, name=name, group=group, show=show)


def register_script_command(
    app: App, name: str, group: Group, show: bool, module: str, path: Path
) -> None:
    """Register a script-style module as command to the CLI."""

    # Check that main method exists in module
    main = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]

    # If main method does not exist, we cannot add it to the pipeline.
    if not main:
        raise NotImplementedError("Workflow must have a `main` method")

    # Add workflow command to pipeline
    app.command(f"{module}:main", name=name, group=group, show=show)
