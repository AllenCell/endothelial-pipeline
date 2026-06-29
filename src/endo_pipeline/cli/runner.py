"""Methods for programmatically running CLI-enabled workflows."""

import asyncio
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field

from termcolor import colored

import endo_pipeline.cli
from endo_pipeline.cli.tags import get_app_tags

if typing.TYPE_CHECKING:
    from datetime import timedelta

    from cyclopts import App


logger = logging.getLogger(__name__)


@dataclass
class _TimerResult:
    elapsed: "timedelta | None"


@dataclass
class _TerminalOutput:
    last_line: str | None = None

    error_logs: list[str] = field(default_factory=list)


@dataclass
class _WorkflowResult:
    name: str
    timeout: int
    exception: Exception | str | None
    elapsed: "timedelta"

    @property
    def elapsed_minutes(self) -> float:
        from datetime import timedelta

        return self.elapsed / timedelta(minutes=1)

    @property
    def elapsed_str(self) -> str:
        return f"{self.elapsed_minutes:.1f}min"

    @property
    def failed(self) -> bool:
        return self.exception is not None and not isinstance(self.exception, TimeoutError)

    @property
    def timed_out(self) -> bool:
        return isinstance(self.exception, TimeoutError)

    @property
    def slow(self) -> bool:
        return not self.failed and self.elapsed_minutes > self.timeout

    @property
    def succeeded(self) -> bool:
        return not self.failed and not self.slow


@contextmanager
def _timer():
    from datetime import datetime

    start = datetime.now()
    result = _TimerResult(elapsed=None)
    try:
        yield result
    finally:
        end = datetime.now()
        result.elapsed = end - start


def _filter_workflows_by_tag(pipeline_app: "App", select_tag: str):
    """Filters workflows in app to only those containing the selected tag."""

    for name, app in pipeline_app.resolved_commands().items():
        tags = get_app_tags(app)

        # Skip any "run all" workflows
        if "run-all" in name:
            continue

        # Do not include the workflow if the selected tag is not found
        if select_tag not in tags:
            continue

        # Do not include the workflow if tagged GPU but GPUs are not specified
        if endo_pipeline.cli.NUM_GPUS is None and "gpu" in tags:
            continue

        yield name


def _make_command(app: str, workflow_name: str, force_demo_mode: bool) -> tuple[str, list[str]]:
    import sys

    # Get tokens list and drop the executable and name of the workflow
    tokens = [arg for arg in sys.argv[1:] if arg != workflow_name]

    # If forcing demo and demo mode is not in the tokens, add to the token list.
    # Otherwise, drop demo mode from the tokens
    if force_demo_mode:
        if "-d" not in tokens and "-vd" not in tokens and "--demo-mode" not in tokens:
            tokens.append("-d")
    else:
        tokens = [token for token in tokens if token not in ("-d", "--demo-mode")]

        if "-vd" in tokens:
            tokens.remove("-vd")
            tokens.append("-v")

    return (app, ["endopipe", app, *tokens])


async def _print_logs(last_line: _TerminalOutput, name: str, stream: asyncio.StreamReader | None):
    if stream is None:
        return

    async for line_bytes in stream:
        line: str = line_bytes.decode().rstrip()
        if line.startswith("╰────"):  # Cyclopts rich footer
            continue
        cyclopts_bar = "│"  # Cyclopts rich error output
        if line.startswith(cyclopts_bar) and line.endswith("│"):
            line = line[1:-1].strip()
        last_line.last_line = line
        if "ERROR" in line:
            last_line.error_logs.append(line.split("[0m - ")[-1])
        print(colored(f"[{name}]", "cyan"), line)


async def _wait_or_kill(
    process: asyncio.subprocess.Process, timeout: float, dependent_tasks: list[asyncio.Task]
):
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        for task in dependent_tasks:
            task.cancel()
    except (TimeoutError, asyncio.CancelledError) as e:
        logger.debug(f"Killing process due to {e.__class__.__name__}")
        process.kill()
        await process.wait()
        raise e


async def _run_workflow(last_line: _TerminalOutput, name: str, command: list[str], timeout: int):
    process = await asyncio.subprocess.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.TaskGroup() as tg:
            stdout_task = tg.create_task(_print_logs(last_line, name, process.stdout))
            stderr_task = tg.create_task(_print_logs(last_line, name, process.stderr))
            tg.create_task(_wait_or_kill(process, timeout * 60 * 2, [stdout_task, stderr_task]))
    except ExceptionGroup as group:
        raise group.exceptions[0] from group
    return process.returncode


async def _manage_workflow(name: str, command: list[str], timeout: int) -> _WorkflowResult:
    error: Exception | str | None = None
    with _timer() as timer:
        try:
            logger.info(f"Starting workflow: {' '.join(command)}")
            terminal_output = _TerminalOutput()
            return_code = await _run_workflow(terminal_output, name, command, timeout)
            if return_code is not None and return_code != 0:
                error = terminal_output.last_line or "Failed. See logs."
            elif terminal_output.error_logs:
                num_errors = len(terminal_output.error_logs)
                if num_errors > 1:
                    error = terminal_output.error_logs[0] + f" (... and {num_errors - 1} more)"
                else:
                    error = terminal_output.error_logs[0]
        except Exception as e:
            error = e
    elapsed = timer.elapsed
    assert elapsed is not None
    return _WorkflowResult(name=name, timeout=timeout, exception=error, elapsed=elapsed)


def summarize_workflow_run_results(results: list[_WorkflowResult]):
    """Print summary of successful/failed workflows, with stacktraces."""

    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    successes = [result for result in results if result.succeeded]
    too_slows = [result for result in results if result.slow]
    failures = [result for result in results if result.failed]

    success_count = Text()
    too_slow_count = Text()
    failure_count = Text()

    if len(successes) > 0:
        success_count = Text(f"{len(successes)} succeeded ", "green bold")

    if len(too_slows) > 0:
        too_slow_count = Text(f"{len(too_slows)} slow workflows ", "yellow bold")

    if len(failures) > 0:
        failure_count = Text(f"{len(failures)} failed ", "red bold")

    if len(failures) > 0:
        final_color = "red"
    elif len(too_slows) > 0:
        final_color = "yellow"
    else:
        final_color = "green"

    banner = Text("━" * 10, final_color)
    summary = Text.assemble(banner, " ", success_count, too_slow_count, failure_count, banner)
    table = Table(title=summary)

    table.add_column("Workflow", style="bold")
    table.add_column("Result")
    table.add_column("Elapsed")
    table.add_column("Message")

    for result in successes:
        table.add_row(result.name, "succeeded", result.elapsed_str, style="green")

    for result in too_slows:
        if result.timed_out:
            table.add_row(result.name, "cancelled", Text(result.elapsed_str, "red"), style="yellow")
        else:
            table.add_row(result.name, "succeeded", result.elapsed_str, style="yellow")

    for result in failures:
        exception = Text.from_ansi(str(result.exception)).split("\n")[-1]
        table.add_row(result.name, "failed", result.elapsed_str, exception, style="red")

    console = Console()
    console.print()  # Line break before summary
    console.print(table)


async def run_all_workflows_with_tag(
    workflow_name: str, select_tag: str, runner_timeout: int, force_demo_mode: bool
) -> list[_WorkflowResult]:
    """Runs all workflows with the given tag."""

    from endo_pipeline.cli.apps import pipeline_app

    commands = [
        _make_command(app, workflow_name, force_demo_mode)
        for app in _filter_workflows_by_tag(pipeline_app, select_tag)
    ]

    # Use semaphore to set max concurrency to 5. This means that no more
    # than five workflows will be run at the same time, which helps avoid
    # overwhelming the system
    semaphore = asyncio.Semaphore(5)

    async def _manage_workflow_semaphore(name: str, command: list[str], timeout: int):
        async with semaphore:
            return await _manage_workflow(name, command, timeout)

    return await asyncio.gather(
        *[_manage_workflow_semaphore(name, command, runner_timeout) for name, command in commands]
    )
