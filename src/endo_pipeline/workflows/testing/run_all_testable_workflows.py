"""
How to register your workflow for testing:

1. Make sure your workflow runs in under 5 minutes in `DEMO_MODE`
2. Add #test-ready (along with #gpu or #cpu-only) to your docstring
3. Run all registered workflows with:

.. code-block:: bash

    # runs all testable workflow
    uv run endopipe -g 1 run-all-testable-workflows

    # run only the CPU workflows
    uv run endopipe run-all-testable-workflows
"""

import asyncio
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass

from termcolor import colored

import endo_pipeline.cli
from endo_pipeline.cli.tags import get_app_tags
from endo_pipeline.settings.testing import TIMEOUT_MIN

if typing.TYPE_CHECKING:
    from datetime import timedelta

    from cyclopts import App


logger = logging.getLogger(__name__)


def _testable_workflows(pipeline_app: "App"):
    for name, app in pipeline_app.resolved_commands().items():
        tags = get_app_tags(app)

        if "test-ready" not in tags:
            continue

        if "cpu-only" not in tags and "gpu" not in tags:
            raise ValueError(
                f"Workflow '{name}' tagged #test-ready, but does not have a #gpu or #cpu-only tag."
            )

        if "cpu-only" in tags and "gpu" in tags:
            raise ValueError(
                f"Workflow '{name}' tagged #test-ready, but has both #gpu and #cpu-only tags."
            )

        if endo_pipeline.cli.NUM_GPUS is None and "gpu" in tags:
            continue

        yield name


@dataclass
class _TimerResult:
    elapsed: "timedelta | None"


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


@dataclass
class _WorkflowResult:
    name: str
    exception: Exception | None
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
        return not self.failed and self.elapsed_minutes > TIMEOUT_MIN

    @property
    def succeeded(self) -> bool:
        return not self.failed and not self.slow


def _make_command(app: str) -> tuple[str, list[str]]:
    import sys

    # Get tokens list and drop the executable and name of the workflow.
    tokens = [arg for arg in sys.argv[1:] if arg != "run-all-testable-workflows"]

    # If "demo mode" is not in tokens, add to the token list.
    if "-d" not in tokens and "--demo-mode" not in tokens:
        tokens.append("-d")

    return (app, ["endopipe", app, *tokens])


@dataclass
class _LastLine:
    text: str | None = None


async def _print_logs(last_line: _LastLine, name: str, stream: asyncio.StreamReader):
    async for line_bytes in stream:
        line: str = line_bytes.decode().rstrip()
        if line.startswith("╰────"):  # Cyclopts rich footer
            continue
        cyclopts_bar = "│"  # Cyclopts rich error output
        if line.startswith(cyclopts_bar) and line.endswith("│"):
            line = line[1:-1].strip()
        last_line.text = line
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


async def _run_workflow(last_line: _LastLine, name: str, command: list[str]):
    process = await asyncio.subprocess.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.TaskGroup() as tg:
            stdout_task = tg.create_task(_print_logs(last_line, name, process.stdout))
            stderr_task = tg.create_task(_print_logs(last_line, name, process.stderr))
            tg.create_task(_wait_or_kill(process, TIMEOUT_MIN * 60 * 2, [stdout_task, stderr_task]))
    except ExceptionGroup as group:
        raise group.exceptions[0] from group
    return process.returncode


async def _manage_workflow(name: str, command: list[str]) -> _WorkflowResult:
    error = None
    with _timer() as timer:
        try:
            logger.info(f"Starting workflow: {' '.join(command)}")
            last_line = _LastLine()
            return_code = await _run_workflow(last_line, name, command)
            if return_code is not None and return_code != 0:
                error = last_line.text if last_line.text is not None else "Failed. See logs."
        except Exception as e:
            error = e
    elapsed = timer.elapsed
    assert elapsed is not None
    return _WorkflowResult(name=name, exception=error, elapsed=elapsed)


def _summarize(results: list[_WorkflowResult]):
    """
    Print a summary of successful/failed workflows, with stacktraces
    """
    successes = [result for result in results if result.succeeded]
    too_slows = [result for result in results if result.slow]
    failures = [result for result in results if result.failed]
    success_count = ""
    too_slow_count = ""
    failure_count = ""

    print()  # Line break before summary
    if len(successes) > 0:
        print(colored("====== Successes ======", "green"))
        success_count = f"{len(successes)} succeeded "
    for result in successes:
        print(colored(result.name, "green"), colored(result.elapsed_str, "yellow"))

    if len(too_slows) > 0:
        print(colored("====== Too slow ======", "yellow"))
        too_slow_count = f"{len(too_slows)} slow workflows "
    for result in too_slows:
        if result.timed_out:
            print(colored(f"{result.name} canceled after {result.elapsed_str}", "red"))
        else:
            print(colored(f"{result.name} succeeded after {result.elapsed_str}", "yellow"))

    if len(failures) > 0:
        print(colored("====== Failures ======", "red"))
        failure_count = f"{len(failures)} failed "
    for result in failures:
        print(colored(f"{result.name} {result.exception}", "red"))

    if len(failures) > 0:
        final_color = "red"
    elif len(too_slows) > 0:
        final_color = "yellow"
    else:
        final_color = "green"
    print(
        "".join(
            [
                colored("====== ", final_color),
                colored(success_count, "green"),
                colored(too_slow_count, "yellow"),
                colored(failure_count, "red"),
                colored("======", final_color),
            ]
        ),
    )


async def _run_all(pipeline_app: "App") -> list[_WorkflowResult]:
    commands = [_make_command(app) for app in _testable_workflows(pipeline_app)]
    return await asyncio.gather(*[_manage_workflow(name, command) for name, command in commands])


def main():
    """Run all workflows marked as test ready."""

    from endo_pipeline.cli.apps import pipeline_app

    logger = logging.getLogger(__name__)

    if not endo_pipeline.cli.DEMO_MODE:
        endo_pipeline.cli.DEMO_MODE = True
        logger.debug("Forcing demo mode on for testing.")

    results = asyncio.run(_run_all(pipeline_app))
    _summarize(results)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
