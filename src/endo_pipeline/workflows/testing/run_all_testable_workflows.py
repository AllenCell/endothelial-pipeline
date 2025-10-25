import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass

import endo_pipeline

if typing.TYPE_CHECKING:
    from datetime import timedelta

    from cyclopts import App

logger = logging.getLogger(__name__)
if not endo_pipeline.DEMO_MODE:
    endo_pipeline.DEMO_MODE = True
    logger.debug("Forcing demo mode on for testing.")


def _testable_workflows(pipeline_app: "App", tags: dict[str, list[str]]):
    from endo_pipeline.cli.tags import CPU_ONLY, GPU, TEST_READY

    for app in pipeline_app.meta.subapps:
        name = app.name[0]
        if name not in tags:
            continue
        these_tags = tags[name]
        if TEST_READY not in these_tags:
            continue
        if CPU_ONLY not in these_tags and GPU not in these_tags:
            raise ValueError(
                f"Workflow {name} claims to be test-ready, but does not have a GPU or CPU_ONLY tag."
            )
        if CPU_ONLY in these_tags and GPU in these_tags:
            raise ValueError(
                f"Workflow {name} claims to be test-ready, but has both GPU and CPU_ONLY tags."
            )
        if endo_pipeline.NUM_GPUS is None and GPU in these_tags:
            continue
        yield app


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


TIMEOUT_MIN = 3


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
        return self.exception is not None

    @property
    def timed_out(self) -> bool:
        return not self.failed and self.elapsed_minutes > TIMEOUT_MIN

    @property
    def succeeded(self) -> bool:
        return not self.failed and not self.timed_out


def _run_workflow(app: "App") -> _WorkflowResult:
    error = None
    with _timer() as timer:
        try:
            # Call the workflow with no arguments
            app("")
        except Exception as e:
            error = e
    elapsed = timer.elapsed
    assert elapsed is not None
    return _WorkflowResult(name=app.name[0], exception=error, elapsed=elapsed)


def _summarize(results: list[_WorkflowResult]):
    """
    Print a summary of successful/failed workflows, with stacktraces
    """
    from termcolor import colored

    successes = [result for result in results if result.succeeded]
    timeouts = [result for result in results if result.timed_out]
    failures = [result for result in results if result.failed]
    success_count = ""
    timeout_count = ""
    failure_count = ""

    if len(successes) > 0:
        print(colored("====== Successes ======", "green"))
        success_count = f"{len(successes)} succeeded"
    for result in successes:
        print(colored(result.name, "green"), colored(result.elapsed_str, "yellow"))

    if len(timeouts) > 0:
        print(colored("====== Too slow ======", "yellow"))
        timeout_count = f"{len(timeouts)} slow workflows"
    for result in timeouts:
        print(colored(f"{result.name} took {result.elapsed_str}", "yellow"))

    if len(failures) > 0:
        print(colored("====== Failures ======", "red"))
        failure_count = f"{len(failures)} failed"
    for result in failures:
        print(colored(f"{result.name} {result.exception}", "red"))

    if len(failures) > 0:
        final_color = "red"
    elif len(timeouts) > 0:
        final_color = "yellow"
    else:
        final_color = "green"
    print(
        colored("\n======", final_color),
        ", ".join(
            [
                colored(success_count, "green"),
                colored(timeout_count, "yellow"),
                colored(failure_count, "red"),
            ]
        ),
        colored("======", final_color),
    )


def main():
    from endo_pipeline.__main__ import pipeline_app, tags

    results = [_run_workflow(workflow) for workflow in _testable_workflows(pipeline_app, tags)]

    _summarize(results)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
