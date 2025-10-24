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


@dataclass
class _WorkflowResult:
    name: str
    error: Exception | None
    elapsed: "timedelta"


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
    return _WorkflowResult(name=app.name[0], error=error, elapsed=elapsed)


def main():
    from datetime import timedelta

    from endo_pipeline.__main__ import pipeline_app, tags

    results = [_run_workflow(workflow) for workflow in _testable_workflows(pipeline_app, tags)]

    # More verbose means lower number
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    TIMEOUT_MIN = 3
    for result in results:
        elapsed_minutes = result.elapsed / timedelta(minutes=1)
        elapsed_str = f"{elapsed_minutes:.1f}min"
        if result.error is not None:
            logger.error(f"Workflow {result.name} failed after {elapsed_str}: {result.error}")
        elif elapsed_minutes > TIMEOUT_MIN:
            logger.error(
                f"Workflow {result.name} took too long ({elapsed_str}): please remove its TEST_READY tag and file an issue."
            )
        else:
            logger.info(f"Workflow {result.name} succeeded in {elapsed_str}")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
