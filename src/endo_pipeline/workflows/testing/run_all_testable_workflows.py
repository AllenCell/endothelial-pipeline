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


def main():
    """Run all workflows marked as test ready."""

    import asyncio
    import logging

    import endo_pipeline.cli
    from endo_pipeline.cli.apps import pipeline_app
    from endo_pipeline.cli.runner import _run_all, _summarize

    logger = logging.getLogger(__name__)

    if not endo_pipeline.cli.DEMO_MODE:
        endo_pipeline.cli.DEMO_MODE = True
        logger.debug("Forcing demo mode on for testing.")

    results = asyncio.run(_run_all(pipeline_app))
    _summarize(results)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
