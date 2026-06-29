def main():
    """
    Run all workflows tagged as test ready in demo mode.

    To register a workflow for testing:

    1. Make sure your workflow runs in under 5 minutes in `DEMO_MODE`
    2. Add #test-ready tag to the docstring
    3. For workflows that use GPU, also add a #gpu tag to the docstring

    ## Example usage

    To test all testable workflows:

    ```bash
    uv run endopipe run-all-testable-workflows -g 1
    ```

    To test only workflows that do not require a GPU:

    ```bash
    uv run endopipe run-all-testable-workflows
    ```

    ## CPU vs. GPU

    When this workflow is run without the GPU flag (`-g` or `--num-gpus`), it will
    not test any workflows that have been tagged with the `#gpu` tag. While these
    workflows are able to be run without a GPU, it is generally too slow to be
    practical for testing workflow functionality.
    """

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
