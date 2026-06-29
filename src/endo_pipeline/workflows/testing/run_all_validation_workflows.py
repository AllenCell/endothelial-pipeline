def main():
    """
    Run all workflows tagged as validation in non-demo mode.

    To register a workflow for validation:

    1. Add #validation tag to the docstring
    2. For workflows that use GPU, also add a #gpu tag to the docstring
    3. Make sure validations log an error rather than raising exceptions

    ## Example usage

    To run all validation workflows:

    ```bash
    uv run endopipe run-all-validation-workflows -g 1
    ```

    To run only validation workflows that do not require a GPU:

    ```bash
    uv run endopipe run-all-validation-workflows
    ```

    ## CPU vs. GPU

    When this workflow is run without the GPU flag (`-g` or `--num-gpus`), it
    will not test any workflows that have been tagged with the `#gpu` tag. While
    these workflows are able to be run without a GPU, it is generally too slow
    to be practical for testing workflow functionality.
    """

    import asyncio

    from endo_pipeline.cli.runner import run_all_workflows_with_tag, summarize_workflow_run_results
    from endo_pipeline.settings.timeouts import WORKFLOW_VALIDATION_TIMEOUT

    results = asyncio.run(
        run_all_workflows_with_tag(
            workflow_name="run-all-validation-workflows",
            select_tag="validation",
            runner_timeout=WORKFLOW_VALIDATION_TIMEOUT,
            force_demo_mode=False,
        )
    )
    summarize_workflow_run_results(results)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
