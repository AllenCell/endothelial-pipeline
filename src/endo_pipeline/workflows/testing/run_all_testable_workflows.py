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

    from endo_pipeline.cli.runner import run_all_workflows_with_tag, summarize_workflow_run_results

    results = asyncio.run(
        run_all_workflows_with_tag(
            workflow_name="run-all-testable-workflows",
            select_tag="test-ready",
            force_demo_mode=True,
        )
    )
    summarize_workflow_run_results(results)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
