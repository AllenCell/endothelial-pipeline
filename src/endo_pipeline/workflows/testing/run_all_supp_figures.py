def main():
    """
    Run all supplemental figure workflows.

    To register a supplemental figure workflow, add a #supp-figure tag to the
    docstring.

    ## Example usage

    To run all supplemental figures:

    ```bash
    uv run endopipe run-all-supp-figures
    ```

    ## CPU vs. GPU

    Some figure panels should be run with an NVIDIA GPU. Run this workflow with
    the GPU flag (`-g` or `--num-gpus`) to make sure GPUs are visible to the
    workflows. While all figures are able to be run without a GPU, they will be
    noticeably slower.
    """

    import asyncio

    from endo_pipeline.cli.runner import run_all_workflows_with_tag, summarize_workflow_run_results
    from endo_pipeline.settings.timeouts import WORKFLOW_FIGURE_TIMEOUT

    results = asyncio.run(
        run_all_workflows_with_tag(
            workflow_name="run-all-supp-figures",
            select_tag="supp-figure",
            runner_timeout=WORKFLOW_FIGURE_TIMEOUT,
            force_demo_mode=False,
        )
    )
    summarize_workflow_run_results(results)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
