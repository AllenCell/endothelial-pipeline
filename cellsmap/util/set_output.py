import os

from deprecated import deprecated


@deprecated(
    """
This method deprecated and will be removed.

1. To get a path for outputs, use the following pattern:

    from src.endo_pipeline.io import get_output_path

    output_path = get_output_path(__file__)

For a workflow named "example_workflow", this method will return:

    Path("/path/to/results/YYYY-MM-DD/example_workflow")

2. If you wish to turn off the timestamp (not recommended), use:

    output_path = get_output_path(__file__, include_timestamp=False)

3. You may provide additional subfolder to the method:

    output_path = get_output_path(__file__, subfolder1, subfolder2)

These subfolders will be added to the returned output Path:

    Path("/path/to/results/YYYY-MM-DD/example_workflow/subfolder1/subfolder2")
"""
)
def get_output_path(workflow_name: str, verbose: bool = True) -> str:
    """
    Save results to a universal results directory in a folder titled after the workflow.
    The contents are gitignored.
    """
    repo_top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    output_dir = os.path.join(repo_top_dir, "results", workflow_name) + os.sep

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print(f"Output saved to directory: {output_dir}")

    return output_dir
