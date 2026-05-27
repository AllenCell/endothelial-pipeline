from pathlib import Path


def main(job_path: Path) -> None:
    """
    Verify manifest staging and update the corresponding manifest.

    #internal

    ## Example usage

    To run the workflow:

    ```bash
    uv run endopipe verify-manifest-staging JOB_PATH
    ```

    Parameters
    ----------
    job_path
        Path to the job file from a staging workflow.
    """

    from s3_uploader import check_completion

    from endo_pipeline.library.process.manifest_staging import update_staged_manifest_locations

    # Run check on completed jobs
    err_path = str(job_path.parent)
    check_completion(str(job_path), err_path)

    # Update manifests with S3 paths
    update_staged_manifest_locations(job_path)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
