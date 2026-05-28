from typing import Annotated

from cyclopts import Parameter


def main(
    manifest_name: str,
    add_files: Annotated[bool, Parameter(negative="--remove-files")] = True,
    dry_run: Annotated[bool, Parameter(negative="--live-run")] = True,
) -> None:
    """
    Stage files from selected image manifest to S3 bucket.

    #internal #vast

    This workflow stages images in the selected image manifest to the internal
    staging S3 bucket. It must be run internally on the HPC and you must have
    write permissions to the bucket via SSO.

    To configure SSO, run `aws configure sso` and follow instructions. Ask a
    team member for the start URL, if you do not already have it.

    To authenticate your AWS CLI session, run `aws sso login`. You will need to
    rerun this when your session expires. Make sure to note the profile name and
    run the following to ensure the workflow uses the correct profile:

    ```bash
    export AWS_PROFILE=<PROFILE NAME>
    ```

    To confirm your credentials are being picked up, try running the following
    to see the currently staged files:

    ```bash
    aws s3 ls s3://allencell-internal-quilt/endo_stg/
    ```

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe stage-image-manifest MANIFEST_NAME -vd
    ```

    To dry run the workflow:

    ```bash
    uv run endopipe stage-image-manifest MANIFEST_NAME --dry-run
    ```

    To run the workflow live:

    ```bash
    uv run endopipe stage-image-manifest MANIFEST_NAME --live-run
    ```

    You may need to add the following to your .bashrc if jobs are failing due to
    metadata service timeouts:

    ```bash
    export AWS_EC2_METADATA_DISABLED=true
    export AWS_CLI_DISABLE_CRT=true
    export AWS_REGION=us-west-2
    export AWS_DEFAULT_REGION=us-west-2
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only run for
    a single location and stage files to a top level `demo` subdirectory.

    Parameters
    ----------
    manifest_name
        Name of image manifest containing images to stage.
    add_files
        True to add files to the staging location, False to remove files.
    dry_run
        True to submit jobs with the `--dryrun` flag, False otherwise.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io.output import get_output_path, make_name_unique
    from endo_pipeline.library.process.manifest_staging import (
        draft_manifest_staging_job,
        generate_manifest_staging_dataframe,
        submit_manifest_staging_job,
    )
    from endo_pipeline.manifests import load_image_manifest
    from endo_pipeline.settings.manifest_staging import IMAGE_MANIFEST_STAGING_FOLDERS

    logger = logging.getLogger(__name__)

    unique_name = str(make_name_unique(f"{manifest_name}{'_demo' if DEMO_MODE else ''}"))
    output_path = get_output_path(__file__, unique_name)

    # Check if requested manifest is supported for staging to bucket
    if manifest_name not in IMAGE_MANIFEST_STAGING_FOLDERS.keys():
        logger.error("Manifest '%s' is not supported for staging to bucket", manifest_name)
        return

    # Load image manifest and set staging folder
    manifest = load_image_manifest(manifest_name)
    folder = IMAGE_MANIFEST_STAGING_FOLDERS[manifest_name]

    # Get list of all available location keys from the manifest
    location_keys = list(manifest.locations.keys())

    # Limit number of locations and use demo folder if running in demo mode.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Staging single location to demo directory")
        location_keys = location_keys[:1]
        folder = f"demo/{folder}"

    # Generate manifest staging dataframe and draft manifest staging job
    csv_path = generate_manifest_staging_dataframe(manifest, location_keys, folder, output_path)
    job_path = draft_manifest_staging_job(csv_path, output_path, dry_run, add_files)

    # Skip if no job path was returned
    if job_path is None:
        logger.error("No jobs were drafted. Exiting")
        return

    # Submit manifest staging job to cluster
    submit_manifest_staging_job(job_path, output_path, unique_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
