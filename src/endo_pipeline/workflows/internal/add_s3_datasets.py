from endo_pipeline.cli import Datasets, UniqueIntList


def main(
    datasets: Datasets | None = None,
    dry_run: bool = True,
    positions_list: UniqueIntList | None = None,
    raw_zarr: bool = False,
    segmentation_zarr: bool = False,
) -> None:
    """
    Upload datasets to S3.

    REQUIRED:
    - Workflow must be run on slurm-master cluster.
    - AWS credentials must be configured.

    export AWS_PROFILE=allencell_internal_quilt
    or add AWS_PROFILE=allencell_internal_quilt to each command below.

    To see what files are already there:
       aws s3 ls s3://allencell-internal-quilt/endo_stg/

    Login (if session expired):
        aws sso login

    Add to your .bashrc if uploads/removals are failing due to metadata service timeouts:
        export AWS_EC2_METADATA_DISABLED=true
        export AWS_CLI_DISABLE_CRT=true
        export AWS_REGION=us-west-2
        export AWS_DEFAULT_REGION=us-west-2

    Parameters
    ----------
    datasets: Datasets | None
        List of dataset names. If None, all datasets in the
        "add_s3_datasets" collection will be used.
    dry_run: bool
        If True, jobs will be drafted but not executed.
    positions_list: UniqueIntList | None
        List of position indices to upload. If None, all positions will be uploaded.
    raw_zarr: bool
        Whether to include raw image zarrs in the upload.
    segmentation_zarr: bool
        Whether to include segmentation zarrs in the upload. If True, both nuclear and
        VE-cadherin segmentation zarrs will be included.

    Example add job:
    endopipe add-s3-datasets --positions-list 0 1 2 3 4 5 --segmentation-zarr
    endopipe add-s3-datasets --no-dry-run --positions-list 0 1 2 3 4 5 --segmentation-zarr
    """
    import logging

    from s3_uploader import run_all_jobs

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.process.data_release.generate_csv import create_s3_upload_csv
    from endo_pipeline.library.process.data_release.s3_utils import create_upload_job

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("add_s3_datasets")

    save_dir = get_output_path("s3_dataset", "add_datasets")
    log_dir_str = str(get_output_path("s3_dataset", "add_datasets", "status"))
    save_dir_str = str(save_dir)

    csv_path = create_s3_upload_csv(
        datasets,
        save_dir,
        positions_list=positions_list,
        raw_zarr=raw_zarr,
        segmentation_zarr=segmentation_zarr,
    )

    jobs_paths = create_upload_job(
        csv_path=csv_path,
        save_dir=save_dir_str,
        log_dir=log_dir_str,
        dry_run=dry_run,
    )

    for job_path in jobs_paths:
        run_all_jobs(
            local=True,
            path=job_path,
            error_dir=log_dir_str,
            slurm_args="",  # update to "--partition=aics" if on hpc cluster.
        )

    if dry_run:
        logger.warning("[DRY RUN] Check files and re-run with --no-dry-run to submit jobs.")

    else:
        # Generate check completion script
        job_code = job_path.name.split("_")[0]
        script_path = save_dir / f"{job_code}_run_check_completion.py"

        jobs_paths_list = [str(p) for p in jobs_paths]

        with open(script_path, "w") as f:
            f.write(
                f"""\
from s3_uploader import check_completion

log_dir_str = {log_dir_str!r}
jobs_paths = {jobs_paths_list!r}

for job_path in jobs_paths:
    check_completion(job_path, log_dir_str)
"""
            )

        print("Wait for the jobs to finish: run `squeue` to check.")
        print("Verify success upon completion by running:")
        print(f"python {script_path}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
