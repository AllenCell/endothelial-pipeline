from s3_uploader import run_all_jobs

from endo_pipeline.cli import Datasets, UniqueIntList
from endo_pipeline.io.output import get_output_path


def main(
    datasets: Datasets | None = None,
    add_datasets: bool = False,
    rm_datasets: bool = False,
    dry_run: bool = True,
    positions_list: UniqueIntList | None = None,
) -> None:
    """
    Upload or remove datasets to/from S3. Must select either add_datasets or rm_datasets.

    REQUIRED:
    - Workflow must be run on slurm-master cluster.
    - AWS credentials must be configured.
    - Every day you need to login using: aws sso login

    To see what files are already there:
    aws s3 ls s3://allencell-internal-quilt/endo_stg/

    Parameters
    ----------
    datasets: Datasets | None
        List of dataset names to upload/remove. If None, all datasets in the
        "dataset_release" collection will be used.
    add_datasets: bool
        If True, datasets will be uploaded to S3.
    rm_datasets: bool
        If True, datasets will be removed from S3.
    dry_run: bool
        If True, the upload/remove jobs will be drafted but not executed.

    Example rm job:
    endopipe datasets-to-s3 --datasets 20241120_20X --rm-datasets  --positions-list 0 1
    endopipe datasets-to-s3 --datasets 20241120_20X --rm-datasets --no-dry-run  --positions-list 0 1

    Example add job:
    endopipe datasets-to-s3 --add-datasets --positions-list 0 1
    endopipe datasets-to-s3 --add-datasets --no-dry-run --positions-list 0 1
    """
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.library.process.data_release.generate_csv import (
        create_s3_rm_csv,
        create_s3_upload_csv,
    )
    from endo_pipeline.library.process.data_release.s3_utils import create_rm_job, create_upload_job

    if datasets is None:
        datasets = get_datasets_in_collection("dataset_release")

    if add_datasets:
        save_dir = get_output_path("s3_dataset", "add_datasets")
        log_dir_str = str(get_output_path("s3_dataset", "add_datasets", "status"))
        save_dir_str = str(save_dir)

        csv_path = create_s3_upload_csv(datasets, save_dir, positions_list=positions_list)

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
            )

    if rm_datasets:
        save_dir = get_output_path("s3_dataset", "remove_datasets")
        log_dir_str = str(get_output_path("s3_dataset", "remove_datasets", "status"))
        save_dir_str = str(save_dir)

        csv_path = create_s3_rm_csv(datasets, save_dir, positions_list=positions_list)

        job_path = create_rm_job(
            csv_path=csv_path,
            save_dir=save_dir_str,
            log_dir=log_dir_str,
            dry_run=dry_run,
        )

        run_all_jobs(
            s3=True,
            path=job_path,
            error_dir=log_dir_str,
        )

    if dry_run:
        print("Check files and re-run with --no-dry-run to submit jobs.")
    else:
        print("Wait for the jobs to finish: run `squeue` to check.")
        print(f"Verify success with check_completion on '{save_dir_str}'")

    # to run after everything is complete.
    # check_completion(
    #     save_dir_str,
    #     log_dir_str,
    # )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
