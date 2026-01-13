from endo_pipeline.cli import Datasets
from endo_pipeline.io.output import get_output_path


def main(
    datasets: Datasets | None = None,
    add_datasets: bool = True,
    dry_run: bool = True,
) -> None:
    """
    Upload or remove datasets to/from S3.

    REQUIRED:
    - Workflow must be run on slurm-master cluster.
    - AWS credentials must be configured.

    Parameters
    ----------
    datasets: Datasets | None
        List of dataset names to upload/remove. If None, all datasets in the
        "dataset_release" collection will be used.
    add_datasets: bool
        If True, datasets will be uploaded to S3. If False, datasets will be removed from S3.
    dry_run: bool
        If True, the upload/remove jobs will be drafted but not executed.
    """
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.library.process.data_release.generate_csv import (
        create_s3_rm_zarr_csv,
        create_s3_upload_csv,
    )
    from endo_pipeline.library.process.data_release.s3_utils import step1_upload_job, step2_run_jobs

    if datasets is None:
        datasets = get_datasets_in_collection("dataset_release")

    save_dir = get_output_path("s3_dataset")
    log_dir = get_output_path("s3_dataset", "status")

    if add_datasets:
        csv_path = create_s3_upload_csv(datasets, save_dir)

        step1_upload_job(
            csv_path=csv_path,
            save_dir=save_dir,
            log_dir=log_dir,
            dry_run=dry_run,
        )

        step2_run_jobs(save_dir, log_dir)

    else:
        csv_path = create_s3_rm_zarr_csv(datasets, save_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
