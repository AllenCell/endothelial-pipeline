from endo_pipeline.cli import Datasets
from endo_pipeline.io.output import get_output_path


def main(
    datasets: Datasets,
    add_datasets: bool = True,
    dry_run: bool = True,
) -> None:
    """Upload or remove datasets to/from S3."""
    from endo_pipeline.library.process.data_release.generate_csv import (
        create_s3_rm_zarr_csv,
        create_s3_upload_csv,
    )
    from endo_pipeline.library.process.data_release.s3_utils import step1_upload_job

    save_dir = get_output_path("s3_dataset")
    log_dir = get_output_path("s3_dataset", "status")

    if add_datasets:
        csv_path = create_s3_upload_csv(datasets, save_dir)
    else:
        csv_path = create_s3_rm_zarr_csv(datasets, save_dir)

    step1_upload_job(
        csv_path=csv_path,
        save_dir=save_dir,
        log_dir=log_dir,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
