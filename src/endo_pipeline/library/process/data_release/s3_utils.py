from pathlib import Path

import pandas as pd
from s3_uploader import check_completion, draft_rm_jobs, draft_sync_jobs, run_all_jobs

from endo_pipeline.settings.data_release import DEST_COL, SOURCE_COL


def step1_upload_job(
    csv_path: str,
    save_dir: str,
    log_dir: str,
    source_col: str = SOURCE_COL,
    dest_col: str = DEST_COL,
    dry_run: bool = True,  # Set to False to do a real upload
):
    output_jobs: set[Path] | pd.DataFrame = draft_sync_jobs(
        input_path=csv_path,
        src_column=source_col,
        dest_column=dest_col,
        output_dir=save_dir,
        log_dir=log_dir,
        dry_run=dry_run,
    )
    if isinstance(output_jobs, pd.DataFrame):
        output_df: pd.DataFrame = output_jobs
        print(
            "There were some errors!",
            output_df[output_df["validation_error"].notnull()],
        )
    else:
        for path in output_jobs:
            print(f"Go read and validate this file! {path}")
            print(f"Then run it with step2('{path}')")
    return output_jobs


def step2_run_jobs(jobs_path: str):
    run_all_jobs(
        local=True,  # Do uploads from local filesystem to S3
        path=jobs_path,
        error_dir="./2025-10-08-to-upload-errors",
    )
    print("Wait for the jobs to finish: run `squeue` to check.")
    print(f"Verify success with step3('{jobs_path}')")


def step3_check_job(jobs_path: str):
    check_completion(
        jobs_path,
        "./2025-10-08-to-upload-errors",  # Same as step2's error_dir
    )


def step1_draft_rm_job():
    draft_rm_jobs(
        input_path="2025-10-08-to-remove.csv",
        target_column="s3_path",
        output_dir="./rm-jobs",
        log_dir="./2025-10-08-to-remove-logs",
        dry_run=True,  # Set to False to do a real deletion
    )
