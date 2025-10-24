from pathlib import Path

import pandas as pd
from s3_uploader import check_completion, draft_rm_jobs, draft_sync_jobs, run_all_jobs


def step1(df: pd.DataFrame):
    df.to_csv("2025-10-08-to-upload.csv", index=False)
    assert "local_path" in df.columns
    assert "s3_path" in df.columns
    output_jobs: set[Path] | pd.DataFrame = draft_sync_jobs(
        input_path="2025-10-08-to-upload.csv",
        src_column="local_path",
        dest_column="s3_path",
        output_dir="./jobs",
        log_dir="./2025-10-08-to-upload-logs",
        dry_run=True,  # Set to False to do a real upload
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


def step2(jobs_path: str):
    run_all_jobs(
        local=True,  # Do uploads from local filesystem to S3
        path=jobs_path,
        error_dir="./2025-10-08-to-upload-errors",
    )
    print("Wait for the jobs to finish: run `squeue` to check.")
    print(f"Verify success with step3('{jobs_path}')")


def step3(jobs_path: str):
    check_completion(
        jobs_path,
        "./2025-10-08-to-upload-errors",  # Same as step2's error_dir
    )


def step4():
    draft_rm_jobs(
        input_path="2025-10-08-to-remove.csv",
        target_column="s3_path",
        output_dir="./rm-jobs",
        log_dir="./2025-10-08-to-remove-logs",
        dry_run=True,  # Set to False to do a real deletion
    )
