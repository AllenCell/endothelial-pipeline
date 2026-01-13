from pathlib import Path

import pandas as pd
from s3_uploader import draft_rm_jobs, draft_sync_jobs

from endo_pipeline.settings.data_release import DEST_COL, SOURCE_COL


def create_upload_job(
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
    return output_jobs


def create_rm_job(
    csv_path: str,
    save_dir: str,
    log_dir: str,
    target_col: str = DEST_COL,
    dry_run: bool = True,  # Set to False to do a real deletion
):
    output_jobs: Path | pd.DataFrame | None = draft_rm_jobs(
        input_path=csv_path,
        target_column=target_col,
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
    if isinstance(output_jobs, Path):
        print(f"Go read and validate this file! {output_jobs}")
    else:
        print("No jobs were created. Input empty")
    return output_jobs
