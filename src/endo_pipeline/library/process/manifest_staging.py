"""Methods for staging manifests to S3 bucket."""

import logging
import re
from pathlib import Path

import pandas as pd
from s3_uploader import draft_rm_jobs, draft_sync_jobs, run_all_jobs
from termcolor import colored

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.manifests import (
    ImageManifest,
    get_image_location_for_dataset,
    load_image_manifest,
    save_image_manifest,
)
from endo_pipeline.settings.manifest_staging import (
    S3_STAGING_DIRECTORY,
    STAGING_SOURCE_COLUMN_NAME,
    STAGING_TARGET_COLUMN_NAME,
)

logger = logging.getLogger(__name__)


def build_image_manifest_staging_entries_for_dataset(
    manifest: ImageManifest, dataset_name: str, folder: str
) -> list[dict[str, str]]:
    """Build image manifest staging entries for given dataset."""

    dataset_config = load_dataset_config(dataset_name)
    positions = dataset_config.zarr_positions

    entries = []

    for position in positions:
        location = get_image_location_for_dataset(manifest, dataset_config, position=position)
        assert location.path is not None
        target = f"{S3_STAGING_DIRECTORY}{folder}{location.path.name}"
        entries.append(
            {
                STAGING_SOURCE_COLUMN_NAME: str(location.path),
                STAGING_TARGET_COLUMN_NAME: target,
            }
        )

    return entries


def generate_manifest_staging_dataframe(
    manifest: ImageManifest,
    dataset_names: list[str],
    folder: str,
    output_path: Path,
) -> Path:
    """Generate manifest staging dataframe for given datasets."""

    staging_entry_builders = {
        ImageManifest: build_image_manifest_staging_entries_for_dataset,
    }
    staging_entry_builder = staging_entry_builders[type(manifest)]

    entries = []

    for dataset_name in dataset_names:
        if dataset_name not in manifest.locations:
            logger.error("Dataset '%s' not found in manifest '%s'", dataset_name, manifest.name)

        logger.debug("Adding dataset '%s' to staging dataframe", dataset_name)
        entries.extend(staging_entry_builder(manifest, dataset_name, folder))

    df = pd.DataFrame(entries)
    csv_path = output_path / f"{manifest.name}.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def draft_manifest_staging_job(
    csv_path: Path, output_path: Path, dry_run: bool, add_files: bool
) -> Path | None:
    """Draft jobs for manifest staging."""

    if add_files:
        output_jobs = draft_sync_jobs(
            input_path=str(csv_path),
            src_column=STAGING_SOURCE_COLUMN_NAME,
            dest_column=STAGING_TARGET_COLUMN_NAME,
            output_dir=str(output_path),
            log_dir=str(output_path),
            dry_run=dry_run,
        )
    else:
        output_jobs = draft_rm_jobs(
            input_path=str(csv_path),
            target_column=STAGING_TARGET_COLUMN_NAME,
            output_dir=str(output_path),
            log_dir=str(output_path),
            dry_run=dry_run,
        )

    # Skip submitting jobs if any errors were encountered
    if isinstance(output_jobs, pd.DataFrame):
        output_df: pd.DataFrame = output_jobs
        logger.error("Error drafting jobs: %s", output_df[output_df["validation_error"].notnull()])
        return None

    return output_jobs.pop() if isinstance(output_jobs, set) else output_jobs


def submit_manifest_staging_job(job_path: Path, output_path: Path, job_name: str) -> None:
    """Submit manifest staging job to cluster."""

    # Submit job to cluster
    run_all_jobs(
        local=True,
        path=str(job_path),
        error_dir=str(output_path),
        slurm_args=f"--job-name={job_name} --partition=aics",
    )

    # Print message for how to check job status
    squeue_command = colored(f"squeue -p aics -n {job_name}", attrs=["bold"])
    print(f"\n\N{BULLET} Run {squeue_command} to check on job status")

    # Print message for how to verify job completion
    check_command = colored(f"endopipe verify-manifest-staging {job_path}", attrs=["bold"])
    print(f"\N{BULLET} Run {check_command} after jobs are finished\n")


def update_staged_manifest_locations(job_path: Path) -> None:
    """Update manifest locations after staging jobs have completed."""

    # Expect there to be a CSV containing the mapping between local paths and
    # S3 URIs, with the name of the file matching the name of the manifest
    csv_path = list(job_path.parent.glob("*csv"))[0]
    manifest_name = csv_path.stem
    df = pd.read_csv(csv_path)

    # Check if the staging job was adding or removing files.
    add_files = "local-to-s3" in job_path.name

    path_to_s3uri = {}

    # Iterate through the records in the CSV and build a mapping between local
    # paths and S3 URIs. Replace instance of P# with P{{position}} placeholder
    for entry in df.to_dict("records"):
        path = entry["local_path_staging"]
        path_with_placeholders = re.sub(r"_P[0-9]\.", "_P{{position}}.", path)

        s3uri = entry["s3_uri_staging"]
        s3uri_with_placeholders = re.sub(r"_P[0-9]\.", "_P{{position}}.", s3uri)

        path_to_s3uri[Path(path_with_placeholders)] = s3uri_with_placeholders

    # Iterate through locations and add or remove S3 URIs for matching paths
    manifest = load_image_manifest(manifest_name)

    for key, location in manifest.locations.items():
        if location.path is not None and location.path in path_to_s3uri:
            location.s3uri = path_to_s3uri[location.path] if add_files else None
            logger.info("Setting '%s' S3 URI location to '%s'", key, location.s3uri)

    save_image_manifest(manifest)
