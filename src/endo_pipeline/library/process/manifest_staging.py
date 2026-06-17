"""Methods for staging manifests to S3 bucket."""

import logging
from pathlib import Path

import pandas as pd
import yaml
from s3_uploader import draft_rm_jobs, draft_sync_jobs, run_all_jobs
from termcolor import colored

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import resolve_dataframe_location, resolve_model_location
from endo_pipeline.manifests import (
    DataframeManifest,
    ImageManifest,
    ModelManifest,
    get_available_dataframe_manifests,
    get_available_image_manifests,
    get_available_model_manifests,
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    load_dataframe_manifest,
    load_image_manifest,
    load_model_manifest,
    save_dataframe_manifest,
    save_image_manifest,
    save_model_manifest,
)
from endo_pipeline.settings.manifest_staging import (
    S3_STAGING_DIRECTORY,
    STAGING_SOURCE_COLUMN_NAME,
    STAGING_TARGET_COLUMN_NAME,
)

logger = logging.getLogger(__name__)


def build_image_manifest_staging_entries_for_location(
    manifest: ImageManifest, dataset_name: str, folder: str
) -> tuple[list[dict[str, str]], str]:
    """Build image manifest staging entries for given dataset."""

    dataset_config = load_dataset_config(dataset_name)
    positions = dataset_config.zarr_positions

    entries = []

    # Iterate over positions and add one entry per position
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

    # Use the original location with placeholders to set the manifest update
    placeholder_location = manifest.locations[dataset_name]
    assert placeholder_location.path is not None
    update = f"{S3_STAGING_DIRECTORY}{folder}{placeholder_location.path.name}"

    return entries, update


def build_dataframe_manifest_staging_entries_for_location(
    manifest: DataframeManifest, location_key: str, folder: str
) -> tuple[list[dict[str, str]], str]:
    """Build dataframe manifest staging entries for given location key."""

    location = get_dataframe_location_for_dataset(manifest, location_key)
    location.s3uri = None
    source = resolve_dataframe_location(location)
    target = f"{S3_STAGING_DIRECTORY}{folder}{Path(source).name}"

    entries = [
        {
            STAGING_SOURCE_COLUMN_NAME: source,
            STAGING_TARGET_COLUMN_NAME: target,
        }
    ]

    return entries, target


def build_model_manifest_staging_entries_for_dataset(
    manifest: ModelManifest, location_key: str, folder: str
) -> tuple[list[dict[str, str]], str | tuple[str, str] | None]:
    """Build model manifest staging entries for given location key."""

    location = manifest.locations[location_key]
    location.s3uri = None
    source = resolve_model_location(location)

    entries = []
    update = None

    if isinstance(source, str):
        target = f"{S3_STAGING_DIRECTORY}{folder}{location_key}/{Path(source).name}"
        update = target
        entries.append(
            {
                STAGING_SOURCE_COLUMN_NAME: source,
                STAGING_TARGET_COLUMN_NAME: target,
            }
        )
    elif isinstance(source, tuple):
        source_ckpt, source_cfg = source
        target_ckpt = f"{S3_STAGING_DIRECTORY}{folder}{location_key}/{Path(source_ckpt).name}"
        target_cfg = f"{S3_STAGING_DIRECTORY}{folder}{location_key}/{Path(source_cfg).name}"
        update = (target_ckpt, target_cfg)
        entries.append(
            {
                STAGING_SOURCE_COLUMN_NAME: source_ckpt,
                STAGING_TARGET_COLUMN_NAME: target_ckpt,
            }
        )
        entries.append(
            {
                STAGING_SOURCE_COLUMN_NAME: source_cfg,
                STAGING_TARGET_COLUMN_NAME: target_cfg,
            }
        )

    return entries, update


def generate_manifest_staging_dataframe(
    manifest: ImageManifest | DataframeManifest | ModelManifest,
    location_keys: list[str],
    folder: str,
    output_path: Path,
) -> Path:
    """Generate manifest staging dataframe for given datasets."""

    staging_entry_builders = {
        ImageManifest: build_image_manifest_staging_entries_for_location,
        DataframeManifest: build_dataframe_manifest_staging_entries_for_location,
        ModelManifest: build_model_manifest_staging_entries_for_dataset,
    }
    staging_entry_builder = staging_entry_builders[type(manifest)]

    all_entries = []
    all_locations = {}

    for location_key in location_keys:
        if location_key not in manifest.locations:
            logger.error("Key '%s' not found in manifest '%s'", location_key, manifest.name)

        logger.debug("Adding '%s' to staging dataframe", location_key)
        entries, location = staging_entry_builder(manifest, location_key, folder)  # type: ignore[operator]
        all_entries.extend(entries)
        all_locations[location_key] = location

    # Combine entries into dataframe and save to CSV
    entries_df = pd.DataFrame(all_entries).drop_duplicates()
    csv_path = output_path / f"{manifest.name}.csv"
    entries_df.to_csv(csv_path, index=False)

    # Save list of update locations to YAML
    yaml_path = output_path / f"{manifest.name}.yaml"
    yaml_content = yaml.safe_dump(all_locations, encoding="utf-8")
    yaml_path.write_bytes(yaml_content)

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

    # Expect there to be a YAML containing the mapping between location key and
    # S3 URIs, with the name of the file matching the name of the manifest
    yaml_path = list(job_path.parent.glob("*yaml"))[0]
    manifest_name = yaml_path.stem
    updates = yaml.safe_load(yaml_path.open())

    # Check if the staging job was adding or removing files.
    add_files = "local-to-s3" in job_path.name

    # Try to identify the manifest type and update with corresponding method
    if manifest_name in get_available_image_manifests():
        manifest_loader = load_image_manifest
        manifest_saver = save_image_manifest
    elif manifest_name in get_available_dataframe_manifests():
        manifest_loader = load_dataframe_manifest
        manifest_saver = save_dataframe_manifest
    elif manifest_name in get_available_model_manifests():
        manifest_loader = load_model_manifest
        manifest_saver = save_model_manifest
    else:
        raise ValueError("Unable to update manifest '%s'", manifest_name)

    manifest = manifest_loader(manifest_name)

    for location_key, s3uri in updates.items():
        manifest.locations[location_key].s3uri = s3uri if add_files else None

    manifest_saver(manifest)
