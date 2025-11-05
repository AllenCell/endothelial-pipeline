import logging

from endo_pipeline.cli import tags

logger = logging.getLogger(__name__)

TAGS = ["internal", tags.TEST_READY, tags.CPU_ONLY]


def main() -> None:
    """
    This workflow is used to keep files from expiring on VAST.
    """
    from collections import Counter

    try:
        from aicsfiles import fms
    except ModuleNotFoundError:
        logger.error("Required dependency [ aicsfiles ] not found")
        raise

    from endo_pipeline.configs import load_all_dataset_configs
    from endo_pipeline.manifests import (
        get_available_dataframe_manifests,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
    )

    # Get all FMSIDs from dataset configs and dataframe manifests
    fmsid_list = []

    dataset_configs = load_all_dataset_configs()
    for dataset_config in dataset_configs:
        fmsid = dataset_config.fmsid
        fmsid_list.append(fmsid)

    df_manifest_names = get_available_dataframe_manifests()
    for df_manifest_name in df_manifest_names:
        df_manifest = load_dataframe_manifest(df_manifest_name)
        datasets = list_datasets_with_dataframes(df_manifest)
        for dataset_name in datasets:
            fmsid = df_manifest.locations[dataset_name].fmsid
            if fmsid is None:
                continue
            fmsid_list.append(fmsid)

    # Cache files to VAST
    cache_file_statuses = fms.cache_files(fmsid_list)

    # Log summary
    logger.info("'%s' datasets queued.", len(fmsid_list))

    status_values = []
    for fmsid, status in cache_file_statuses["cacheFileStatuses"].items():
        status_values.append(status)

    status_counts = Counter(status_values)
    for status, count in status_counts.items():
        logger.info(f"'{status}': {count}")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
