import logging

from endo_pipeline.cli import tags

logger = logging.getLogger(__name__)

TAGS = ["internal", tags.TEST_READY, tags.CPU_ONLY]


def main() -> None:
    """
    Ensures files do not expire on VAST by managing their cache status.
    """
    from collections import Counter

    from endo_pipeline.configs import load_all_dataset_configs
    from endo_pipeline.io import cache_fms_files
    from endo_pipeline.manifests import (
        get_available_dataframe_manifests,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
    )

    # Get all FMSIDs from dataset configs and dataframe manifests
    fmsid_list = []

    dataset_configs = load_all_dataset_configs()
    for dataset_config in dataset_configs:
        dataset_fmsid = dataset_config.fmsid
        fmsid_list.append(dataset_fmsid)

    df_manifest_names = get_available_dataframe_manifests()
    for df_manifest_name in df_manifest_names:
        df_manifest = load_dataframe_manifest(df_manifest_name)
        datasets = list_datasets_with_dataframes(df_manifest)
        for dataset_name in datasets:
            df_fmsid = df_manifest.locations[dataset_name].fmsid
            if df_fmsid is None:
                continue
            fmsid_list.append(df_fmsid)

    # Cache files to VAST
    cache_file_statuses = cache_fms_files(fmsid_list)

    # Log summary
    logger.info("'%s' FMSIDs queued.", len(fmsid_list))

    status_values = []
    for status in cache_file_statuses["cacheFileStatuses"].values():
        status_values.append(status)

    status_counts = Counter(status_values)
    for status, count in status_counts.items():
        logger.info(f"'{status}': {count}")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
