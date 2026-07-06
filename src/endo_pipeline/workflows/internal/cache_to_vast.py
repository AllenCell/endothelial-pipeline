def main() -> None:
    """
    Cache FMS files to Vast.

    #internal #vast

    This script iterates through all dataset configs and dataframe/model
    manifests to identify locations with FMS file IDs. All file IDs are then
    submitted for caching to Vast. If file is already cached, then renew lease.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe cache-to-vast -d
    ```

    To run the full workflow:

    ```bash
    uv run endopipe cache-to-vast
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will skip actually
    submitting the cache request, and instead return "dummy" responses.
    """

    import logging
    from collections import Counter

    from endo_pipeline.configs import load_all_dataset_configs
    from endo_pipeline.io import cache_fms_files
    from endo_pipeline.manifests import (
        get_available_dataframe_manifests,
        get_available_model_manifests,
        load_dataframe_manifest,
        load_model_manifest,
    )

    logger = logging.getLogger(__name__)

    fmsid_list = []

    # Get all FMS file IDs from dataset configs
    for dataset_config in load_all_dataset_configs():
        dataset_fmsid = dataset_config.fmsid
        fmsid_list.append(dataset_fmsid)

    # Get all FMS file IDs from dataframe manifests
    for dataframe_manifest_name in get_available_dataframe_manifests():
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
        for location in dataframe_manifest.locations.values():
            if location.fmsid is None:
                continue
            fmsid_list.append(location.fmsid)

    # Get all FMS file IDs from model manifests
    for model_manifest_name in get_available_model_manifests():
        model_manifest = load_model_manifest(model_manifest_name)
        for location in model_manifest.locations.values():
            if location.fmsid is None:
                continue
            if isinstance(location.fmsid, str):
                fmsid_list.append(location.fmsid)
            else:
                fmsid_list.extend(location.fmsid)

    # Cache files to VAST
    cache_file_statuses = cache_fms_files(fmsid_list)

    # Log summary
    logger.info("Queued '%d' FMS files to be cached.", len(fmsid_list))

    status_values = []
    for status in cache_file_statuses["cacheFileStatuses"].values():
        status_values.append(status)

    status_counts = Counter(status_values)
    for status, count in status_counts.items():
        print(f"Cache Status '{status}' = {count}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
