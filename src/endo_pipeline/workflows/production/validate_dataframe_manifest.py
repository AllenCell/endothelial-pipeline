def main(dataframe_manifest_name: str) -> None:
    """
    Validate a given dataframe manifest.

    #validation #manifests
    """

    import logging

    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import load_dataframe_manifest

    logger = logging.getLogger(__name__)

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    logger.info("Starting validation of dataframe manifest [ %s ]", dataframe_manifest_name)

    for dataset_name, location in dataframe_manifest.locations.items():
        if location.fmsid is None and location.s3uri is None and location.path is None:
            logger.error(
                "Dataset [ %s ] in dataframe manifest [ %s ] does not have a location supplied.",
                dataset_name,
                dataframe_manifest_name,
            )
            raise ValueError(
                f"Dataset {dataset_name} in dataframe manifest {dataframe_manifest_name} "
                "does not have a valid location."
            )
        # confirm we can load the dataframe
        _ = load_dataframe(location)

    logger.info("Finished validation of dataframe manifest [ %s ]", dataframe_manifest_name)
