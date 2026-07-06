def main():
    """
    Create a BFF-compatible CSV for releasing manifests with metadata.
    """

    import logging

    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.release_metadata import (
        build_dataframe_manifest_release_metadata,
        build_image_manifest_release_metadata,
        build_model_manifest_release_metadata,
    )
    from endo_pipeline.manifests import (
        load_dataframe_manifest,
        load_image_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.manifest_staging import (
        STAGING_DATAFRAME_MANIFEST_NAMES,
        STAGING_IMAGE_MANIFEST_NAMES,
        STAGING_MODEL_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    metadata = []

    # Get list of datasets to include in release
    datasets = get_datasets_in_collection("dataset_release")

    # Iterate through image manifests and add to release
    for manifest_name in STAGING_IMAGE_MANIFEST_NAMES:
        logger.info("Building release metadata for image manifest '%s'", manifest_name)
        manifest = load_image_manifest(manifest_name)

        for location_key, location in manifest.locations.items():
            # Skip if location does not have an S3 URI
            if location.s3uri is None:
                continue

            # Skip if dataset is not in list of datasets to release
            if location_key not in datasets:
                continue

            dataset_config = load_dataset_config(location_key)

            for position in dataset_config.zarr_positions:
                metadata.append(
                    build_image_manifest_release_metadata(manifest, dataset_config, position)
                )

    # Iterate through dataframe manifests and add to release
    for manifest_name in STAGING_DATAFRAME_MANIFEST_NAMES:
        logger.info("Building release metadata for dataframe manifest '%s'", manifest_name)
        manifest = load_dataframe_manifest(manifest_name)

        for location_key, location in manifest.locations.items():
            # Skip if location does not have an S3 URI
            if location.s3uri is None:
                continue

            # Skip if dataset is not in list of datasets to release
            if location_key not in datasets and "training" not in manifest_name:
                continue

            metadata.append(build_dataframe_manifest_release_metadata(manifest, location_key))

    # Iterate through model manifests and add to release
    for manifest_name in STAGING_MODEL_MANIFEST_NAMES:
        logger.info("Building release metadata for model manifest '%s'", manifest_name)
        manifest = load_model_manifest(manifest_name)

        for location_key, location in manifest.locations.items():
            # Skip if location does not have an S3 URI
            if location.s3uri is None:
                continue

            metadata.extend(build_model_manifest_release_metadata(manifest, location_key))

    metadata_df = pd.DataFrame(metadata)
    output_file = output_path / "release_metadata.csv"
    metadata_df.to_csv(output_file, index=False)
    print(f"\nRelease metadata CSV saved to: {output_file}\n")
