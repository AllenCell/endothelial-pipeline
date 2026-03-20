from endo_pipeline.cli import CropPattern, Datasets


def main(
    datasets: Datasets | None = None,
    crop_pattern: CropPattern = "grid",
    upload_to_fms: bool = False,
) -> None:
    """
    Calculate PCA features from DiffAE latent features.

    #diffae-features #dataframe-production #pca

    This workflow calculates PCA using features from the default Diff AE model
    from a default collection of datasets (see documentation for `fit_pca`).

    For each of the specified datasets, this workflow:

    - Loads the DiffAE feature dataframe as output by `eval-diffae`
    - Adds a unique `crop_index` identifier for downstream workflows
    - Projects the latent DiffAE features onto the principal components
        of the fit PCA model (see above)
    - Calculates additional features as transforms of the PC-projected
        features (e.g., polar coordinates, see `project_features_to_pcs`)
    - Performs additional timepoint filtering, if using grid-based crops

    **Dataframe output and tracking**

    By default, the final dataframe for each dataset is saved out locally with a
    unique file path. If this workflow is run with `--upload-to-fms`, these
    dataframes are uploaded to FMS instead. In both cases, the dataframe
    manifest is updated with the corresponding location.

    Parameters
    ----------
    datasets
        Dataset(s) or dataset collections(s) to process.
    crop_pattern
        Crop pattern used to generate the feature dataframe.
    upload_to_fms
        If true, upload dataframe(s) to FMS and track FMS ID via dataframe
        manifest. Else, save dataframe(s) locally.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        get_subset_of_timepoint_annotations,
        load_dataset_config,
    )
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        load_dataframe,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        add_crop_index,
        filter_dataframe_by_annotations,
        fit_pca,
        project_features_to_pcs,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_FEATURE_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    dataset_names = datasets or get_datasets_in_collection("timelapse")

    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)
    feature_dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern
    )
    feature_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    output_path = get_output_path(__file__)

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    # Get fit PCA object using grid-based crops
    pca = fit_pca()

    for dataset_name in dataset_names:
        logger.info("Calculating PCA features for dataset '%s'", dataset_name)

        dataset_config = load_dataset_config(dataset_name)
        location = get_dataframe_location_for_dataset(feature_manifest, dataset_name)
        df = load_dataframe(location)

        # Add unique crop indices for downstream workflows
        df_with_crop_index = add_crop_index(df, crop_pattern)

        # Project feature data onto PC axes and compute additional transformed
        # features (e.g. polar coordinates) from the PC-projected features
        df_with_pcs = project_features_to_pcs(
            df_with_crop_index,
            pca,
            feat_cols=DIFFAE_FEATURE_COLUMN_NAMES,
            compute_polar=True,
            rescale_theta=True,
            flip_pc3_sign=True,
        )

        # Drop original feature columns to save memory
        full_pca_df = df_with_pcs.drop(columns=DIFFAE_FEATURE_COLUMN_NAMES)

        full_pca_df_path = output_path / f"{dataset_name}_{crop_pattern}_pca.parquet"
        full_pca_manifest_name = f"{feature_dataframe_manifest_name}_pca"
        full_pca_manifest = create_dataframe_manifest(full_pca_manifest_name, __name__)

        full_pca_df.to_parquet(full_pca_df_path, index=False)

        if upload_to_fms:
            fms_annotations = build_fms_annotations(dataset_config)
            fmsid = upload_file_to_fms(
                output_path, annotations=fms_annotations, file_type="parquet"
            )
            full_pca_location = DataframeLocation(fmsid=fmsid)
        else:
            full_pca_location = DataframeLocation(path=full_pca_df_path)

        full_pca_manifest.locations[dataset_name] = full_pca_location
        save_dataframe_manifest(full_pca_manifest)

        # For grid-based crops, filter out annotated timepoints, except for
        # timepoints flagged as "not steady state" (those can be filtered out
        # dynamically as necessary in downstream workflows)
        if crop_pattern == "grid":
            logger.info("Filtering grid-based PCA features for dataset '%s'", dataset_name)

            timepoint_annotations = get_subset_of_timepoint_annotations(
                annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
            )
            filtered_pca_df = filter_dataframe_by_annotations(
                full_pca_df,
                dataset_config,
                timepoint_annotations=timepoint_annotations,
            )

            filtered_pca_df_path = output_path / f"{dataset_name}_{crop_pattern}_pca.parquet"
            filtered_pca_manifest_name = f"{feature_dataframe_manifest_name}_pca_filtered"
            filtered_pca_manifest = create_dataframe_manifest(filtered_pca_manifest_name, __name__)

            filtered_pca_df.to_parquet(filtered_pca_df_path, index=False)

            if upload_to_fms:
                fms_annotations = build_fms_annotations(dataset_config)
                fmsid = upload_file_to_fms(
                    filtered_pca_df_path, annotations=fms_annotations, file_type="parquet"
                )
                filtered_pca_location = DataframeLocation(fmsid=fmsid)
            else:
                filtered_pca_location = DataframeLocation(path=filtered_pca_df_path)

            filtered_pca_manifest.locations[dataset_name] = filtered_pca_location
            save_dataframe_manifest(filtered_pca_manifest)

        # For track-based crops, filtering happens downstream, where the output
        # dataframe of this workflow PC-transformed features is merged with the
        # segmentation + tracking dataframes


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
