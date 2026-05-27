from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
) -> None:
    """
    Calculate PCA features from DiffAE latent features.

    #diffae #pca #cell-centered #grid-based

    This workflow calculates PCA using features from the default Diff AE model
    from a default collection of datasets.

    For each of the specified datasets, this workflow will:

    - Load the DiffAE feature dataframe as output by `eval-diffae`
    - Add a unique `crop_index` identifier for downstream workflows
    - Project the latent DiffAE features onto the principal components of the
      fit PCA model (see `fit_pca`)
    - Calculate additional features such as transforms of the PC-projected
      features (e.g., polar coordinates, see `project_features_to_pcs`)
    - Perform additional timepoint-based and position-based filtering
    - For cell-centered crops, perform filtering based on the "is_included"
      column, which removes tracks that do not pass segmentation QC

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe calculate-pca-features CROP_PATTERN -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe calculate-pca-features CROP_PATTERN --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `timelapse` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only
    calculate PCA features for the first dataset.

    Parameters
    ----------
    crop_pattern
        Crop pattern used for model evaluation.
    datasets
        List of datasets or dataset collections.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
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
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.pca import fit_pca, project_features_to_pcs
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        get_dataframe_location_for_dataset,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_FEATURE_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(
        "shear_stress"
    ) + get_datasets_in_collection("perturbation")

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    # Define output manifest name and list of required columns for selected
    # crop pattern. Note the modified name for the cell centered features; the
    # "full" set of features requires merging the outputs of the segmentation
    # workflows, which is handled by `combine_cell_centered_features`.
    if crop_pattern == "tracked":
        base_pca_manifest_name = "diffae_pca_features_tracked"
        required_columns = [Column.POSITION, Column.TRACK_ID]
    elif crop_pattern == "grid":
        base_pca_manifest_name = "grid_based_features"
        required_columns = [Column.POSITION, Column.DiffAEData.START_X, Column.DiffAEData.START_Y]
    else:
        raise ValueError("Crop pattern '%s' is not supported", crop_pattern)

    # Create manifest for unfiltered dataframes and add workflow parameters
    unfiltered_pca_manifest_name = f"{base_pca_manifest_name}_unfiltered"
    unfiltered_pca_manifest = create_dataframe_manifest(unfiltered_pca_manifest_name, __file__)
    unfiltered_pca_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "crop_pattern": crop_pattern,
        "filtered": False,
    }

    # Create manifest for filtered dataframes and add workflow parameters
    filtered_pca_manifest_name = f"{base_pca_manifest_name}_filtered"
    filtered_pca_manifest = create_dataframe_manifest(filtered_pca_manifest_name, __file__)
    filtered_pca_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "crop_pattern": crop_pattern,
        "filtered": True,
    }

    # Load dataframe manifest containing raw latent features
    feature_dataframe_manifest_name = (
        f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    )
    feature_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Get fit PCA object using grid-based crops
    pca = fit_pca()

    for dataset_name in dataset_names:
        if dataset_name not in feature_manifest.locations:
            logger.warning(
                "No location found in dataframe manifest '%s' for dataset '%s'. "
                "Skipping PCA feature calculation for this dataset.",
                feature_dataframe_manifest_name,
                dataset_name,
            )
            continue

        logger.info("Calculating PCA features for dataset '%s'", dataset_name)

        # Load dataframe containing raw latent features
        dataset_config = load_dataset_config(dataset_name)
        location = get_dataframe_location_for_dataset(feature_manifest, dataset_name)
        df = load_dataframe(location)

        # Group by the required columns and assign a unique integer (the crop
        # index) to each group based on the index of that group
        check_required_columns_in_dataframe(df, required_columns)
        df_with_crop_index = df.copy()
        df_with_crop_index[Column.CROP_INDEX] = (
            df_with_crop_index.groupby(required_columns, as_index=False).ngroup().astype(int)
        )

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
        unfiltered_pca_df = df_with_pcs.drop(columns=DIFFAE_FEATURE_COLUMN_NAMES)

        # Filter out annotated timepoints, except for timepoints flagged as "not
        # steady state" (those can be filtered out dynamically as necessary in
        # downstream workflows)
        logger.info(
            "Filtering %s crop-based PCA features for dataset '%s'", crop_pattern, dataset_name
        )
        timepoint_annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
        )
        filtered_pca_df = filter_dataframe_by_annotations(
            unfiltered_pca_df,
            dataset_config,
            timepoint_annotations=timepoint_annotations,
        )

        # For track-based crops, do additional filtering using the "is_included"
        # column from the segmentation features dataframe to remove the segmentations
        # that don't pass the segmentation quality control filters.
        if crop_pattern == "tracked":
            # Load and merge segmentation features dataframe to get
            # "is_included" column for filtering. Also add a track length column
            # for downstream filtering based on track length, if necessary.
            seg_feat_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
            seg_feat_loc = get_dataframe_location_for_dataset(seg_feat_manifest, dataset_name)
            df_segmentations_delayed = load_dataframe(seg_feat_loc, delay=True)
            # Columns to merge segmentation dataframe onto PCA dataframe. These
            # columns uniquely identify each crop/track in both dataframes,
            # allowing us to merge the relevant information for filtering.
            columns_to_merge_on = [
                Column.DATASET,
                Column.POSITION,
                Column.TIMEPOINT,
                Column.TRACK_ID,
            ]
            # Columns to compute from the delayed segmentation dataframe for
            # merging (merge columns + columns needed for filtering)
            columns_to_compute = [
                *columns_to_merge_on,
                Column.TRACK_LENGTH,
                Column.SegDataFilters.IS_INCLUDED,
            ]
            df_segmentations = df_segmentations_delayed[columns_to_compute].compute()
            merged_unfiltered_pca_df = filtered_pca_df.merge(
                df_segmentations,
                on=columns_to_merge_on,
                how="left",
                validate="one_to_one",
            )
            # Drop rows where "is_included" is False (i.e. segmentation didn't
            # pass QC filters).
            filtered_pca_df = merged_unfiltered_pca_df[
                merged_unfiltered_pca_df[Column.SegDataFilters.IS_INCLUDED]
            ]
            # Drop IS_INCLUDED column (no longer needed after filtering). Keep
            # TRACK_LENGTH column for potential downstream filtering based on
            # track length. Keep TRACK_ID column for use in workflow that creates
            # the merged segmentation-PCA dataframe.
            filtered_pca_df = filtered_pca_df.drop(columns=[Column.SegDataFilters.IS_INCLUDED])

        # Save dataframes to path, and, if request, upload to FMS
        for manifest, pca_df, filtering_note in [
            (unfiltered_pca_manifest, unfiltered_pca_df, "No filtering applied."),
            (filtered_pca_manifest, filtered_pca_df, "Filtering by timepoint and position."),
        ]:
            # Save the dataframe to file
            suffix = "_filtered" if manifest.parameters["filtered"] else ""
            pca_df_path = output_path / f"{dataset_name}_{crop_pattern}_pca{suffix}.parquet"
            pca_df.to_parquet(pca_df_path, index=False)

            # Create location object with output path
            pca_location = manifest.locations.get(dataset_name, DataframeLocation())
            pca_location.path = pca_df_path

            # Upload to FMS (internal only) and replace local path with file id
            if UPLOAD_TO_FMS:
                annotations = build_fms_annotations(
                    dataset=dataset_config,
                    additional_notes=(
                        "Dataframe with PCA features calculated from DiffAE latent features "
                        f"for {crop_pattern} crops. {filtering_note}"
                    ),
                )
                fmsid = upload_file_to_fms(
                    pca_df_path, annotations=annotations, file_type="parquet"
                )
                pca_location.fmsid = fmsid
                location.path = None

            # Add dataframe location to dataframe manifest and save.
            manifest.locations[dataset_name] = pca_location
            save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
