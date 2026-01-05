from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    datasets: Datasets | None = None,
) -> None:
    """
    Run inference on immunofluorescence data using a pre-trained DiffAE model and centered on
    nuclear segmenation locations.
    """
    import logging

    from endo_pipeline import DEMO_MODE, NUM_GPUS
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_config,
        load_model_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.immunofluorescence import filter
    from endo_pipeline.library.model import load_model_for_inference
    from endo_pipeline.library.model.eval_model import (
        add_diffae_model_eval_crop_columns,
        generate_overrides_for_track_based_crops,
        update_prediction_from_tracks_with_metadata,
    )
    from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import (
        DIFFAE_MODEL_EVAL_CONFIG,
        LOWER_Z_SLICE_OFFSET,
        NATIVE_ZARR_RESOLUTION_CROP_SIZE,
        UPPER_Z_SLICE_OFFSET,
        ZARR_BRIGHTFIELD_CHANNEL,
        ColumnName,
        CytoDLLoadDataKeys,
        IMG_SHAPE_RESOLUTION_0_3i_X,
        IMG_SHAPE_RESOLUTION_0_3i_Y,
    )

    logger = logging.getLogger(__name__)
    output_dir = get_output_path("if_inference")

    # Load Data and add info to dataframe
    if datasets is None:
        datasets = get_datasets_in_collection("smad1")

    if DEMO_MODE:
        logger.info("Demo mode active, limiting to first dataset only.")
        datasets = datasets[:1]

    if_df_manifest = load_dataframe_manifest("immunofluorescence")

    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        df_location = get_dataframe_location_for_dataset(if_df_manifest, dataset_name)
        df_dataset = load_dataframe(df_location)
        zarr_positions = dataset_config.zarr_positions

        if DEMO_MODE:
            logger.info("Demo mode active, limiting to first position only.")
            zarr_positions = zarr_positions[:1]

        for position in zarr_positions:
            df = df_dataset[df_dataset["position"] == position]

            zarr_path = get_zarr_location_for_position(dataset_config, position).path
            if dataset_config.center_z_plane is None:
                raise ValueError(f"Dataset {dataset_name} is missing center_z_plane information.")
            center_slice = dataset_config.center_z_plane[position]

            # Filter and preprocess features for immunofluorescence.
            df = filter.filter_small_objects(df)
            df = filter.filter_img_center(df)
            df = df[df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"] < 1.0]

            if DEMO_MODE:
                logger.info("Demo mode active, using only the first 5 cells in FOV.")
                df = df.head(5)

            # Add columns required for DiffAE model inference
            df[ColumnName.ZARR_PATH] = str(zarr_path)
            df[CytoDLLoadDataKeys.Z_START] = center_slice - LOWER_Z_SLICE_OFFSET
            df[CytoDLLoadDataKeys.Z_END] = center_slice + UPPER_Z_SLICE_OFFSET
            df[CytoDLLoadDataKeys.Z_STEP] = 1
            df["image_index"] = 0
            df["centroid_X"] = df["centroid_x"]
            df["centroid_Y"] = df["centroid_y"]
            df["image_size_x"] = IMG_SHAPE_RESOLUTION_0_3i_X
            df["image_size_y"] = IMG_SHAPE_RESOLUTION_0_3i_Y
            df["crop_size"] = NATIVE_ZARR_RESOLUTION_CROP_SIZE
            df = add_diffae_model_eval_crop_columns(df)
            df["track_id"] = df["label"]

            # Adjust the crop coordinates to be consistent with the resolution level
            resolution = sequence_to_scalar(df["diffae_resolution_level_to_use"])
            columns_to_downsample = [
                ColumnName.START_X,
                ColumnName.START_Y,
                ColumnName.END_X,
                ColumnName.END_Y,
            ]
            for col in columns_to_downsample:
                df[col] = df[col] // (2**resolution)

            # group df by zarr_path and convert start and end coordinates to list
            grouped_df = (
                df.groupby([ColumnName.ZARR_PATH, "image_index"])
                .agg(
                    {
                        ColumnName.START_Y: lambda x: list(x),
                        ColumnName.START_X: lambda x: list(x),
                        ColumnName.END_Y: lambda x: list(x),
                        ColumnName.END_X: lambda x: list(x),
                        "track_id": lambda x: list(x),
                        CytoDLLoadDataKeys.Z_START: lambda x: x.iloc[0],
                        CytoDLLoadDataKeys.Z_END: lambda x: x.iloc[0],
                        CytoDLLoadDataKeys.Z_STEP: lambda x: x.iloc[0],
                    }
                )
                .reset_index()
            )
            # Add which channel to load and what resolution to load it at
            grouped_df[CytoDLLoadDataKeys.CHANNELS] = ZARR_BRIGHTFIELD_CHANNEL
            grouped_df[ColumnName.RESOLUTION] = resolution

            # only run a single timepoint from zarr
            grouped_df[CytoDLLoadDataKeys.TIME_START] = grouped_df["image_index"]
            grouped_df[CytoDLLoadDataKeys.TIME_END] = grouped_df["image_index"]
            grouped_df = grouped_df.rename(
                {
                    ColumnName.ZARR_PATH: CytoDLLoadDataKeys.FILE_PATH,
                    "image_index": CytoDLLoadDataKeys.TIMEPOINT,
                },
                axis=1,
            )

            data_path = output_dir / "aggregated_crop_manifest.parquet"
            grouped_df.to_parquet(data_path, index=False)

            # Load model for inference
            model_manifest = load_model_manifest(model_manifest_name)
            eval_config = load_model_config(DIFFAE_MODEL_EVAL_CONFIG)
            model = load_model_for_inference(model_manifest, run_name, eval_config)

            prediction_filename_suffix = (
                f"{dataset_name}_P{position}_{model_manifest_name}_{run_name}"
            )
            prediction_filename_suffix = f"{prediction_filename_suffix}_if_crop_features"
            overrides = generate_overrides_for_track_based_crops(
                save_path=output_dir.as_posix(),
                data_path=data_path.as_posix(),
                dataset_name=dataset_config.name,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                prediction_filename_suffix=prediction_filename_suffix,
                num_gpus=NUM_GPUS,
            )

            model.override_config(overrides)
            local_config_save_path = get_output_path(
                "models", "evaluation_configs", model_manifest_name, run_name, "if_crops"
            )
            model.save_config(local_config_save_path / "eval.yaml")
            logger.info(
                "Evaluation config saved to [ %s ]",
                local_config_save_path / "eval.yaml",
            )
            model.predict()

            prediction_path = output_dir / f"predict_{prediction_filename_suffix}.parquet"
            update_prediction_from_tracks_with_metadata(
                dataset_name=dataset_config.name,
                model_manifest_name=model_manifest_name,
                run_name=run_name,
                prediction_path=prediction_path,
            )

    if __name__ == "__main__":
        from endo_pipeline.__main__ import workflow_cli

        workflow_cli(main)
