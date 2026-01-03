from endo_pipeline import DEMO_MODE, NUM_GPUS
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

    import numpy as np

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
    )
    from endo_pipeline.settings.image_data import (
        IMG_SHAPE_RESOLUTION_0_3i_X,
        IMG_SHAPE_RESOLUTION_0_3i_Y,
    )

    logger = logging.getLogger(__name__)
    output_dir = get_output_path("if_inference")

    # Load Data and add info to dataframe

    if datasets is None:
        datasets = get_datasets_in_collection("smad1")
    if_df_manifest = load_dataframe_manifest("immunofluorescence")

    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        df_location = get_dataframe_location_for_dataset(if_df_manifest, dataset_name)
        df_dataset = load_dataframe(df_location)

        for position in dataset_config.zarr_positions:
            df = df_dataset[df_dataset["position"] == position]

            df[ColumnName.ZARR_PATH] = str(
                get_zarr_location_for_position(dataset_config, position).path
            )
            df[CytoDLLoadDataKeys.Z_START] = (
                dataset_config.center_z_plane[position] - LOWER_Z_SLICE_OFFSET
            )
            df[CytoDLLoadDataKeys.Z_END] = (
                dataset_config.center_z_plane[position] + UPPER_Z_SLICE_OFFSET
            )
            df[CytoDLLoadDataKeys.Z_STEP] = 1
            df["date"] = dataset_config.date

            shear_regime = "_to_".join(
                [shear.value for shear in dataset_config.shear_stress_regime]
            )
            df["shear_stress_regime"] = shear_regime
            shear_stress_list = [
                condition.shear_stress for condition in dataset_config.flow_conditions
            ]
            df["shear_stress_1"] = shear_stress_list[0]
            df["shear_stress_2"] = shear_stress_list[1] if len(shear_stress_list) > 1 else np.nan

            durations = [
                condition.stop - condition.start for condition in dataset_config.flow_conditions
            ]
            df["duration_at_ss_1_hr"] = durations[0] * 5 / 60
            df["duration_at_ss_2_hr"] = (durations[1] * 5 / 60) if len(durations) > 1 else np.nan

            # Filter and preprocess features for immunofluorescence analysis.
            df = filter.filter_small_objects(df)
            df = filter.filter_img_center(df)
            df["SMAD1_norm_NucViolet_mean_sum_proj"] = (
                df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]
            )
            df["SMAD1_norm_area_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["area"]
            df = df[df["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]

            # Prepare dataframe for DiffAE model inference
            df["image_index"] = 0
            df["centroid_X"] = df["centroid_x"]
            df["centroid_Y"] = df["centroid_y"]
            df["image_size_x"] = IMG_SHAPE_RESOLUTION_0_3i_X
            df["image_size_y"] = IMG_SHAPE_RESOLUTION_0_3i_Y
            df["crop_size"] = NATIVE_ZARR_RESOLUTION_CROP_SIZE
            df = add_diffae_model_eval_crop_columns(df)

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
            if DEMO_MODE:
                data_path = output_dir / "DEMO_aggregated_crop_manifest.parquet"
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
