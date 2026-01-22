from endo_pipeline.cli import Datasets, tags

TAGS = ["immunofluorescence", tags.TEST_READY, tags.CPU_ONLY]


def main(datasets: Datasets | None = None) -> None:
    """
    Generate an immunofluorescence manifest for a given dataset.

    Use demo mode to process one position and skip uploading to FMS / updating manifest config.

    Parameters
    ----------
    datasets
        List of dataset names to process. If None, processes all datasets in the
        "immunofluorescence" collection. If DEMO_MODE is enabled,
        only the first position in the first dataset will be processed.
    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.library.model.eval_model import add_diffae_model_eval_crop_columns
    from endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction
    from endo_pipeline.library.process.if_manifest import (
        save_dataframe_to_parquet,
        update_dataframe_manifest,
        upload_manifest_to_fms,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings import (
        LOWER_Z_SLICE_OFFSET,
        NATIVE_ZARR_RESOLUTION_CROP_SIZE,
        UPPER_Z_SLICE_OFFSET,
        ColumnName,
        CytoDLLoadDataKeys,
        IMG_SHAPE_RESOLUTION_0_3i_X,
        IMG_SHAPE_RESOLUTION_0_3i_Y,
    )

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("immunofluorescence")

    if DEMO_MODE:
        logger.info("DEMO_MODE enabled: only processing the first dataset")
        datasets = datasets[:1]

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        dataset_config = load_dataset_config(dataset)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]

        # Step 1: Run feature extraction
        df = run_nuclei_feature_extraction(dataset_config, positions)

        # add data info to dataframe
        df["date"] = dataset_config.date
        shear_regime = "_to_".join([shear.value for shear in dataset_config.shear_stress_regime])
        df["shear_stress_regime"] = shear_regime

        shear_stress_list = [condition.shear_stress for condition in dataset_config.flow_conditions]
        df["shear_stress_1"] = shear_stress_list[0]
        df["shear_stress_2"] = shear_stress_list[1] if len(shear_stress_list) > 1 else np.nan

        durations = [
            condition.stop - condition.start for condition in dataset_config.flow_conditions
        ]
        duration_1 = durations[0]
        duration_2 = durations[1] if len(durations) > 1 else np.nan

        df["duration_at_ss_1_hr"] = duration_1 * 5 / 60  # convert to hrs
        df["duration_at_ss_2_hr"] = duration_2 * 5 / 60  # convert to hrs

        # add columns required for DiffAE model inference
        zarr_positions = dataset_config.zarr_positions
        for position in zarr_positions:
            zarr_path = get_zarr_location_for_position(dataset_config, position).path
            if dataset_config.center_z_plane is None:
                raise ValueError(f"Dataset {dataset} is missing center_z_plane information.")
            center_slice = dataset_config.center_z_plane[position]

            df.loc[df["position"] == position, ColumnName.ZARR_PATH] = str(zarr_path)
            df.loc[df["position"] == position, CytoDLLoadDataKeys.Z_START] = (
                center_slice - LOWER_Z_SLICE_OFFSET
            )
            df.loc[df["position"] == position, CytoDLLoadDataKeys.Z_END] = (
                center_slice + UPPER_Z_SLICE_OFFSET
            )

        df[CytoDLLoadDataKeys.Z_STEP] = 1
        df["image_index"] = 0  # IF data only have one image per position
        df["centroid_X"] = df["centroid_x"]
        df["centroid_Y"] = df["centroid_y"]
        df["image_size_x"] = IMG_SHAPE_RESOLUTION_0_3i_X
        df["image_size_y"] = IMG_SHAPE_RESOLUTION_0_3i_Y
        df["crop_size"] = NATIVE_ZARR_RESOLUTION_CROP_SIZE
        df = add_diffae_model_eval_crop_columns(df)
        # track ids need to be unique across positions
        # Position 0, trackids 1,2,3 -> track_ids 11,12,13
        # Position 1, trackids 1,2,3 -> track_ids 21,22,23
        if len(zarr_positions) > 9:
            raise ValueError("Position index exceeds 9, track_id generation may not be unique.")
        track_id_str = (df["position"] + 1).astype(str) + df["label"].astype(str)
        df["track_id"] = track_id_str.astype(int)

        # Step 2: Save to CSV
        save_path = save_dataframe_to_parquet(dataset, df)

        if DEMO_MODE:
            logger.info("DEMO_MODE mode enabled. Skipping fms upload and dataset config update")
            continue

        # Step 3: Upload to FMS
        fms_id = upload_manifest_to_fms(save_path, dataset)

        # Step 4: Update dataset configuration
        update_dataframe_manifest(dataset, fms_id)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
