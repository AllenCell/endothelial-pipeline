from pathlib import Path
from typing import Literal

from src.endo_pipeline.configs import load_dataset_config, load_model_config, save_dataset_config
from src.endo_pipeline.configs.model_config_io import get_labelfree_nuclei_prediction_model_name
from src.endo_pipeline.io import build_fms_annotations, upload_file_to_fms


def fms_upload_cdh5_classic_seg_tracking(
    dataset_name: str, path_to_file: Path, env: Literal["stg", "prod"] = "stg"
) -> str:
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    model_name = get_labelfree_nuclei_prediction_model_name()
    dataset_config = load_dataset_config(dataset_name)
    model = load_model_config(model_name)
    annotations = build_fms_annotations(dataset_config, model=model)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="tsv", env=env
    )

    # Update the dataset config with the FMS file ID
    dataset_config.cdh5_classic_seg_tracking_manifest_fmsid = file_id  # type: ignore[attr-defined]
    save_dataset_config(dataset_config)

    return file_id


def fms_upload_cdh5_get_measured_features(
    dataset_name: str, path_to_file: Path, env: Literal["stg", "prod"] = "stg"
) -> str:
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    model_name = get_labelfree_nuclei_prediction_model_name()
    dataset_config = load_dataset_config(dataset_name)
    model = load_model_config(model_name)
    annotations = build_fms_annotations(dataset_config, model=model)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="tsv", env=env
    )

    # Update the dataset config with the FMS file ID
    dataset_config.cdh5_classic_seg_manifest_fmsid = file_id  # type: ignore[attr-defined]
    save_dataset_config(dataset_config)

    return file_id


def fms_upload_nuc_get_measured_features(
    dataset_name: str, path_to_file: Path, env: Literal["stg", "prod"] = "stg"
) -> str:
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    model_name = get_labelfree_nuclei_prediction_model_name()
    dataset_config = load_dataset_config(dataset_name)
    model = load_model_config(model_name)
    annotations = build_fms_annotations(dataset_config, model=model)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="tsv", env=env
    )

    # Update the dataset config with the FMS file ID
    dataset_config.nuclei_label_free_seg_manifest_fmsid = file_id  # type: ignore[attr-defined]
    save_dataset_config(dataset_config)

    return file_id


def fms_upload_make_seg_feats_manifest(
    dataset_name: str, path_to_file: Path, env: Literal["stg", "prod"] = "stg"
) -> str:
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    model_name = get_labelfree_nuclei_prediction_model_name()
    dataset_config = load_dataset_config(dataset_name)
    model = load_model_config(model_name)
    annotations = build_fms_annotations(dataset_config, model=model)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="tsv", env=env
    )

    # Update the dataset config with the FMS file ID
    dataset_config.merged_seg_features_manifest_fmsid = file_id  # type: ignore[attr-defined]
    save_dataset_config(dataset_config)

    return file_id
