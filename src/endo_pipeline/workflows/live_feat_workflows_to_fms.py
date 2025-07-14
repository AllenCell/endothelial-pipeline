import logging
from pathlib import Path
from typing import Literal

import fire

from src.endo_pipeline.configs import load_dataset_config, load_model_config, save_dataset_config
from src.endo_pipeline.configs.model_config_io import get_labelfree_nuclei_prediction_model_name
from src.endo_pipeline.io import (
    build_fms_annotations,
    configure_logging,
    get_output_path,
    upload_file_to_fms,
)

"""
These functions are used to upload feature tables to FMS.
NOTE These functions DO NOT and WILL NOT work on Windows.
They must be run on the Allen Institute intranet either
in a Linux or MacOS environment through the CLI.
"""

logger = logging.getLogger(__name__)
out_dir = get_output_path(Path(__file__).stem, include_timestamp=False)
configure_logging(out_dir, logger, verbose=True)


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

    logger.info(
        f"[Environment: {env}] Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

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

    logger.info(
        f"[Environment: {env}] Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

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

    logger.info(
        f"[Environment: {env}] Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

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
    dataset_config.live_merged_seg_features_manifest_fmsid = file_id  # type: ignore[attr-defined]
    save_dataset_config(dataset_config)

    logger.info(
        f"[Environment: {env}] Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


if __name__ == "__main__":
    fire.Fire()
