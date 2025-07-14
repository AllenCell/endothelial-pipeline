import logging
from pathlib import Path
from typing import Literal

import fire
from tqdm import tqdm

from src.endo_pipeline.configs import (
    load_all_dataset_configs,
    load_dataset_config,
    load_model_config,
    save_dataset_config,
)
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


def upload_multiple_datasets(
    manifest_kind: Literal[
        "cdh5_seg_tracking",
        "cdh5_seg_measurements",
        "nuclei_labelfree",
        "merged_live_data_manifests",
    ],
    dataset_name_list: list | None = None,
    fms_env: Literal["stg", "prod"] = "prod",
) -> None:
    """
    This is a convenience function to upload multiple datasets to FMS
    from the endothelial project folder.

    NOTE Intended only for internal use.
    """
    assert Path("//allen/aics/endothelial/morphological_features/analysis").exists(), (
        "The path to the endothelial project directory is not accessible."
        "This function is only available for Allen Institute internal use."
    )
    fms_upload_func_dict = {
        "cdh5_seg_tracking": fms_upload_cdh5_classic_seg_tracking,
        "cdh5_seg_measurements": fms_upload_cdh5_get_measured_features,
        "nuclei_labelfree": fms_upload_nuc_get_measured_features,
        "merged_live_data_manifests": fms_upload_make_seg_feats_manifest,
    }
    if dataset_name_list is None:
        # This is the current list of all analyzed datasets
        dataset_name_list = [
            "20241016_20X",
            "20241120_20X",
            "20241217_20X",
            "20250224_20X",
            "20250319_20X",
            "20250326_20X",
            "20250331_20X",
            "20250409_20X",
            "20250428_20X",
            "20250604_20X",
            "20250611_20X",
        ]
    else:
        pass

    available_live_datasets = []
    for ds_cfg in load_all_dataset_configs():
        if ds_cfg.live_or_fixed_sample == "live":
            available_live_datasets.append(ds_cfg.name)

    endo_project_dir = Path("//allen/aics/endothelial/morphological_features/analysis")
    path_modifiers = {
        "cdh5_seg_tracking": {"subdir": "cdh5_classic_seg_tracking", "suffix": "_tracking.tsv"},
        "cdh5_seg_measurements": {
            "subdir": "cdh5_get_measured_features",
            "suffix": "_cdh5_segprops.tsv",
        },
        "nuclei_labelfree": {
            "subdir": "nuc_labelfree_get_measured_features",
            "suffix": "_nuclei_labelfree_features.tsv",
        },
        "merged_live_data_manifests": {
            "subdir": "segmentation_features",
            "suffix": "_live_segmentation_features.tsv",
        },
    }

    for dataset_name in tqdm(dataset_name_list):
        assert dataset_name in available_live_datasets, (
            f"Dataset {dataset_name} is not in the list of available live datasets: "
            f"{available_live_datasets}"
        )
        manifest_filepath = (
            endo_project_dir
            / path_modifiers[manifest_kind]["subdir"]
            / f"{dataset_name}{path_modifiers[manifest_kind]['suffix']}"
        )
        assert manifest_filepath.exists(), (
            f"Manifest file {manifest_filepath} does not exist. "
            "Please double check the file location."
        )
        fms_upload_func_dict[manifest_kind](dataset_name, manifest_filepath, fms_env)


if __name__ == "__main__":
    fire.Fire()
