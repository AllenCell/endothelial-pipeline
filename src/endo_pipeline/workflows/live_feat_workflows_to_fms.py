import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from endo_pipeline.configs import load_all_dataset_configs, load_dataset_config, load_model_config
from endo_pipeline.configs.model_config_utils import get_labelfree_nuclei_prediction_model_name
from endo_pipeline.io import (
    build_fms_annotations,
    configure_logging,
    get_output_path,
    upload_file_to_fms,
)
from endo_pipeline.manifests import (
    DataframeLocation,
    DataframeManifest,
    load_dataframe_manifest,
    save_dataframe_manifest,
)

"""
These functions are used to upload feature tables to FMS.
NOTE These functions DO NOT and WILL NOT work on Windows.
They must be run on the Allen Institute intranet either
in a Linux or MacOS environment through the CLI.
"""

logger = logging.getLogger(__name__)
out_dir = get_output_path(__file__, include_timestamp=False)
configure_logging(out_dir, logger, verbose=True)


def fms_upload_cdh5_classic_seg_tracking(dataset_name: str, path_to_file: Path) -> str:
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
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "cdh5_classic_segmentation_tracking"
    workflow_name = "live_feat_workflows_to_fms"

    try:
        manifest = load_dataframe_manifest(manifest_name)
    except FileNotFoundError:
        manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_cdh5_get_measured_features(dataset_name: str, path_to_file: Path) -> str:
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
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "cdh5_classic_segmentation"
    workflow_name = "live_feat_workflows_to_fms"

    try:
        manifest = load_dataframe_manifest(manifest_name)
    except FileNotFoundError:
        manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_nuc_get_measured_features(dataset_name: str, path_to_file: Path) -> str:
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
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "nuclei_label_free_segmentation"
    workflow_name = "live_feat_workflows_to_fms"

    try:
        manifest = load_dataframe_manifest(manifest_name)
    except FileNotFoundError:
        manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_make_seg_feats_manifest(dataset_name: str, path_to_file: Path) -> str:
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
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "live_merged_seg_features"
    workflow_name = "live_feat_workflows_to_fms"

    try:
        manifest = load_dataframe_manifest(manifest_name)
    except FileNotFoundError:
        manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def main(
    manifest_kind: Literal[
        "cdh5_seg_tracking",
        "cdh5_seg_measurements",
        "nuclei_labelfree",
        "merged_live_data_manifests",
    ],
    dataset_name_list: list | None = None,
    endo_project_analysis_dir: (
        str | Path
    ) = "//allen/aics/endothelial/morphological_features/analysis",
) -> None:
    """
    This is a convenience function to upload multiple datasets to FMS
    from the endothelial project folder.

    NOTE Intended only for internal use.
    """
    endo_project_analysis_dir = Path(endo_project_analysis_dir).resolve()
    assert endo_project_analysis_dir.exists(), (
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

    all_available_datasets = load_all_dataset_configs()
    available_live_datasets = []
    for ds_cfg in all_available_datasets:
        if ds_cfg.live_or_fixed_sample == "live":
            available_live_datasets.append(ds_cfg.name)

    path_modifiers = {
        "cdh5_seg_tracking": {"subdir": "cdh5_classic_seg_tracking", "suffix": "_tracking.parquet"},
        "cdh5_seg_measurements": {
            "subdir": "cdh5_get_measured_features",
            "suffix": "_cdh5_segprops.parquet",
        },
        "nuclei_labelfree": {
            "subdir": "nuc_labelfree_get_measured_features",
            "suffix": "_nuclei_labelfree_features.parquet",
        },
        "merged_live_data_manifests": {
            "subdir": "cdh5_live_seg_features",
            "suffix": "_live_segmentation_features.parquet",
        },
    }

    for dataset_name in tqdm(dataset_name_list):
        assert dataset_name in available_live_datasets, (
            f"Dataset {dataset_name} is not in the list of available live datasets: "
            f"{available_live_datasets}"
        )
        manifest_filepath = (
            endo_project_analysis_dir
            / path_modifiers[manifest_kind]["subdir"]
            / f"{dataset_name}{path_modifiers[manifest_kind]['suffix']}"
        )
        assert manifest_filepath.exists(), (
            f"Manifest file {manifest_filepath} does not exist. "
            "Please double check the file location."
        )
        # add timestamp to the manifest filename and rename it
        timestamp = datetime.now(UTC).strftime("%Y%m%d")
        manifest_filepath_timestamped = manifest_filepath.with_name(
            f"{manifest_filepath.stem}_fms{timestamp}{manifest_filepath.suffix}"
        )
        manifest_filepath.rename(manifest_filepath_timestamped)

        fms_upload_func_dict[manifest_kind](dataset_name, manifest_filepath_timestamped)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
