import logging
from pathlib import Path

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    create_dataframe_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME

logger = logging.getLogger(__name__)


def get_model_annotations_for_upload() -> dict:
    """Return dictionary of label-free nuclei Cellpose model info for FMS upload annotations."""
    from endo_pipeline.manifests import load_model_manifest

    model_name = "nuc_pred_labelfree"
    run_name = "finetuned_20250419"
    return {"model_manifest": load_model_manifest(model_name), "run_name": run_name}


def fms_upload_cdh5_classic_seg_tracking(dataset_name: str, path_to_file: Path) -> str:
    """Upload the cdh5 segmentation tracking results to FMS and store the FMS ID in a manifest."""
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    dataset_config = load_dataset_config(dataset_name)
    model_annotations = get_model_annotations_for_upload()
    annotations = build_fms_annotations(dataset_config, **model_annotations)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "cdh5_classic_segmentation_tracking"
    workflow_name = "live_feat_workflows_to_fms"
    manifest = create_dataframe_manifest(manifest_name, workflow_name)
    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_cdh5_get_measured_features(dataset_name: str, path_to_file: Path) -> str:
    """Upload the cdh5 segmentation features to FMS and store the FMS ID in a manifest."""
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    dataset_config = load_dataset_config(dataset_name)
    model_annotations = get_model_annotations_for_upload()
    annotations = build_fms_annotations(dataset_config, **model_annotations)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "cdh5_classic_segmentation"
    workflow_name = "live_feat_workflows_to_fms"
    manifest = create_dataframe_manifest(manifest_name, workflow_name)
    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_nuc_get_measured_features(dataset_name: str, path_to_file: Path) -> str:
    """Upload the nuclei label-free features to FMS and store the FMS ID in a manifest."""
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    dataset_config = load_dataset_config(dataset_name)
    model_annotations = get_model_annotations_for_upload()
    annotations = build_fms_annotations(dataset_config, **model_annotations)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    manifest_name = "nuclei_label_free_segmentation"
    workflow_name = "live_feat_workflows_to_fms"
    manifest = create_dataframe_manifest(manifest_name, workflow_name)
    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id


def fms_upload_make_seg_feats_manifest(
    dataset_name: str,
    path_to_file: Path,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
) -> str:
    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    dataset_config = load_dataset_config(dataset_name)
    model_annotations = get_model_annotations_for_upload()
    annotations = build_fms_annotations(dataset_config, **model_annotations)

    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    workflow_name = "live_feat_workflows_to_fms"
    manifest = create_dataframe_manifest(seg_feature_manifest_name, workflow_name)
    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id
