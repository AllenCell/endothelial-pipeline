import logging
from pathlib import Path

import dask.dataframe as dd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, upload_file_to_fms
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.manifests import (
    DataframeLocation,
    create_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    FIXED_SEG_FEATURE_MANIFEST_NAME,
    Column,
)

logger = logging.getLogger(__name__)


def get_model_annotations_for_upload() -> dict:
    """Return dictionary of label-free nuclei Cellpose model info for FMS upload annotations."""
    from endo_pipeline.manifests import load_model_manifest

    model_name = "nuc_pred_labelfree"
    run_name = "finetuned_20250419"
    return {"model_manifest": load_model_manifest(model_name), "run_name": run_name}


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
    if "_live_segmentation_features" in path_to_file.name:
        model_annotations = get_model_annotations_for_upload()
        seg_feature_manifest_name = DEFAULT_SEG_FEATURE_MANIFEST_NAME
    elif "_fixed_segmentation_features" in path_to_file.name:
        model_annotations = {}
        seg_feature_manifest_name = FIXED_SEG_FEATURE_MANIFEST_NAME
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


def fms_upload_merge_pc_diffae_seg_features(
    dataset_name: str,
    path_to_file: Path,
) -> str:

    # Define the metadata associated with the file being uploaded to FMS
    # The segmentations make use of label-free nuclei predictions
    # to improve segmentation quality, so we include model config
    # info along with the FMS upload here.
    dataset_config = load_dataset_config(dataset_name)
    # Get the DiffAE model annotations
    if "_pc_diffae_seg_feats_merged_filtered" in path_to_file.name:
        run_name = None
        model_manifest = None
    else:
        df = dd.read_parquet(path_to_file)
        model_manifest_name = sequence_to_scalar(
            df[Column.DiffAEData.MODEL_MANIFEST].compute().dropna()
        )
        run_name = sequence_to_scalar(df[Column.DiffAEData.MODEL_RUN].compute().dropna())
        model_manifest = load_model_manifest(model_manifest_name)
    # Prepare the annotations for FMS upload
    annotations = build_fms_annotations(
        dataset_config,
        model_manifest=model_manifest,
        run_name=run_name,
    )
    # Upload the file to FMS
    file_id = upload_file_to_fms(
        file_path=path_to_file, annotations=annotations, file_type="parquet"
    )

    # Store FMS ID in dataframe manifest
    if "_pc_diffae_seg_feats_merged_filtered" in path_to_file.name:
        manifest_name = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED
    else:
        manifest_name = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME
    workflow_name = "merge_pc_diffae_seg_features"
    manifest = create_dataframe_manifest(manifest_name, workflow_name)
    manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
    save_dataframe_manifest(manifest)

    logger.info(
        f"Dataset {dataset_name} with FMS ID {file_id} uploaded to FMS from {path_to_file}."
    )

    return file_id
