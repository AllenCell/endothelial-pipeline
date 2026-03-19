import logging
from pathlib import Path

import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    create_dataframe_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.optical_flow import DEFAULT_OPTICAL_FLOW_MANIFEST_NAME

logger = logging.getLogger(__name__)


def _save_optical_flow_parquet(dataset: str, df: pd.DataFrame) -> Path:
    """Write a per-dataset optical-flow dataframe to parquet.

    Parameters
    ----------
    dataset
        Dataset name, used to construct the output filename.
    df
        DataFrame to persist.

    Returns
    -------
        Absolute path to the written parquet file.
    """
    output_dir = get_output_path("optical_flow", "manifests")
    parquet_path = output_dir / f"{dataset}_optical_flow_manifest.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet: %s", parquet_path)
    return parquet_path


def save_and_upload_optical_flow_df(
    dataset: str,
    df: pd.DataFrame,
    workflow: str = "optical_flow",
    parameters: dict | None = None,
) -> str:
    """Save a parquet, upload it to FMS, and register it in the manifest.

    Parameters
    ----------
    dataset
        Dataset name used for the parquet filename and manifest key.
    df
        Optical-flow feature DataFrame to persist.
    workflow
        Workflow name recorded in the manifest.  Callers should pass
        ``Path(__file__).stem`` so the manifest points back to the
        invoking script.
    parameters
        Workflow parameters to store in the manifest for reproducibility
        (e.g. channel, max_dt, annotations excluded).  Merged into any
        existing manifest parameters on each call.

    Returns
    -------
        FMS file identifier for the uploaded parquet.
    """
    logger.info("Saving and uploading results for %s", dataset)
    parquet_path = _save_optical_flow_parquet(dataset, df)
    dataset_config = load_dataset_config(dataset)
    fms_id = upload_file_to_fms(parquet_path, build_fms_annotations(dataset_config), "parquet")
    logger.info("Uploaded to FMS: %s", fms_id)
    manifest = create_dataframe_manifest(DEFAULT_OPTICAL_FLOW_MANIFEST_NAME, workflow)
    if parameters:
        manifest.parameters.update(parameters)
    manifest.locations[dataset] = DataframeLocation(fmsid=fms_id)
    save_dataframe_manifest(manifest)
    return fms_id
