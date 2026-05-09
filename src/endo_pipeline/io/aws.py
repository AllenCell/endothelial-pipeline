import logging
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from endo_pipeline.io.output import get_output_path

logger = logging.getLogger(__name__)


def download_s3_file_to_path(s3uri: str) -> Path:
    """
    Get local path to config file from given MLFlow run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.

    Returns
    -------
    :
        Local path to config file.
    """

    bucket, _, key = s3uri[5:].partition("/")

    # Check if file is already downloaded.
    output_path = get_output_path("s3_downloads", bucket, include_timestamp=False)
    path = output_path / key
    path.parent.mkdir(parents=True, exist_ok=True)

    # Log warning about using existing downloaded file
    if path.exists():
        logger.warning(
            "File [ %s ] available at [ %s ]. "
            "Using this file. If you want to redownload from S3, delete this file.",
            s3uri,
            path,
        )
        return path

    # Download file from S3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket, key, path)

    return path
