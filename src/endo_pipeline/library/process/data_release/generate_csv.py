from pathlib import Path

import pandas as pd

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import make_name_unique
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_image_manifest,
)
from endo_pipeline.settings.data_release import (
    DEST_CDH5_SEG_DIR,
    DEST_COL,
    DEST_NUC_SEG_DIR,
    S3_INTERNAL_DIRECTORY,
    SOURCE_COL,
)


def create_s3_upload_csv(
    datasets: Datasets,
    save_dir: Path,
    s3_directory: str = S3_INTERNAL_DIRECTORY,
    source_col: str = SOURCE_COL,
    dest_col: str = DEST_COL,
    positions_list: list[int] | None = None,
    raw_zarr: bool = False,
    segmentation_zarr: bool = False,
) -> str:
    """
    This function a CSV defining files to upload to S3.

    Parameters
    ----------
    datasets:
        List of dataset names to include in the CSV. If None, all datasets in the
        "dataset_release" collection will be used.
    save_dir:
        Directory where the generated CSV will be saved.
    s3_directory:
        Prefix S3 directory where the zarr files will be uploaded.
    source_col:
        Name of the column for local zarr paths in the CSV. In the future, this could
        be the staging s3 location if uploading from there to the final s3 location.
    dest_col:
        Name of the column for S3 zarr paths in the CSV.
    raw_zarr:
        Whether to include raw image zarrs in the CSV.
    segmentation_zarr:
        Whether to include segmentation zarrs in the CSV.
        If True, both nuclear and VE-cadherin segmentation zarrs will be included.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    if raw_zarr is False and segmentation_zarr is False:
        raise ValueError(
            "At least one of raw_zarr or segmentation_zarr must be True to generate a remove CSV."
        )

    rows = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)

        if positions_list is None:
            positions_list = dataset_config.zarr_positions

        for position in positions_list:
            if raw_zarr:
                img_location = get_zarr_location_for_position(dataset_config, position)
                zarr_name = img_location.path.name
                rows.append(
                    {
                        source_col: str(img_location.path),
                        dest_col: s3_directory + zarr_name,
                    }
                )
            if segmentation_zarr:
                for manifest_name, destination_dir in [
                    ("nuclear_labelfree_seg_zarr", DEST_NUC_SEG_DIR),
                    ("cdh5_classic_seg_zarr", DEST_CDH5_SEG_DIR),
                ]:
                    img_manifest = load_image_manifest(manifest_name)
                    img_location = get_image_location_for_dataset(
                        img_manifest, dataset_config, position
                    )
                    zarr_name = img_location.path.name
                    rows.append(
                        {
                            source_col: str(img_location.path),
                            dest_col: s3_directory + destination_dir + zarr_name,
                        }
                    )

    df = pd.DataFrame(rows)
    file_path = make_name_unique(save_dir / "upload_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)


def create_s3_remove_csv(
    datasets: Datasets,
    save_dir: Path,
    s3_directory: str = S3_INTERNAL_DIRECTORY,
    target_col: str = DEST_COL,
    positions_list: list[int] | None = None,
    raw_zarr: bool = False,
    segmentation_zarr: bool = False,
) -> str:
    """
    This function creates a CSV defining files to remove from S3.

    Parameters
    ----------
    datasets:
        List of dataset names to include in the CSV. If None, all datasets in the
        "dataset_release" collection will be used.
    save_dir:
        Directory where the generated CSV will be saved.
    s3_directory:
        Prefix S3 directory where the zarr files will be removed from.
    target_column:
        Name of the column for S3 zarr paths in the CSV.
    positions_list:
        List of position indices to include in the CSV. If None, all positions will be included
    raw_zarr:
        Whether to include raw image zarrs in the CSV.
    segmentation_zarr:
        Whether to include segmentation zarrs in the CSV.
        If True, both nuclear and VE-cadherin segmentation zarrs will be included.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    if raw_zarr is False and segmentation_zarr is False:
        raise ValueError(
            "At least one of raw_zarr or segmentation_zarr must be True to generate a remove CSV."
        )

    rows = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)

        if positions_list is None:
            positions_list = dataset_config.zarr_positions

        for position in positions_list:
            if raw_zarr:
                img_location = get_zarr_location_for_position(dataset_config, position)
                zarr_name = img_location.path.name
                s3_zarr_path = s3_directory + zarr_name
                rows.append(
                    {
                        target_col: str(s3_zarr_path),
                    }
                )

            if segmentation_zarr:
                for manifest_name, destination_dir in [
                    ("nuclear_labelfree_seg_zarr", DEST_NUC_SEG_DIR),
                    ("cdh5_classic_seg_zarr", DEST_CDH5_SEG_DIR),
                ]:
                    img_manifest = load_image_manifest(manifest_name)
                    img_location = get_image_location_for_dataset(
                        img_manifest, dataset_config, position
                    )
                    zarr_name = img_location.path.name
                    s3_zarr_path = s3_directory + destination_dir + zarr_name
                    rows.append(
                        {
                            target_col: str(s3_zarr_path),
                        }
                    )

    df = pd.DataFrame(rows)
    file_path = make_name_unique(save_dir / "remove_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)
