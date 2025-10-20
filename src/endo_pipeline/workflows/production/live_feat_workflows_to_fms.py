"""
These functions are used to upload feature tables to FMS.
NOTE These functions DO NOT and WILL NOT work on Windows.
They must be run on the Allen Institute intranet either
in a Linux or MacOS environment through the CLI.
"""

import logging
from pathlib import Path
from typing import Literal

from endo_pipeline.__main__ import workflow_cli
from endo_pipeline.cli import Datasets


def main(
    manifest_kind: Literal[
        "cdh5_seg_tracking",
        "cdh5_seg_measurements",
        "nuclei_labelfree",
        "merged_live_data_manifests",
    ],
    datasets: Datasets,
) -> None:
    """
    Upload specified feature table to FMS from the endothelial project folder for multiple datasets.

    NOTE Intended only for internal use.
    """
    import os

    from tqdm import tqdm

    from endo_pipeline.configs import load_all_dataset_configs
    from endo_pipeline.io import get_timestamp
    from endo_pipeline.library.process.lib_live_feat_workflows_to_fms import (
        fms_upload_cdh5_classic_seg_tracking,
        fms_upload_cdh5_get_measured_features,
        fms_upload_make_seg_feats_manifest,
        fms_upload_nuc_get_measured_features,
    )

    endo_project_analysis_dir = Path(
        "//allen/aics/endothelial/morphological_features/analysis"
    ).resolve()
    if not endo_project_analysis_dir.exists() or os.name == "nt":
        error_msg = f"""The path to the endothelial project directory ({endo_project_analysis_dir})
            is not accessible.
            Possible reasons are that you are not on the Allen Institute network or that you are
            not using a Linux or Mac machine. FMS uploads are only permitted from Linux/Mac
            machines that are directly connected to the Allen Institute network."""
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    fms_upload_func_dict = {
        "cdh5_seg_tracking": fms_upload_cdh5_classic_seg_tracking,
        "cdh5_seg_measurements": fms_upload_cdh5_get_measured_features,
        "nuclei_labelfree": fms_upload_nuc_get_measured_features,
        "merged_live_data_manifests": fms_upload_make_seg_feats_manifest,
    }
    print(f"Uploading {datasets}")

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

    for dataset_name in tqdm(datasets):
        if dataset_name not in available_live_datasets:
            error_msg = (
                f"Dataset {dataset_name} is not in the list of available live datasets: "
                f"{available_live_datasets}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        manifest_filepath = (
            endo_project_analysis_dir
            / path_modifiers[manifest_kind]["subdir"]
            / f"{dataset_name}{path_modifiers[manifest_kind]['suffix']}"
        )
        if not manifest_filepath.exists():
            error_msg = (
                f"Manifest file {manifest_filepath} does not exist. "
                "Please double check the file location."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        # add timestamp to the manifest filename and rename it
        timestamp = get_timestamp()
        manifest_filepath_timestamped = manifest_filepath.with_name(
            f"{manifest_filepath.stem}_fms{timestamp}{manifest_filepath.suffix}"
        )
        manifest_filepath.rename(manifest_filepath_timestamped)

        fms_upload_func_dict[manifest_kind](dataset_name, manifest_filepath_timestamped)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    workflow_cli(main)
