from typing import Literal

from endo_pipeline.cli import Datasets


def main(
    table_type: Literal["cdh5_seg_measurements",],
    datasets: Datasets,
) -> None:
    """
    Upload selected table to FMS and update corresponding dataframe manifest.

    #internal #fms #vast

    This workflow supports uploads for the following table types.

    | Table                   | Workflow                       |
    | ----------------------- | ------------------------------ |
    | `cdh5_seg_measurements` | `get_cdh5_measured_features`   |

    Tables are produced by the listed workflow, and must be copied to Vast at
    `//allen/aics/endothelial/morphological_features/analysis` before running
    this upload workflow because FMS can only upload files located on Vast.

    Parameters
    ----------
    table_type
        Table type to upload.
    datasets
        List of datasets or dataset collections to upload.
    """

    import logging
    from collections import namedtuple
    from pathlib import Path

    from tqdm import tqdm

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, make_name_unique, upload_file_to_fms
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )

    logger = logging.getLogger(__name__)

    root_path = Path("//allen/aics/endothelial/morphological_features/analysis").resolve()

    TableUploadArgs = namedtuple("ManifestUploadArgs", ["subdir", "suffix", "manifest", "workflow"])
    table_upload_args = {
        "cdh5_seg_measurements": TableUploadArgs(
            subdir="cdh5_get_measured_features",
            suffix="_cdh5_segprops.parquet",
            manifest="cdh5_classic_segmentation",
            workflow="get_cdh5_measured_features",
        ),
    }

    # Select upload arguments for the selected table type
    upload_args = table_upload_args[table_type]

    # Specify label-free nuclei prediction model for upload annotations
    model_manifest = load_model_manifest("nuc_pred_labelfree")
    run_name = "finetuned_20250419"

    for dataset_name in tqdm(datasets):
        # Load dataset config
        dataset_config = load_dataset_config(dataset_name)

        # Build expected path to file
        path_to_table = root_path / upload_args.subdir / f"{dataset_name}{upload_args.suffix}"

        # Check if the file exists
        if not path_to_table.exists():
            error_msg = (
                f"Table file '{path_to_table}' does not exist. "
                "Please double check the file location."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Add timestamp to the table file name and rename it to ensure we keep a
        # local copy of different versions and that the file name is unique
        path_to_table_unique = make_name_unique(Path(path_to_table))
        path_to_table.rename(path_to_table_unique)

        # Build annotations and upload to FMS
        annotations = build_fms_annotations(
            dataset_config,
            model_manifest=model_manifest,
            run_name=run_name,
        )
        file_id = upload_file_to_fms(
            file_path=path_to_table_unique,
            annotations=annotations,
            file_type="parquet",
        )

        # Store FMS ID in dataframe manifest
        manifest = create_dataframe_manifest(upload_args.manifest, upload_args.workflow)
        manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
        save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
