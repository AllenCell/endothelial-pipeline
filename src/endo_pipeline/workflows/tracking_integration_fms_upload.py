from pathlib import Path

from src.endo_pipeline.configs import load_dataset_config
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from src.endo_pipeline.io import build_fms_annotations, upload_file_to_fms

"""
This is a rough script to get track integration for Benji and Erin
quickly before SAC.
"""


def main(dataset_name: str) -> None:
    integration_data_dir = Path(
        "//allen/aics/endothelial/morphological_features/single_cell_track_integration"
    )
    integration_data_paths = [dataset for dataset in integration_data_dir.glob("*.csv")]
    file_path = [path.as_posix() for path in integration_data_paths if dataset_name in path.name][0]

    dataset_config = load_dataset_config(dataset_name)
    annotations = build_fms_annotations(
        dataset_config,
        additional_notes=(
            "This is an initial effort for tracking integration. "
            "The tracking data still needs manual curation."
        ),
    )
    upload_file_to_fms(file_path, annotations, "csv")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
