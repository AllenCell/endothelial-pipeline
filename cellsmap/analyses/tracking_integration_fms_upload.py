import subprocess
from pathlib import Path
from typing import Literal

from cellsmap.util.dataset_io import ipython_cli_flexecute
from cellsmap.util.manifest_preprocessing.fms_upload import save_file_to_fms

"""
This is a rough script to get track integration for Benji and Erin
quickly before SAC.
"""


def main(dataset_name: str, env: Literal["stg", "prod"]) -> None:
    # the current commit hash:
    git_commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    integration_data_dir = Path(
        "//allen/aics/endothelial/morphological_features/single_cell_track_integration"
    )
    integration_data_paths = [dataset for dataset in integration_data_dir.glob("*.csv")]
    file_path = [
        path.as_posix() for path in integration_data_paths if dataset_name in path.name
    ][0]

    notes = f"""
    Dataset: {dataset_name}.
    This is an initial effort for tracking integration.
    The tracking data still needs manual curation.
    """

    save_file_to_fms(
        file_path=file_path,
        dataset=dataset_name,
        commit_hash=git_commit_hash,
        file_type="csv",
        misc_notes=notes,
        env=env,
    )


if __name__ == "__main__":
    ipython_cli_flexecute(main)
