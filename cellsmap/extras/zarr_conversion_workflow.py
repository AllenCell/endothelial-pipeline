from cellsmap.extras.zarr_conversion import convert
from pathlib import Path


if __name__ == "__main__":
    convert(dataset='20240305_T01_001',
            output_folder=Path('/allen/aics/assay-dev/users/Chantelle/outputs/temp_zarrs/'),
            fname='20240305_T01_001_test',
            save_to_zarr=False)
