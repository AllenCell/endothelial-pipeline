
#%%
from cellsmap.extras.zarr_conversion import process_timelapse, convert
from pathlib import Path


if __name__ == "__main__":
    dataset = '20240305_T01_001'
    images = process_timelapse(dataset)
    convert(dataset, 
            Path(f'/allen/aics/assay-dev/users/Chantelle/outputs/temp_tiffs/{dataset}.zarr'),
            )
#%%