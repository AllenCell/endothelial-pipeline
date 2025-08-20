# %%
import pandas as pd

from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.model import MultiDimImageDataset

# %%

zarr_file_path = (
    "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/"
    "20241016_230d119061e749d98c1abde77f2f4fa3/20241016_230d119061e749d98c1abde77f2f4fa3_P0.ome.zarr"
)

resolution = 1
start = 300
stop = 500
step = 50
channel = [0, 1]
exclude_frames = [350, 400]

df = pd.DataFrame(
    {
        "path": [zarr_file_path],
        "resolution": [resolution],
        "channel": [channel],
        "frame_start": [start],
        "frame_stop": [stop],
        "frame_step": [step],
        "exclude_frames": [exclude_frames],
    }
)

output_path = get_output_path("test_image_loading")
file_path = output_path / "image_loading_test.parquet"

df.to_parquet(file_path, index=False)
# %%
image_dataset = MultiDimImageDataset(dataframe_path=file_path)
# %%
