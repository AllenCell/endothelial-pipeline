#%%
from cellsmap.analyses.workflows.nuclear_based_features.support.plot_nuclei import plot_number_of_nuclei_per_fov, plot_number_of_nuclei_per_dataset
from cellsmap.analyses.workflows.nuclear_based_features.support.visualize_nuclei import visualize_nuclear_seg
from cellsmap.util.set_output import get_output_path
from pathlib import Path
import pandas as pd
from cellsmap.util import dataset_io
from cellsmap.util.manifest_io import get_nuclear_manifest

#%%
SAVE_DIR = get_output_path("nuclear_based_density_workflow/figs")
#%%
df = get_nuclear_manifest("20241016_20X")

#%%
dataframes = {file.stem: pd.read_parquet(file) for file in parquet_files}
print(f"Loaded {len(dataframes)} dataframes.")

#%%
df = dataframes['20241217_20X_nuclear_manifest']
df = df[(df["frame"] < 100) | ((df["frame"] > 300) & (df["frame"] < 420))]
dataframes['20241217_20X_nuclear_manifest'] = df

#%%
for df in list(dataframes.values()):
    plot_number_of_nuclei_per_fov(df, df.dataset.iloc[0], SAVE_DIR) 
plot_number_of_nuclei_per_dataset(list(dataframes.values()), SAVE_DIR)

# %%
for df in list(dataframes.values()):
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 0, SAVE_DIR)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 1, SAVE_DIR)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 2, SAVE_DIR)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 3, SAVE_DIR)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 4, SAVE_DIR)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 5, SAVE_DIR)

#%%
fmsid = "a90e321dff584e7f95e0c5e81122c751"
#%%
from aicsfiles import fms, FileLevelMetadataKeys
annotations = {
   FileLevelMetadataKeys.FILE_ID.value: fmsid
}
record = fms.find(annotations=annotations)
#%%
print(record)
# %%
recordpath = record.path