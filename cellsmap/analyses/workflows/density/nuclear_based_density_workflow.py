#%%
from cellsmap.analyses.workflows.density.support.plot_nuclei import plot_number_of_nuclei_per_fov, plot_number_of_nuclei_per_dataset
from cellsmap.analyses.workflows.density.support.visualize_nuclei import visualize_nuclear_seg
from pathlib import Path
import pandas as pd
#%%
PATH_PREFIX = '/allen/aics/users/chantelle.leveille/repos/cellsmap/results/nuclear_seg_manifests/'
parquet_files = [f for f in Path(PATH_PREFIX).glob('*.parquet')]
dataframes = {file.stem: pd.read_parquet(file) for file in parquet_files}
print(f"Loaded {len(dataframes)} dataframes.")

#%%
df = dataframes['20241217_20X_nuclear_manifest']
df = df[(df["frame"] < 100) | ((df["frame"] > 300) & (df["frame"] < 420))]
dataframes['20241217_20X_nuclear_manifest'] = df

#%%
for df in dataframes:
    plot_number_of_nuclei_per_fov(df, df.dataset.iloc[0]) 
plot_number_of_nuclei_per_dataset(list(dataframes.values()))

# %%
for df in list(dataframes.values()):
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 0)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 1)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 2)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 3)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 4)
    visualize_nuclear_seg(df, df.dataset.iloc[0], 0, 5)
