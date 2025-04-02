#%%
from cellsmap.analyses.workflows.density.support.plot_nuclei import plot_number_of_nuclei_per_fov, plot_number_of_nuclei_per_dataset
from cellsmap.analyses.workflows.density.support.visualize_nuclei import visualize_nuclear_seg
import pandas as pd
#%%
# df = create_nuclear_manifest('20241016_20X')

PATH_PREFIX = '/allen/aics/users/chantelle.leveille/repos/cellsmap/results/nuclear_seg_manifests/'

# for parquet files in PATH_PREFIX, load them into dataframes




df1016 = pd.read_parquet(f"{PATH_PREFIX}20241016_20X_nuclear_manifest.parquet")
df1022 = pd.read_parquet(f"{PATH_PREFIX}20241022_20X_mito_nuclear_manifest.parquet")
df1217 = pd.read_parquet(f"{PATH_PREFIX}20241217_20X_nuclear_manifest.parquet")
df0224 = pd.read_parquet(f"{PATH_PREFIX}20250224_20X_nuclear_manifest.parquet")

#%%
for df in [df1016, df1022, df1217, df0224]:
    plot_number_of_nuclei_per_fov(df, df.dataset.iloc[0]) 
plot_number_of_nuclei_per_dataset([df1016, df1022, df1217, df0224])
# %%
visualize_nuclear_seg(df, '20241016_20X', 0, 0)

# %%
for frame in [100,125,150,175,200,225,250,300]:
    visualize_nuclear_seg(df, '20241016_20X', frame, 3)
# %%
for pos in range(6):
    visualize_nuclear_seg(df, '20241016_20X', 0, pos)

