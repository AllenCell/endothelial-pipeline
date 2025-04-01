#%%
from cellsmap.analyses.workflows.density.support.get_nuclear_manifest import create_nuclear_manifest
from cellsmap.analyses.workflows.density.support.plot_nuclei import plot_number_of_nuclei_per_fov
from cellsmap.analyses.workflows.density.support.visualize_nuclei import visualize_nuclear_seg
#%%
import pandas as pd
#%%
# df = create_nuclear_manifest('20241016_20X')

PATH_PREFIX = '/allen/aics/users/chantelle.leveille/repos/cellsmap/results/nuclear_seg_manifests/'

df = pd.read_parquet(f"{PATH_PREFIX}20241016_20X_nuclear_manifest.parquet")
df.sort_values(by=['frame'], ascending=True, inplace=True)
#%%
plot_number_of_nuclei_per_fov(df, '20241016_20X') 

# %%
visualize_nuclear_seg(df, '20241016_20X', 0, 0)
# %%
for frame in [100,125,150,175,200,225,250,300]:
    visualize_nuclear_seg(df, '20241016_20X', frame, 3)
# %%
for pos in range(6):
    visualize_nuclear_seg(df, '20241016_20X', 0, pos)
# %%
