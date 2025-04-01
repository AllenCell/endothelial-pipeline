#%%
from cellsmap.analyses.workflows.density.support.get_nuclear_manifest import create_nuclear_manifest
from cellsmap.analyses.workflows.density.support.plot_nuclei import plot_number_of_nuclei_per_fov
from cellsmap.analyses.workflows.density.support.visualize_nuclei import visualize_nuclear_seg

#%%
df = create_nuclear_manifest('20241016_20X')

#%%
plot_number_of_nuclei_per_fov(df, '20241016_20X') 