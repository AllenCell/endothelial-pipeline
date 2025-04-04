# %%
from cellsmap.util.manifest_io import list_datasets_with_manifest, get_nuclear_manifest
from cellsmap.analyses.workflows.nuclear_based_features.calculate.calculate_density import (
    get_nuclear_centroids,
)

# %%
dataset_list = list_datasets_with_manifest("nuclear_seg_manifest_fmsid")
print(dataset_list)
# %%

df = get_nuclear_manifest("20241022_20X_mito")
centroid_list = get_nuclear_centroids(
    df, dataset="20241022_20X_mito", position=0, frame=0
)

# %%
print(centroid_list)
# %%
