#%%
from cellsmap.util import dataset_io, manifest_io
#%%
dataset = "20241120_20X"
#%%
df = manifest_io.get_nuclear_manifest(dataset)
# %%
# Add an object ID where each object at each timepoint gets a unique ID, starting at 0 for each position
df["object_id"] = (
    df.groupby("position", group_keys=False)
    .apply(lambda group: group.groupby(["frame", "nuclear_label"]).ngroup())
)

# %% temp track
df["track_id"] = df["object_id"]
# %%
