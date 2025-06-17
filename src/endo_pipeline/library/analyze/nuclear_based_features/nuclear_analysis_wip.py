# %%
from cellsmap.util.manifest_io import get_nuclear_manifest, list_datasets_with_manifest
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.nuclear_based_features.plot_nuclei import (
    plot_flow_over_time_per_dataset,
    plot_number_of_nuclei_per_dataset,
    plot_number_of_nuclei_per_fov,
)

# %%
SAVE_DIR = get_output_path("nuclear_based_density_workflow/figs")

# %% Load datasets
dataset_list = list_datasets_with_manifest("nuclear_seg_manifest_fmsid")
dataframes = {
    dataset_name: get_nuclear_manifest(dataset_name) for dataset_name in dataset_list
}
print(f"Loaded {len(dataframes)} datasets.")
print(f"Datasets: {list(dataframes.keys())}")

# %% Plot the number of nuclei per fov and per dataset
for df in list(dataframes.values()):
    plot_number_of_nuclei_per_fov(df, df.dataset.iloc[0], SAVE_DIR)
# %%
plot_number_of_nuclei_per_dataset(list(dataframes.values()), SAVE_DIR)
# %%
plot_flow_over_time_per_dataset(dataset_list, SAVE_DIR)
# %%
