# %%
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.io import load_dataframe
from src.endo_pipeline.library.analyze.nuclear_based_features import nuclear_plots
from src.endo_pipeline.manifests import load_dataframe_manifest

# %%
SAVE_DIR = get_output_path("nuclear_based_density_workflow/figs")

# %% Load datasets
manifest = load_dataframe_manifest("nuclear_segmentation_features")
dataset_list = list(manifest.dataframe_locations.keys())
dataframes = {
    dataset_name: load_dataframe(dataframe_location)
    for dataset_name, dataframe_location in manifest.dataframe_locations.items()
}
print(f"Loaded {len(dataframes)} datasets.")
print(f"Datasets: {list(dataframes.keys())}")

# %% Plot the number of nuclei per fov and per dataset
for df in list(dataframes.values()):
    nuclear_plots.plot_number_of_nuclei_per_fov(df, df.dataset.iloc[0], SAVE_DIR)
# %%
nuclear_plots.plot_number_of_nuclei_per_dataset(list(dataframes.values()), SAVE_DIR)
# %%
nuclear_plots.plot_flow_over_time_per_dataset(dataset_list, SAVE_DIR)
# %%
