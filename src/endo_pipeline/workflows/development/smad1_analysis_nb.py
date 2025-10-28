# %%
import numpy as np
import pandas as pd

from endo_pipeline import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.immunofluorescence import filter, plot
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

# %%
DESCRIPTION = "Analyze SMAD1 intensity distributions under shear stress conditions."

TAGS = ["immunofluorescence"]


# %% Load Data and add info to dataframe
output_dir = get_output_path("SMAD1")
smad1_datasets = get_datasets_in_collection("smad1")
if_df_manifest = load_dataframe_manifest("immunofluorescence")

df_smad1_list = []
for dataset_name in smad1_datasets:
    dataset_config = load_dataset_config(dataset_name)
    df_location = get_dataframe_location_for_dataset(if_df_manifest, dataset_name)
    df_dataset = load_dataframe(df_location)

    shear_regime = "_to_".join([shear.value for shear in dataset_config.shear_stress_regime])
    df_dataset["shear_stress_regime"] = shear_regime

    shear_stress_list = [condition.shear_stress for condition in dataset_config.flow_conditions]
    df_dataset["shear_stress_1"] = shear_stress_list[0]
    df_dataset["shear_stress_2"] = shear_stress_list[1] if len(shear_stress_list) > 1 else np.nan

    durations = [condition.stop - condition.start for condition in dataset_config.flow_conditions]
    duration_1 = durations[0]
    duration_2 = durations[1] if len(durations) > 1 else np.nan

    df_dataset["duration_at_ss_1_hr"] = duration_1 * 5 / 60  # convert to hrs
    df_dataset["duration_at_ss_2_hr"] = duration_2 * 5 / 60  # convert to hrs

    df_smad1_list.append(df_dataset)

df = pd.concat(df_smad1_list, ignore_index=True)

# %% Filter and preprocess features for immunofluorescence analysis.
df = filter.filter_small_objects(df)
df = filter.filter_edge_objects(df)
df["SMAD1_norm_NucViolet_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]
df["SMAD1_norm_area_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["area"]
df = df[df["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]

# %%
all_dataset_list = df["dataset"].unique().tolist()

if DEMO_MODE:
    all_dataset_list = all_dataset_list[:2]

# %% Plot distributions of SMAD1 intensity features
PLOT_FEAT = "SMAD1_mean_sum_proj"
xlim = 30000
ylim = 0.00035
for dataset in all_dataset_list:
    df_dataset = df[df["dataset"] == dataset]
    plot.plot_channel_intensity_histograms(
        df_dataset,
        df,
        ["NucViolet_mean_sum_proj", "SMAD1_mean_sum_proj", PLOT_FEAT],
        dataset,
        positions=df_dataset["position"].unique().tolist(),
        save_dir=output_dir,
    )
# %%
plot.feature_density(
    df_all=df,
    dataset_name_list=all_dataset_list,
    feature=PLOT_FEAT,
    feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
    save_dir=output_dir,
    xlim=xlim,
    ylim=ylim,
    pool_positions=True,
)
# %%
plot.feature_density(
    df_all=df,
    dataset_name_list=all_dataset_list,
    feature=PLOT_FEAT,
    feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
    save_dir=output_dir,
    xlim=xlim,
    ylim=ylim,
    pool_positions=True,
    per_dataset=True,
)
# %%
