# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.immunofluorescence import filter, plot
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

# %%
output_dir = get_output_path("SMAD1")
# %%
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
    # Assign to variables with fallback to NaN
    shear_stress_value_1 = shear_stress_list[0]
    shear_stress_value_2 = shear_stress_list[1] if len(shear_stress_list) > 1 else np.nan

    df_dataset["shear_stress_1"] = shear_stress_value_1
    df_dataset["shear_stress_2"] = shear_stress_value_2

    durations = [condition.stop - condition.start for condition in dataset_config.flow_conditions]
    duration_1 = durations[0]
    duration_2 = durations[1] if len(durations) > 1 else np.nan

    df_dataset["duration_at_ss_1_hr"] = duration_1 * 5 / 60  # convert to hrs
    df_dataset["duration_at_ss_2_hr"] = duration_2 * 5 / 60  # convert to hrs

    df_dataset["num_nuclei"] = len(df_dataset)

    df_smad1_list.append(df_dataset)
# %%
df = pd.concat(df_smad1_list, ignore_index=True)


def if_feature_preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    df = filter.filter_small_objects(df)
    df = filter.filter_edge_objects(df)
    df["SMAD1_norm_NucViolet_mean_sum_proj"] = (
        df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]
    )
    df["SMAD1_norm_area_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["area"]
    df = df[df["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]
    return df


df = if_feature_preprocessing(df)

# %%
plt.figure(figsize=(11, 8))

datasets = df["dataset"].unique()
cmap = colormaps.get_cmap("tab20")
colors = {dataset: cmap(i / len(datasets)) for i, dataset in enumerate(datasets)}

for dataset, df_dataset in df.groupby("dataset"):
    color = colors[dataset]

    # Determine marker style based on whether the dataset name ends with a digit
    marker_style = "d" if dataset[-1].isdigit() else "o"

    num_nuclei = df_dataset.num_nuclei.iloc[0]
    shear_stress_value_1 = df_dataset.shear_stress_1.iloc[0]
    shear_stress_value_2 = df_dataset.shear_stress_2.iloc[0]
    shear_regime = df_dataset.shear_stress_regime.iloc[0]
    duration_1 = df_dataset.duration_at_ss_1_hr.iloc[0]
    duration_2 = df_dataset.duration_at_ss_2_hr.iloc[0]

    final_shear_stress = (
        shear_stress_value_2 if not np.isnan(shear_stress_value_2) else shear_stress_value_1
    )
    data_label = f"{dataset}\n{shear_regime} shear stress"
    duration_label1 = f"{duration_1:.2f} hr @ {shear_stress_value_1} dyn/cm²\n"
    duration_label2 = (
        f"{duration_2:.2f} hr @ {shear_stress_value_2} dyn/cm²\n"
        if not np.isnan(shear_stress_value_2)
        else ""
    )

    plt.scatter(
        final_shear_stress,
        num_nuclei,
        marker=marker_style,
        s=100,
        color=color,
        label=f"{data_label}\n{duration_label1} {duration_label2}",
    )

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=10)
plt.xlabel("Final Shear Stress (dyn/cm²)")
plt.ylabel("Number of Nuclei Detected")
plt.tight_layout()

# %%
PLOT_FEAT = "SMAD1_mean_sum_proj"
xlim = 30000
ylim = 0.00035
for dataset, df_dataset in df.groupby("dataset"):
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
    dataset_name_list=df["dataset"].unique().tolist(),
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
    dataset_name_list=df["dataset"].unique().tolist(),
    feature=PLOT_FEAT,
    feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
    save_dir=output_dir,
    xlim=xlim,
    ylim=ylim,
    pool_positions=True,
    per_dataset=True,
)

# %%
