# %%
import logging

import numpy as np
import pandas as pd

from endo_pipeline import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.immunofluorescence import filter, plot
from endo_pipeline.library.analyze.immunofluorescence.plot import if_dataset_contact_sheet
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

# %%
DESCRIPTION = "Analyze SMAD1 intensity distributions under shear stress conditions."

TAGS = ["immunofluorescence", "test_ready", "CPU_only"]

logger = logging.getLogger(__name__)
# %% Load Data and add info to dataframe
output_dir = get_output_path("SMAD1")
smad1_datasets = get_datasets_in_collection("smad1")
if_df_manifest = load_dataframe_manifest("immunofluorescence")

df_smad1_list = []
for dataset_name in smad1_datasets:
    dataset_config = load_dataset_config(dataset_name)

    df_location = get_dataframe_location_for_dataset(if_df_manifest, dataset_name)
    df_dataset = load_dataframe(df_location)

    df_dataset["date"] = dataset_name[:8]

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
df = filter.filter_img_center(df)
df["SMAD1_norm_NucViolet_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]
df["SMAD1_norm_area_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["area"]
df = df[df["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]

# %% Plot distributions of SMAD1 intensity features
PLOT_FEAT = "SMAD1_mean_sum_proj"
xlim = 35000.0
ylim = 0.0004
# %%
for dataset in smad1_datasets:
    df_dataset = df[df["dataset"] == dataset]
    plot.plot_channel_intensity_histograms(
        df_dataset,
        df,
        ["NucViolet_mean_sum_proj", "SMAD1_mean_sum_proj", PLOT_FEAT],
        dataset,
        positions=df_dataset["position"].unique().tolist(),
        save_dir=output_dir,
    )
    if DEMO_MODE:
        break

# %%
for date, df_date in df.groupby("date"):
    date_datasets = df_date["dataset"].unique().tolist()
    plot.feature_density(
        df_all=df_date,
        dataset_name_list=date_datasets,
        feature=PLOT_FEAT,
        feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
        save_dir=output_dir,
        xlim=xlim,
        ylim=ylim,
        pool_positions=True,
    )
    if DEMO_MODE:
        break
# %%
for date, df_date in df.groupby("date"):
    date_datasets = df_date["dataset"].unique().tolist()
    plot.feature_density(
        df_all=df_date,
        dataset_name_list=date_datasets,
        feature=PLOT_FEAT,
        feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
        save_dir=output_dir,
        xlim=xlim,
        ylim=ylim,
        pool_positions=True,
        per_dataset=True,
    )
    if DEMO_MODE:
        break

# %%
DATASET_GROUPS = {
    "20250509": [
        [
            "20250509_20X_IF3",
            "20250509_20X_IF5",
            "20250509_20X_IF7",
            "20250509_20X_IF9",
        ],  # 24 hr low density varied shear stress
        [
            "20250509_20X_IF2",
            "20250509_20X_IF12",
            "20250509_20X_IF1",
        ],  #  24 hr high density varied shear stress
    ],
    "20250522": [
        ["20250522_20X_IFH", "20250522_20X_IFJ"],  # 24 hr low density varied shear stress
        ["20250522_20X_IFI", "20250522_20X_IFG", "20250522_20X_IFN"],  # 48 hr varied shear stress
        [
            "20250522_20X_IFH",
            "20250522_20X_IFA",
            "20250522_20X_IFB",
            "20250522_20X_IFD",
            "20250522_20X_IFC",
            "20250522_20X_IFF",
            "20250522_20X_IFE",
            "20250522_20X_IFG",
        ],  # 24 low + X high over time
        [
            "20250522_20X_IFJ",
            "20250522_20X_IFK",
            "20250522_20X_IFL",
            "20250522_20X_IFM",
            "20250522_20X_IFN",
        ],  # 24 high + X low over time
    ],
    "20250929": [
        ["20250929_20X_IF0", "20250929_20X_IF2", "20250929_20X_IF9"],  # no shear stress over time
        ["20250929_20X_IF2", "20250929_20X_IF1", "20250929_20X_IF3"],  # 24 hr varied shear stress
        ["20250929_20X_IF9", "20250929_20X_IF8", "20250929_20X_IF10"],  # 48 hr varied shear stress
        [
            "20250929_20X_IF1",
            "20250929_20X_IF4",
            "20250929_20X_IF5",
            "20250929_20X_IF6",
            "20250929_20X_IF8",
        ],  # 24 hr low + X high over time
        [
            "20250929_20X_IF3",
            "20250929_20X_IF7",
            "20250929_20X_IF10",
        ],  # 24 hr high + X low over time
    ],
    "20251103": [
        ["20251103_20X_IF0", "20251103_20X_IF2"],  # no shear stress over time
        ["20251103_20X_IF2", "20251103_20X_IF3", "20251103_20X_IF1"],  # 24 hr varied shear stress
        ["20251103_20X_IF8", "20251103_20X_IF10"],
        [
            "20251103_20X_IF1",
            "20251103_20X_IF4",
            "20251103_20X_IF5",
            "20251103_20X_IF6",
            "20251103_20X_IF8",
        ],  # 24 hr high + X low over time
        [
            "20251103_20X_IF3",
            "20251103_20X_IF7",
            "20251103_20X_IF10",
        ],  # medium shear stress over time
    ],
}


# %%
for date, df_date in df.groupby("date"):
    group = DATASET_GROUPS[date]
    for subgroup in group:
        plot.stacked_feature_density(
            df_all=df_date,
            dataset_name_list=subgroup,
            feature=PLOT_FEAT,
            feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
            save_dir=output_dir,
            xlim=xlim,
        )
        plot.feature_density(
            df_all=df_date,
            dataset_name_list=subgroup,
            feature=PLOT_FEAT,
            feature_name="SMAD1 mean intensity of sum projection\nin nuclear mask",
            save_dir=output_dir,
            xlim=xlim,
            ylim=ylim,
            pool_positions=True,
        )
        if DEMO_MODE:
            break
    if DEMO_MODE:
        break

# %%
DEMO_MODE = True
for date, df_date in df.groupby("date"):
    group = DATASET_GROUPS[date]
    for subgroup in group:
        reversed_group = list(reversed(subgroup))
        if_dataset_contact_sheet(df_date, reversed_group, output_dir)
        if DEMO_MODE:
            break
    if DEMO_MODE:
        break
# %%
