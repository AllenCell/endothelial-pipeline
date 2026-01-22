# %%
import logging

import pandas as pd

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.immunofluorescence import filter, plot
from endo_pipeline.library.analyze.immunofluorescence.dataset_groupings import DATASET_GROUPS
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

    df_smad1_list.append(df_dataset)

df = pd.concat(df_smad1_list, ignore_index=True)

# %% Filter and preprocess features for immunofluorescence analysis.
df = filter.filter_small_objects(df)
df = filter.filter_img_center(df)
df["SMAD1_norm_NucViolet_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]
df["SMAD1_norm_area_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["area"]
df = df[df["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]

# %% Plot distributions of SMAD1 intensity features
plot_feat_cols = ["SMAD1_sum_sum_proj", "SMAD1_norm_area_sum_max_proj"]  # "SMAD1_mean_sum_proj"
plot_feat_names = [
    "Total SMAD1 intensity in nuclear mask volume",
    "Total SMAD1 intensity / N pixels \nin nuclear mask volume",
]  # "SMAD1 mean intensity of sum projection\nin nuclear mask"
date_list = df["date"].unique().tolist()

if DEMO_MODE:
    smad1_datasets = smad1_datasets[:1]
    plot_feat_cols = plot_feat_cols[:1]
    smad1_datasets = smad1_datasets[:1]
    date_list = date_list[:1]

# %%
for plot_feat, plot_feat_name in zip(plot_feat_cols, plot_feat_names, strict=False):
    xlim = df[plot_feat].quantile(0.99) * 1.1
    ylim = df[plot_feat].value_counts().max() + 10
    output_dir = get_output_path("SMAD1", "intensity_distribution_histograms", plot_feat)
    for dataset in smad1_datasets:
        df_dataset = df[df["dataset"] == dataset]
        plot.plot_channel_intensity_histograms(
            df_dataset,
            df,
            ["NucViolet_mean_sum_proj", "SMAD1_mean_sum_proj", plot_feat],
            dataset,
            positions=df_dataset["position"].unique().tolist(),
            save_dir=output_dir,
        )

# %%
for plot_feat, plot_feat_name in zip(plot_feat_cols, plot_feat_names, strict=False):
    xlim = df[plot_feat].quantile(0.99) * 1.1
    ylim = None
    for date in date_list:
        df_date = df[df["date"] == date]
        output_dir = get_output_path("SMAD1", "feature_density_by_date", plot_feat, str(date))
        date_datasets = df_date["dataset"].unique().tolist()
        plot.feature_density(
            df_all=df_date,
            dataset_name_list=date_datasets,
            feature=plot_feat,
            feature_name=plot_feat_name,
            save_dir=output_dir,
            xlim=xlim,
            ylim=ylim,
            pool_positions=True,
        )

# %%
for plot_feat, plot_feat_name in zip(plot_feat_cols, plot_feat_names, strict=False):
    xlim = df[plot_feat].quantile(0.99) * 1.1
    ylim = None
    for date in date_list:
        df_date = df[df["date"] == date]
        date_datasets = df_date["dataset"].unique().tolist()
        plot.feature_density(
            df_all=df_date,
            dataset_name_list=date_datasets,
            feature=plot_feat,
            feature_name=plot_feat_name,
            save_dir=output_dir,
            xlim=xlim,
            ylim=ylim,
            pool_positions=True,
            per_dataset=True,
        )


for plot_feat, plot_feat_name in zip(plot_feat_cols, plot_feat_names, strict=False):
    xlim = df[plot_feat].quantile(0.99) * 1.1
    ylim = None
    for date in date_list:
        df_date = df[df["date"] == date]
        group = DATASET_GROUPS[str(date)]
        for subgroup, group_name in group:
            output_dir = get_output_path(
                "SMAD1", "feature_density_by_date", plot_feat, str(date), group_name
            )
            plot.stacked_feature_density(
                df_all=df_date,
                dataset_name_list=subgroup,
                feature=plot_feat,
                feature_name=plot_feat_name,
                save_dir=output_dir,
                xlim=xlim,
            )
            plot.feature_density(
                df_all=df_date,
                dataset_name_list=subgroup,
                feature=plot_feat,
                feature_name=plot_feat_name,
                save_dir=output_dir,
                xlim=xlim,
                ylim=ylim,
                pool_positions=True,
            )

# %%
for date in date_list:
    df_date = df[df["date"] == date]
    group = DATASET_GROUPS[str(date)]
    for subgroup, group_name in group:
        output_dir = get_output_path("SMAD1", "contact_sheet", str(date), group_name)
        reversed_group = list(reversed(subgroup))
        if_dataset_contact_sheet(df_date, reversed_group, output_dir)
