# %%
from cellsmap.util.manifest_io import get_manifest
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.immunofluorescence import filter, plot

output_dir = get_output_path("immunofluorescence_analysis/SMAD1")
# %%
IF_SMAD_DATASETS = [
    "20250509_20X_IF2",
    "20250509_20X_IF3",
    "20250509_20X_IF12",
    "20250509_20X_IF5",
    "20250509_20X_IF7",
    "20250509_20X_IF1",
    "20250509_20X_IF9",
]

# 1, 2, 12 are higher density and fixed on a different date
# 5, 7, 1, 9 are lower density and fixed on the same date

flow_rates = {
    "20250509_20X_IF1": 20.8,
    "20250509_20X_IF2": 0,
    "20250509_20X_IF3": 0,
    "20250509_20X_IF5": 5.98,
    "20250509_20X_IF7": 10.96,
    "20250509_20X_IF9": 23.67,
    "20250509_20X_IF12": 5.82,
}

# %%
if_manifest = get_manifest(IF_SMAD_DATASETS, "immunofluorescence_manifest_fmsid")
if_manifest = filter.filter_small_objects(if_manifest)
if_manifest = filter.filter_edge_objects(if_manifest)
if_manifest["SMAD1_norm_NucViolet_mean_sum_proj"] = (
    if_manifest["SMAD1_mean_sum_proj"] / if_manifest["NucViolet_mean_sum_proj"]
)
if_manifest["SMAD1_norm_area_mean_sum_proj"] = (
    if_manifest["SMAD1_mean_sum_proj"] / if_manifest["area"]
)
if_manifest = if_manifest[if_manifest["SMAD1_norm_NucViolet_mean_sum_proj"] < 1.0]
# %%
PLOT_FEAT = "SMAD1_mean_sum_proj"
xlim = 30000
ylim = 0.00035
# %%
for dataset in IF_SMAD_DATASETS:
    df = if_manifest[if_manifest["dataset"] == dataset]
    plot.plot_channel_intensity_histograms(
        df,
        if_manifest,
        ["NucViolet_mean_sum_proj", "SMAD1_mean_sum_proj", PLOT_FEAT],
        dataset,
        positions=[0, 1],
    )

plot.feature_density(
    df_all=if_manifest,
    dataset_name_list=IF_SMAD_DATASETS,
    feature=PLOT_FEAT,
    save_dir=output_dir,
    xlim=xlim,
    ylim=ylim,
    pool_positions=True,
)

plot.feature_density(
    df_all=if_manifest,
    dataset_name_list=IF_SMAD_DATASETS,
    feature=PLOT_FEAT,
    save_dir=output_dir,
    xlim=xlim,
    ylim=ylim,
    pool_positions=True,
    per_dataset=True,
)

# %%
plot.feature_boxplot_vs_flowrate(
    df_all=if_manifest, dataset_name_list=IF_SMAD_DATASETS, feature=PLOT_FEAT, save_dir=output_dir
)

plot.feature_boxplot_vs_sample_size(
    df_all=if_manifest, dataset_name_list=IF_SMAD_DATASETS, feature=PLOT_FEAT, save_dir=output_dir
)


# %%
plot.feature_scatter_vs_flowrate(
    df_all=if_manifest,
    dataset_name_list=IF_SMAD_DATASETS,
    feature=PLOT_FEAT,
    save_dir=output_dir,
    by_flowrate=False,
)
plot.feature_scatter_vs_flowrate(
    df_all=if_manifest,
    dataset_name_list=IF_SMAD_DATASETS,
    feature=PLOT_FEAT,
    save_dir=output_dir,
    by_flowrate=True,
)
# %%
