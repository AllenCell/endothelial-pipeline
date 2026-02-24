# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.migration_pc.lda_analysis import apply_lda_projection
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %% Use saved lda weights to apply LDA projection to original dataframe
lda_dataframe_manifest = load_dataframe_manifest("lda_weights")
lda_location = get_dataframe_location_for_dataset(lda_dataframe_manifest, "80_pcs")
df_lda = load_dataframe(lda_location)

# %% diffae features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)

# %% optical flow features
dataframe_manifest_optical_flow = load_dataframe_manifest("optical_flow")

# %%
datasets = get_datasets_in_collection("diffae_model_training")

# %%
df_proj_full_list = []

for dataset_name in datasets:
    print(f"Processing dataset: {dataset_name}")
    df_dataset = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
    )

    df_proj_full = apply_lda_projection(
        df_dataset,
        features_in_lda_rank=df_lda["features"],
        lda_weights=df_lda["weights"],
        lda_intercept=df_lda["intercept"][0],
        sparse_axes=[2.0, 3.0, 4.0],
    )

    optical_flow_location = get_dataframe_location_for_dataset(
        dataframe_manifest_optical_flow, dataset_name
    )
    df_optical_flow = load_dataframe(optical_flow_location)

    # merge the two dataframes on the dataset, position, frame_number, start_x, start_y columns
    df_proj_full = df_proj_full.merge(
        df_optical_flow,
        on=["dataset", "position", "frame_number", "start_x", "start_y"],
        how="inner",
        suffixes=("", "_optical_flow"),
    )

    df_proj_full_list.append(df_proj_full)

# %% concatenate and drop rows with NaNs (ie last timepoint of movie)
df = pd.concat(df_proj_full_list, ignore_index=True)
df = df.dropna(subset=["optical_flow_angle_std"])


# %%
def plot_lda_vs_optical_flow(
    df,
    features,
    optical_flow_features,
    color_by_dataset=True,
    point_alpha=0.25,
    figsize=(24, 2.5 * 9),
):
    datasets_used = df["dataset"].unique()
    fig, axes = plt.subplots(
        len(optical_flow_features), len(features), figsize=figsize, sharex="col", sharey="row"
    )
    if color_by_dataset:
        colors = plt.cm.tab10(np.arange(len(datasets_used)))
    else:
        colors = ["tab:blue"] * len(datasets_used)

    for row, of_feature in enumerate(optical_flow_features):
        for col, feature in enumerate(features):
            ax = axes[row, col]
            for i, dataset in enumerate(datasets_used):
                mask = df["dataset"] == dataset
                ax.scatter(
                    df.loc[mask, feature],
                    df.loc[mask, of_feature],
                    alpha=point_alpha,
                    color=colors[i],
                    label=dataset if (row == 0 and col == len(features) - 1) else None,
                )
            if row == 0:
                ax.set_title(feature)
            if col == 0:
                ax.set_ylabel(of_feature)
            corr_coef = np.corrcoef(df[feature], df[of_feature])[0, 1]
            ax.annotate(f"r={corr_coef:.2f}", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10)
            ax.grid(True)

    # Always add dataset legend to top-right subplot
    handles, labels = axes[0, -1].get_legend_handles_labels()
    axes[0, -1].legend(handles, datasets_used, loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()


# %%
optical_flow_features = [
    "optical_flow_mean_speed",
    "optical_flow_median_speed",
    "optical_flow_std_speed",
    "optical_flow_mean_angle",
    "optical_flow_angle_std",
    "optical_flow_mean_u",
    "optical_flow_mean_v",
    "optical_flow_std_u",
    "optical_flow_std_v",
]

features = ["LDA", "LDA_SP_2", "LDA_SP_3", "LDA_SP_4", "pc_1", "pc_2", "pc_3"]

# %%
datasets_used = ["20250618_20X", "20250611_20X"]

plot_lda_vs_optical_flow(
    df[df["dataset"].isin(datasets_used)], features, optical_flow_features, color_by_dataset=True
)

# %%
datasets_used = ["20250319_20X", "20250813_20X"]

plot_lda_vs_optical_flow(
    df[df["dataset"].isin(datasets_used)], features, optical_flow_features, color_by_dataset=True
)
# %%
datasets_used = [
    "20250618_20X",
    "20250428_20X",
    "20250319_20X",
    "20250813_20X",
    "20250611_20X",
    # "20250818_20X", Remove no shear stress, not migrating like the rest
    "20250714_20X",
    "20250827_20X",
]

plot_lda_vs_optical_flow(
    df[df["dataset"].isin(datasets_used)], features, optical_flow_features, color_by_dataset=False
)
# %%
