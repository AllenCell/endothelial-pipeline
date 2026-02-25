# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# %%
# datasets = get_datasets_in_collection("diffae_model_training")
datasets = ["20250611_20X", "20250618_20X", "20250813_20X", "20250319_20X"]

# %% Load lda weights to apply LDA projection to original dataframe
lda_dataframe_manifest = load_dataframe_manifest("lda_weights")
lda_location = get_dataframe_location_for_dataset(lda_dataframe_manifest, "80_pcs")
df_lda = load_dataframe(lda_location)

# %% Load diffae features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)

# %% Load optical flow features
dataframe_manifest_optical_flow = load_dataframe_manifest("optical_flow")
dataframe_manifest_optical_flow_new = load_dataframe_manifest("optical_flow_new")


# %% Combine and calculate features
df_proj_full_list = []
df_proj_full_list_new = []

for dataset_name in datasets:
    print(f"Processing dataset: {dataset_name}")
    # Get PCS and LDA features
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

    # Get optical flow features
    optical_flow_location = get_dataframe_location_for_dataset(
        dataframe_manifest_optical_flow, dataset_name
    )
    df_optical_flow = load_dataframe(optical_flow_location)

    optical_flow_location_new = get_dataframe_location_for_dataset(
        dataframe_manifest_optical_flow_new, dataset_name
    )
    df_optical_flow_new = load_dataframe(optical_flow_location_new)

    # merge the two dataframes on the dataset, position, frame_number, start_x, start_y columns
    df_proj_full_old = df_proj_full.merge(
        df_optical_flow,
        on=["dataset", "position", "frame_number", "start_x", "start_y"],
        how="inner",
        suffixes=("", "_optical_flow"),
    )

    df_proj_full_new = df_proj_full.merge(
        df_optical_flow_new,
        on=["dataset", "position", "frame_number", "start_x", "start_y"],
        how="inner",
        suffixes=("", "_optical_flow_new"),
    )

    df_proj_full_list.append(df_proj_full_old)
    df_proj_full_list_new.append(df_proj_full_new)

# %% concatenate
df = pd.concat(df_proj_full_list, ignore_index=True)
# %%
df_new = pd.concat(df_proj_full_list_new, ignore_index=True)


# %%
def plot_lda_vs_optical_flow(
    df,
    features,
    optical_flow_features,
    color_by_dataset=True,
    point_alpha=0.25,
    figsize=(24, 2.5 * 9),
    max_points=10000,
):
    # Drop rows with NaNs in relevant columns
    df = df.dropna(subset=features + optical_flow_features + ["dataset"])
    rng = np.random.default_rng(42)
    datasets_used = df["dataset"].unique()
    n_rows, n_cols = len(optical_flow_features), len(features)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex="col", sharey="row")
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape((n_rows, n_cols))

    if color_by_dataset:
        colors = plt.cm.tab10(np.arange(len(datasets_used)))
    else:
        colors = ["tab:blue"] * len(datasets_used)

    df_np = df.copy()
    for col in features + optical_flow_features + ["dataset"]:
        if col not in df_np:
            continue
        df_np[col] = df_np[col].to_numpy()

    # Compute N for each dataset for legend
    legend_labels = []
    for i, dataset in enumerate(datasets_used):
        mask = df["dataset"] == dataset
        n_points = np.sum(mask)
        legend_labels.append(f"{dataset} (N={n_points})")

    for row, of_feature in enumerate(optical_flow_features):
        of_data = df_np[of_feature]
        for col, feature in enumerate(features):
            ax = axes[row, col]
            feat_data = df_np[feature]
            for i, dataset in enumerate(datasets_used):
                mask = df["dataset"] == dataset
                x_full = feat_data[mask]
                y_full = of_data[mask]
                x, y = x_full, y_full
                if max_points is not None and len(x) > max_points:
                    idx = rng.choice(len(x), max_points, replace=False)
                    x = x.iloc[idx]
                    y = y.iloc[idx]
                ax.scatter(
                    x,
                    y,
                    alpha=point_alpha,
                    color=colors[i],
                    label=legend_labels[i] if (row == 0 and col == 0) else None,  # Only label once
                    rasterized=True,
                )
            if row == 0:
                ax.set_title(feature)
            if col == 0:
                ax.set_ylabel(of_feature)
            if len(x_full) > 1:
                corr_coef = np.corrcoef(x_full, y_full)[0, 1]
                ax.annotate(
                    f"r={corr_coef:.2f}", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10
                )
            ax.grid(True)

    # Add legend to the first axis with handles
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=10, frameon=True)
            break

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %%
features = ["LDA", "LDA_SP_2", "LDA_SP_3", "LDA_SP_4", "pc_1", "pc_2", "pc_3"]
datasets_used = ["20250618_20X", "20250611_20X"]

# for dt in range(1, 11):
dt = 1
optical_flow_features = [
    f"optical_flow_mean_speed_dt{dt}",
    f"optical_flow_median_speed_dt{dt}",
    f"optical_flow_std_speed_dt{dt}",
    f"optical_flow_mean_angle_dt{dt}",
    f"optical_flow_angle_std_dt{dt}",
    f"optical_flow_mean_u_dt{dt}",
    f"optical_flow_mean_v_dt{dt}",
    f"optical_flow_std_u_dt{dt}",
    f"optical_flow_std_v_dt{dt}",
]

df_sub_new = df_new[df_new["dataset"].isin(datasets_used)]
plot_lda_vs_optical_flow(df_sub_new, features, optical_flow_features, color_by_dataset=True)
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
df_sub = df[df["dataset"].isin(datasets_used)]
plot_lda_vs_optical_flow(df_sub, features, optical_flow_features, color_by_dataset=True)


# %% Load classic features

# dataframe_manifest_classic = load_dataframe_manifest("test_live_merged_seg_features")
# dataframe_cell_centric_diffae = load_dataframe_manifest("pc_diffae_tracked_seg_features")

# # for dataset_name in datasets:
# dataset_name = datasets[0]
# dataframe_location_classic = get_dataframe_location_for_dataset(dataframe_manifest_classic, dataset_name)
# df_classic_delayed = load_dataframe(dataframe_location_classic, delay=True)

# dataframe_location_cell_centered = get_dataframe_location_for_dataset(dataframe_cell_centric_diffae, dataset_name)
# df_cell_centered = load_dataframe(dataframe_location_cell_centered)

# cols_to_compute = list(
#         set(
#             SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
#             + SEGMENTATION_FEATURE_COLUMNS["filters"]
#         )
#     )
# df_classic = df_classic_delayed[cols_to_compute].compute()
# df_classic = df_classic[df_classic.is_included]
# df_classic = calculate_derived_data_dynamics_dependent(df_classic, compute_per_crop_metrics=True)

# df_merged = pd.merge(
#     df_classic,
#     df_cell_centered,
#     left_on=["dataset_name", "position", "frame_number", "label", "track_id"],
#     right_on=["dataset_name", "position", "image_index", "label", "track_id"],
#     how="inner"
# )
