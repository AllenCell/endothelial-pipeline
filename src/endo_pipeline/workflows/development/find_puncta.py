# %%
import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.library.analyze.integration.track_integration import (
    load_pc_diffae_liveseg_feats_merged_table,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES

# %%
pc_columns_to_keep = DIFFAE_PC_COLUMN_NAMES[:80]
other_cols_to_keep = ["track_id", "image_index", "position"]

# %%
annotation_path = "//allen/aics/users/chantelle.leveille/annotations"
has_puncta_files = [
    {
        "dataset_name": "20250618_20X",
        "position": 1,
        "fname": "5.82 shear stress 20250618_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250604_20X",
        "position": 2,
        "fname": "12.32 shear stress 20250604_20X_P2-annotations.csv",
    },
    {
        "dataset_name": "20250604_20X",
        "position": 4,
        "fname": "12.32 shear stress 20250604_20X_P4-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 0,
        "fname": "14.65 shear stress 20250813_20X_P0-annotations.csv",
    },
]

no_puncta_files = [
    {
        "dataset_name": "20250618_20X",
        "position": 1,
        "fname": "5.82 shear stress 20250618_20X_P1-annotations (1).csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 1,
        "fname": "12.2 shear stress 20250319_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250326_20X",
        "position": 1,
        "fname": "15.74 shear stress 20250326_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250326_20X",
        "position": 4,
        "fname": "15.74 shear stress 20250326_20X_P4-annotations.csv",
    },
]

# %%
df_puncta_list = []
for file_info in has_puncta_files:
    dataset_name = file_info["dataset_name"]
    df = load_pc_diffae_liveseg_feats_merged_table(dataset_name)
    df = df[pc_columns_to_keep + other_cols_to_keep].compute()

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
    df_sub = df[df["position"] == file_info["position"]]
    merged = df_sub.merge(
        pairs_df, left_on=["track_id", "image_index"], right_on=["Track", "Frame"], how="inner"
    )
    df_puncta_list.append(merged)

df_puncta = pd.concat(df_puncta_list, ignore_index=True)

# %%
df_no_puncta_list = []
for file_info in no_puncta_files:
    dataset_name = file_info["dataset_name"]
    df = load_pc_diffae_liveseg_feats_merged_table(dataset_name)
    df = df[pc_columns_to_keep + other_cols_to_keep].compute()

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
    df_sub = df[df["position"] == file_info["position"]]
    merged = df_sub.merge(
        pairs_df, left_on=["track_id", "image_index"], right_on=["Track", "Frame"], how="inner"
    )
    df_no_puncta_list.append(merged)

df_no_puncta = pd.concat(df_no_puncta_list, ignore_index=True)


# %%
# create another version where you only include pc 1 and 2 that are negative in the plot
pc_columns_to_adjust = pc_columns_to_keep[:2]  # Adjust this if you want to include more PCs
mask = (df_puncta[pc_columns_to_adjust] < 1).all(axis=1)
df_puncta_subset = df_puncta[mask]

mask = (df_no_puncta[pc_columns_to_adjust] < 1).all(axis=1)
df_no_puncta_subset = df_no_puncta[mask]


# %%
def plot_pc_histograms(df_puncta, df_no_puncta, pc_columns_to_keep):
    n_pcs = len(pc_columns_to_keep)
    ncols = 10
    nrows = (n_pcs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i, pc_axis in enumerate(pc_columns_to_keep):
        x_min = min(df_puncta[pc_axis].min(), df_no_puncta[pc_axis].min())
        x_max = max(df_puncta[pc_axis].max(), df_no_puncta[pc_axis].max())
        if i == 0:
            axes[i].hist(
                df_puncta[pc_axis],
                bins=30,
                range=(x_min, x_max),
                alpha=0.75,
                color="blue",
                label="",
            )
            axes[i].hist(
                df_no_puncta[pc_axis],
                bins=30,
                range=(x_min, x_max),
                alpha=0.75,
                color="orange",
                label="",
            )
        else:
            axes[i].hist(
                df_puncta[pc_axis], bins=30, range=(x_min, x_max), alpha=0.75, color="blue"
            )
            axes[i].hist(
                df_no_puncta[pc_axis], bins=30, range=(x_min, x_max), alpha=0.75, color="orange"
            )
        axes[i].set_xlabel(f"PC {i+1}")
        axes[i].set_ylabel("Count")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a single legend for the whole figure
    n_puncta = len(df_puncta)
    n_no_puncta = len(df_no_puncta)
    legend_labels = [f"With Puncta (N={n_puncta})", f"Without Puncta (N={n_no_puncta})"]

    fig.legend(legend_labels, loc="upper right")

    plt.tight_layout()
    plt.show()
    plt.close()


# %%
print("All:")
plot_pc_histograms(df_puncta, df_no_puncta, pc_columns_to_keep)
# %%
print("Puncta subset:")
plot_pc_histograms(df_puncta_subset, df_no_puncta_subset, pc_columns_to_keep)
# %%
