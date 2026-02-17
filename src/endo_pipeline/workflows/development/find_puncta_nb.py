# %%

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.workflow_defaults import DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME

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
for file_info in has_puncta_files + no_puncta_files:
    dataset_name = file_info["dataset_name"]
    cell_centric_feats_manifest = load_dataframe_manifest(
        DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME
    )
    cell_centric_feats_location = get_dataframe_location_for_dataset(
        cell_centric_feats_manifest, dataset_name
    )
    df = load_dataframe(cell_centric_feats_location, delay=True)
    df = df.reset_index(drop=True)

    df = df[pc_columns_to_keep + other_cols_to_keep].compute()

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
    df_sub = df[df["position"] == file_info["position"]]
    merged = df_sub.merge(
        pairs_df, left_on=["track_id", "image_index"], right_on=["Track", "Frame"], how="inner"
    )
    merged["has_puncta"] = file_info in has_puncta_files
    df_puncta_list.append(merged)

df_puncta = pd.concat(df_puncta_list, ignore_index=True)

print(df_puncta.head())


def compute_separation_power(X, y, verbose=True):
    # Assuming 'X' is your (M samples x N features) matrix
    # Assuming 'y' is your binary label vector (0s and 1s)
    ranking = []
    for feature_name in X.columns:
        # Calculate AUC
        score = roc_auc_score(y, X[feature_name])
        # We care about "Separation Power", so 0.1 is just as good as 0.9.
        # We calculate 'power' as distance from 0.5 (randomness)
        separation_power = 2.0 * abs(score - 0.5)

        ranking.append({"feature": feature_name, "auc": score, "power": separation_power})

    print("Top features by separation power:")
    ranking_sorted = sorted(ranking, key=lambda x: x["power"], reverse=True)
    if verbose:
        for item in ranking_sorted[:10]:  # Print top 10 features
            print(f"{item['feature']}: AUC={item['auc']:.3f}, Power={item['power']:.3f}")
    return ranking_sorted


def rank_features_and_plot_histograms(df, features_to_rank, label_column="has_puncta"):

    ranking = compute_separation_power(df[features_to_rank], df["has_puncta"])

    n_pcs = len(features_to_rank)
    ncols = 10
    nrows = (n_pcs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i, item in enumerate(ranking):
        col = item["feature"]
        x_min = df[col].min()
        x_max = df[col].max()
        for label in df[label_column].unique():
            subset = df[df[label_column] == label]
            axes[i].hist(subset[col], bins=30, range=(x_min, x_max), alpha=0.75, label=f"{label}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"{col}, Power: {item['power']:.3f}")

    n_puncta = df[label_column].sum()
    n_no_puncta = len(df) - n_puncta
    legend_labels = [f"With Puncta (N={n_puncta})", f"Without Puncta (N={n_no_puncta})"]
    fig.legend(legend_labels, loc="upper right")

    plt.tight_layout()
    plt.show()
    fig_savedir = get_output_path("find_puncta")
    save_plot_to_path(fig, fig_savedir, "find_puncta_histograms.png")
    plt.close()


rank_features_and_plot_histograms(
    df_puncta, features_to_rank=pc_columns_to_keep, label_column="has_puncta"
)

# %%
