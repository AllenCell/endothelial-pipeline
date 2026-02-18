# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
pc_columns_to_keep = DIFFAE_PC_COLUMN_NAMES[:80]
other_cols_to_keep = ["polar_r", "polar_theta", "rho"]

# %%
annotation_path = "//allen/aics/users/chantelle.leveille/annotations"
mixed_mig_files = [
    {
        "dataset_name": "20250319_20X",
        "position": 2,
        "fname": "mixed_mig/12.2 shear stress 20250319_20X_P2-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 3,
        "fname": "mixed_mig/12.2 shear stress 20250319_20X_P3-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 0,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P0-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 1,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 4,
        "fname": "mixed_mig/14.65 shear stress 20250813_20X_P4-annotations.csv",
    },
]

coherent_mig_files = [
    {
        "dataset_name": "20250319_20X",
        "position": 0,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P0-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 2,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P2-annotations.csv",
    },
    {
        "dataset_name": "20250319_20X",
        "position": 5,
        "fname": "coherent_mig/12.2 shear stress 20250319_20X_P5-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 1,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P1-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 3,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P3-annotations.csv",
    },
    {
        "dataset_name": "20250813_20X",
        "position": 5,
        "fname": "coherent_mig/14.65 shear stress 20250813_20X_P5-annotations.csv",
    },
]

# %%
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)
# %%
df_mig_list = []
for file_info in mixed_mig_files + coherent_mig_files:
    dataset_name = file_info["dataset_name"]

    df = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca, filter_dataframe=False
    )

    # df = df[pc_columns_to_keep + other_cols_to_keep]

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
    # df_sub = df[df["position"] == file_info["position"]]
    merged = df.merge(
        pairs_df, left_on=["crop_index", "frame_number"], right_on=["Track", "Frame"], how="inner"
    )
    merged["coherent_migration"] = file_info in coherent_mig_files
    df_mig_list.append(merged)

    assert len(merged) == len(
        df_annotation
    ), f"Expected {len(df_annotation)} rows after merge, got {len(merged)} for file {fname}"

df_mig = pd.concat(df_mig_list, ignore_index=True)

print(df_mig.head())


# %%
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


# %%


def rank_features_and_plot_histograms(df, features_to_rank, label_column="coherent_migration"):

    ranking = compute_separation_power(df[features_to_rank], df["coherent_migration"])

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

    n_coherent_migration = df[label_column].sum()
    n_no_coherent_migration = len(df) - n_coherent_migration
    legend_labels = [
        f"Coherent Migration (N={n_coherent_migration})",
        f"Mixed Migration (N={n_no_coherent_migration})",
    ]
    fig.legend(legend_labels, loc="upper right")

    plt.tight_layout()
    plt.show()
    fig_savedir = get_output_path("find_coherent_mig")
    save_plot_to_path(fig, fig_savedir, "find_coherent_mig_histograms.png")
    plt.close()


# %%
rank_features_and_plot_histograms(
    df_mig,
    features_to_rank=pc_columns_to_keep + other_cols_to_keep,
    label_column="coherent_migration",
)

# %%
