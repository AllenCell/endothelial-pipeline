# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
output_dir = get_output_path("find_coherent_mig")
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
    dataset_name: str = file_info["dataset_name"]

    df = get_dataframe_for_dynamics_workflows(
        dataset_name, dataframe_manifest, pca, filter_dataframe=False
    )

    fname = file_info["fname"]
    df_annotation = pd.read_csv(f"{annotation_path}/{fname}")
    pairs_df = df_annotation[["Track", "Frame"]]
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
def rank_features_and_plot_histograms(
    df,
    features_to_rank,
    label_column="coherent_migration",
    fname="find_coherent_mig_histograms.png",
    output_dir=output_dir,
):

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
        for label in [True, False]:
            subset = df[df[label_column] == label]
            if label is True:
                label_name = "Coherent Migration"
            if label is False:
                label_name = "Mixed Migration"
            axes[i].hist(
                subset[col], bins=30, range=(x_min, x_max), alpha=0.75, label=f"{label_name}"
            )
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
    save_plot_to_path(fig, output_dir, fname)
    plt.close()


def run_lda_feature_ranking(
    df_mig: pd.DataFrame,
    features_to_rank: list,
    output_dir: Path,
    fname_suffix: str = "",
):
    features_to_rank = [
        col.value if hasattr(col, "value") else str(col) for col in features_to_rank
    ]
    df_features = df_mig[features_to_rank]
    df_features.columns = [
        col.value if hasattr(col, "value") else str(col) for col in df_features.columns
    ]

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(df_features, df_mig["coherent_migration"])
    optimal_axis = lda.coef_[0]
    projected_data = lda.transform(df_features)

    # Plot the weights of each pc in the optimal axis
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(features_to_rank, optimal_axis)
    ax.set_xticks(range(len(features_to_rank)))
    ax.set_xticklabels(features_to_rank, rotation=45, ha="right", fontsize=6)
    fig.tight_layout()
    plt.show()
    fig.savefig(output_dir / f"lda_optimal_axis_{fname_suffix}.png", dpi=150)
    plt.close(fig)

    df_proj = pd.DataFrame(
        np.c_[projected_data, df_mig["coherent_migration"]], columns=["LDA", "coherent_migration"]
    )

    lda_transform = {
        "weights": optimal_axis.tolist(),
        "intercept": float(lda.intercept_[0]),
        "features": features_to_rank,
    }
    json_path = output_dir / f"lda_transform_{fname_suffix}.json"
    with open(json_path, "w") as f:
        json.dump(lda_transform, f, indent=4)

    for minimal_weight in [2.0, 3.0, 4.0]:
        sparse_axis = np.where(np.abs(optimal_axis) >= minimal_weight, optimal_axis, 0)
        print("Highly contributing pcs at minimal weight threshold of", minimal_weight)
        print([features_to_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
        projected_data_sparse = df_features @ sparse_axis + lda.intercept_[0]
        df_proj[f"LDA_SP_{int(minimal_weight)}"] = projected_data_sparse

    return lda_transform, df_proj


# %%
def apply_lda_projection(
    df, features_included_in_lda_rank, lda_weights, lda_intercept, sparse_axes=None
) -> pd.DataFrame:

    df_features = df[features_included_in_lda_rank]
    lda_weights = np.array(lda_weights)
    lda_intercept = float(lda_intercept)
    df_result = pd.DataFrame(index=df_features.index)
    # LDA projection
    df_result["LDA"] = df_features @ lda_weights + lda_intercept
    # Sparse projections
    if sparse_axes is not None:
        for minimal_weight in [2.0, 3.0, 4.0]:
            sparse_axis = np.where(np.abs(lda_weights) >= minimal_weight, lda_weights, 0)
            print("Highly contributing pcs at minimal weight threshold of", minimal_weight)
            print([features_to_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = df_features @ sparse_axis + lda_intercept
            df_result[f"LDA_SP_{int(minimal_weight)}"] = projected_data_sparse

    # merge the df_result with the original df to keep all other columns
    df_result = df.merge(df_result, left_index=True, right_index=True)
    return df_result


# %% PC ranking and histogram plotting
features_to_rank = pc_columns_to_keep + other_cols_to_keep
rank_features_and_plot_histograms(
    df_mig,
    features_to_rank=features_to_rank,
    label_column="coherent_migration",
    output_dir=output_dir,
)

# %% LDA feature ranking and histogram plotting, pcs only
lda_transform, df_proj = run_lda_feature_ranking(df_mig, pc_columns_to_keep, output_dir, "pcs_only")
rank_features_and_plot_histograms(
    df_proj,
    df_proj.columns.drop("coherent_migration"),
    label_column="coherent_migration",
    fname="find_coherent_mig_histograms_lda_pcs_only.png",
)

# %% apply LDA projection to original dataframe
dataset_name = "20250319_20X"
df = get_dataframe_for_dynamics_workflows(
    dataset_name, dataframe_manifest, pca, filter_dataframe=True
)
# %%
df_proj_full = apply_lda_projection(
    df,
    features_included_in_lda_rank=lda_transform["features"],
    lda_weights=lda_transform["weights"],
    lda_intercept=lda_transform["intercept"],
    sparse_axes=[2.0, 3.0, 4.0],
)
# %%
