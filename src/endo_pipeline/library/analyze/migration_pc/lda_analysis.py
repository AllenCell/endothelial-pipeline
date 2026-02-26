import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from endo_pipeline.io import save_plot_to_path

logger = logging.getLogger(__name__)


def compute_separation_power(df_features: pd.DataFrame, y: pd.Series, verbose: bool = True):
    # Assuming 'df_features' is your (M samples x N features) matrix
    # Assuming 'y' is your binary label vector (0s and 1s)
    ranking = []
    for feature_name in df_features.columns:
        # Calculate AUC
        score = roc_auc_score(y, df_features[feature_name])
        # We care about "Separation Power", so 0.1 is just as good as 0.9.
        # We calculate 'power' as distance from 0.5 (randomness)
        separation_power = 2.0 * abs(score - 0.5)

        ranking.append({"feature": feature_name, "auc": score, "power": separation_power})

    logger.info("Top features by separation power:")
    ranking_sorted = sorted(ranking, key=lambda x: x["power"], reverse=True)
    if verbose:
        for item in ranking_sorted[:10]:
            logger.info(f"{item['feature']}: AUC={item['auc']:.3f}, Power={item['power']:.3f}")
    return ranking_sorted


def run_lda_feature_ranking(
    df_mig: pd.DataFrame,
    features_to_rank: list[str],
    output_dir: Path,
    fname_suffix: str = "",
    minimal_weight: list[float] | None = [2.0, 3.0, 4.0, 5.0],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:

    features_to_rank = [str(col) for col in features_to_rank]
    df_features = df_mig[features_to_rank]

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

    # Convert to DataFrame
    df_lda = pd.DataFrame({"weights": optimal_axis.tolist(), "features": features_to_rank})
    df_lda["intercept"] = float(lda.intercept_[0])

    # Save as CSV
    csv_path = output_dir / f"lda_transform_{fname_suffix}.csv"
    df_lda.to_csv(csv_path, index=False)

    if minimal_weight is not None:
        for weight in minimal_weight:
            sparse_axis = np.where(np.abs(optimal_axis) >= weight, optimal_axis, 0)
            logger.info(f"Highly contributing pcs at minimal weight threshold of {weight}")
            logger.info([features_to_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = df_features @ sparse_axis + lda.intercept_[0]
            df_proj[f"LDA_SP_{int(weight)}"] = projected_data_sparse

    return df_lda, df_proj, csv_path


def apply_lda_projection(
    df: pd.DataFrame,
    features_in_lda_rank: list[str],
    lda_weights: np.ndarray,
    lda_intercept: float,
    sparse_axes: list[float] | None = None,
) -> pd.DataFrame:

    df_features = df[features_in_lda_rank]
    lda_weights_array = np.asarray(lda_weights, dtype=float)
    lda_intercept = float(lda_intercept)
    df_result = pd.DataFrame(index=df_features.index)
    # LDA projection
    df_result["LDA"] = df_features @ lda_weights_array + lda_intercept
    # Sparse projections
    if sparse_axes is not None:
        for minimal_weight in [2.0, 3.0, 4.0]:
            sparse_axis = np.where(
                np.abs(lda_weights_array) >= minimal_weight, lda_weights_array, 0
            )
            logger.info(f"Highly contributing pcs at minimal weight threshold of {minimal_weight}")
            logger.info([features_in_lda_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = df_features @ sparse_axis + lda_intercept
            df_result[f"LDA_SP_{int(minimal_weight)}"] = projected_data_sparse

    # merge the df_result with the original df to keep all other columns
    df_result = df.merge(df_result, left_index=True, right_index=True)
    return df_result


def rank_features_and_plot_histograms(
    df: pd.DataFrame,
    features_to_rank: list[str],
    output_dir: Path,
    label_column: str = "migration_type",
    fname: str = "find_coherent_mig_histograms.png",
):
    ranking = compute_separation_power(df[features_to_rank], df[label_column])

    n_pcs = len(features_to_rank)
    ncols = len(ranking) if len(ranking) < 10 else 10
    nrows = (n_pcs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows + ncols > 2:
        axes = axes.flatten()
    else:
        axes = [axes]

    label_map: dict[bool | str, str]
    if label_column == "coherent_migration":
        label_map = {True: "coherent", False: "mixed"}
    else:
        label_map = {"coherent": "Coherent Migration", "mixed": "Mixed Migration"}

    for i, item in enumerate(ranking):
        col = item["feature"]
        x_min = df[col].min()
        x_max = df[col].max()
        for label, label_name in label_map.items():
            subset = df[df[label_column] == label]
            axes[i].hist(subset[col], bins=30, range=(x_min, x_max), alpha=0.75, label=label_name)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"{col}, Power: {item['power']:.3f}")

    if label_column == "coherent_migration":
        n_coherent = int(df[label_column].eq(True).sum())
        n_mixed = int(df[label_column].eq(False).sum())
    else:
        n_coherent = int(df[label_column].eq("coherent").sum())
        n_mixed = int(df[label_column].eq("mixed").sum())
    legend_labels = [
        f"Coherent Migration (N={n_coherent})",
        f"Mixed Migration (N={n_mixed})",
    ]
    fig.legend(legend_labels, loc="lower right", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, fname)
    plt.close()
