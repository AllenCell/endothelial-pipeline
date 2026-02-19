import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


def apply_lda_projection(
    df: pd.DataFrame,
    features_in_lda_rank: list[str],
    lda_weights: np.ndarray | list,
    lda_intercept: float,
    sparse_axes: list[float] | None = None,
) -> pd.DataFrame:

    df_features = df[features_in_lda_rank]
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
            print([features_in_lda_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = df_features @ sparse_axis + lda_intercept
            df_result[f"LDA_SP_{int(minimal_weight)}"] = projected_data_sparse

    # merge the df_result with the original df to keep all other columns
    df_result = df.merge(df_result, left_index=True, right_index=True)
    return df_result
