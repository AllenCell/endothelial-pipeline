import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from endo_pipeline.io import save_plot_to_path

logger = logging.getLogger(__name__)


def compute_separation_power(
    df_features: pd.DataFrame, y: pd.Series
) -> list[dict[str, float | str]]:
    """Rank features by class separation power based on ROC AUC.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Feature matrix with shape ``(n_samples, n_features)``.
    y : pandas.Series
        Binary labels aligned to ``df_features`` rows.

    Returns
    -------
    list[dict[str, float | str]]
        Sorted ranking entries with keys ``feature``, ``auc``, and ``power``.
    """
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
    for item in ranking_sorted[:10]:
        logger.info(f"{item['feature']}: AUC={item['auc']:.3f}, Power={item['power']:.3f}")
    return ranking_sorted


def fit_lda_feature_ranking(
    df_mig: pd.DataFrame,
    features_to_rank: list[str],
    binary_target_feature: str,
) -> tuple[list[str], np.ndarray, float, np.ndarray]:
    """Fit a one-component LDA model for coherent migration labels.

    Parameters
    ----------
    df_mig : pandas.DataFrame
        Input dataframe containing feature columns and binary target labels.
    features_to_rank : list[str]
        Feature columns used to fit the LDA model.
    binary_target_feature : str
        Column name used as the binary target label for LDA fitting.

    Returns
    -------
    tuple[list[str], numpy.ndarray, float, numpy.ndarray]
        Normalized feature names, LDA coefficients, intercept, and transformed
        projections of shape ``(n_samples, 1)``.
    """
    features_to_rank = [str(col) for col in features_to_rank]
    df_features = df_mig[features_to_rank]

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(df_features, df_mig[binary_target_feature])
    optimal_axis = lda.coef_[0]
    projected_data = lda.transform(df_features)
    lda_intercept = float(lda.intercept_[0])

    return features_to_rank, optimal_axis, lda_intercept, projected_data


def plot_lda_optimal_axis(
    features_to_rank: list[str],
    optimal_axis: np.ndarray,
    output_dir: Path,
    fname_suffix: str = "",
    title_suffix: str = "",
):
    """Plot and save LDA coefficients across ranked features.

    Parameters
    ----------
    features_to_rank : list[str]
        Ordered feature names corresponding to coefficient values.
    optimal_axis : numpy.ndarray
        LDA coefficient vector.
    output_dir : pathlib.Path
        Directory where the figure is written.
    fname_suffix : str, default=""
        Optional suffix appended to the output filename.
    title_suffix : str, default=""
        Optional suffix appended to the plot title.

    Returns
    -------
    None
        This function saves a plot and returns nothing.
    """
    # Plot the weights of each pc in the optimal axis
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(features_to_rank, optimal_axis)
    ax.set_title(f"LDA Optimal Axis {title_suffix}".strip())
    ax.set_xticks(range(len(features_to_rank)))
    ax.set_xticklabels(features_to_rank, rotation=45, ha="right", fontsize=6)
    fig.tight_layout()
    plt.show()
    fig.savefig(output_dir / f"lda_optimal_axis_{fname_suffix}.png", dpi=150)
    plt.close(fig)


def build_lda_outputs(
    df_mig: pd.DataFrame,
    features_to_rank: list[str],
    optimal_axis: np.ndarray,
    lda_intercept: float,
    projected_data: np.ndarray,
    binary_target_feature: str,
    minimal_weight: list[float] | None = [2.0, 3.0, 4.0, 5.0],
    save: bool = False,
    output_dir: Path | None = None,
    fname_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """Build LDA output tables and save transform weights CSV.

    Parameters
    ----------
    df_mig : pandas.DataFrame
        Input dataframe containing feature columns and binary target labels.
    features_to_rank : list[str]
        Ordered feature names corresponding to the LDA coefficients.
    optimal_axis : numpy.ndarray
        LDA coefficient vector.
    lda_intercept : float
        LDA intercept term.
    projected_data : numpy.ndarray
        LDA projections from ``fit_lda_feature_ranking``.
    binary_target_feature : str
        Column name copied to the projection dataframe as the target label.
    minimal_weight : list[float] or None, default=[2.0, 3.0, 4.0, 5.0]
        Thresholds used to create sparse projection columns; if ``None``, no
        sparse columns are added.
    save : bool, default=False
        If ``True``, saves ``df_lda`` as a CSV and returns its path.
    output_dir : pathlib.Path or None, default=None
        Directory used to save the transform CSV.
    fname_suffix : str, default=""
        Optional suffix appended to the CSV filename.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pathlib.Path | None]
        LDA transform dataframe, projection dataframe, and optional CSV path.
    """
    df_features = df_mig[features_to_rank]

    df_proj = pd.DataFrame(
        np.c_[projected_data, df_mig[binary_target_feature]],
        columns=["LDA", binary_target_feature],
    )

    # Convert to DataFrame
    df_lda = pd.DataFrame({"weights": optimal_axis.tolist(), "features": features_to_rank})
    df_lda["intercept"] = lda_intercept

    if minimal_weight is not None:
        for weight in minimal_weight:
            sparse_axis = np.where(np.abs(optimal_axis) >= weight, optimal_axis, 0)
            logger.info(f"Highly contributing pcs at minimal weight threshold of {weight}")
            logger.info([features_to_rank[pc] for pc in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = df_features @ sparse_axis + lda_intercept
            df_proj[f"LDA_SP_{int(weight)}"] = projected_data_sparse

    csv_path: Path | None = None
    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True")
        csv_path = output_dir / f"lda_feature_ranking_{fname_suffix}.csv"
        df_lda.to_csv(csv_path, index=False)

    return df_lda, df_proj, csv_path


def apply_lda_projection(
    df: pd.DataFrame,
    features_in_lda_rank: list[str],
    lda_weights: np.ndarray,
    lda_intercept: float,
    sparse_axes: list[float] | None = None,
) -> pd.DataFrame:
    """Apply stored LDA coefficients to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing ``features_in_lda_rank``.
    features_in_lda_rank : list[str]
        Feature column order matching ``lda_weights``.
    lda_weights : numpy.ndarray
        LDA coefficient vector.
    lda_intercept : float
        LDA intercept term.
    sparse_axes : list[float] or None, default=None
        If provided, sparse projections are computed at fixed thresholds.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with appended LDA projection columns.
    """
    df_features = df[features_in_lda_rank]
    lda_weights_array = np.asarray(lda_weights, dtype=float)
    lda_intercept = float(lda_intercept)
    df_result = pd.DataFrame(index=df_features.index)
    # LDA projection
    df_result["LDA"] = df_features @ lda_weights_array + lda_intercept
    # Sparse projections
    if sparse_axes is not None:
        for minimal_weight in sparse_axes:
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


def plot_ranked_feature_histograms(
    df: pd.DataFrame,
    ranking: list[dict[str, float | str]],
    output_dir: Path,
    label_column: str = "migration_type",
    fname: str = "find_coherent_mig_histograms.png",
    legend_suffix: str = "",
):
    """Plot and save histograms for ranked features by class label.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing ranked features and labels.
    ranking : list[dict[str, float | str]]
        Ranking metadata from ``compute_separation_power``.
    output_dir : pathlib.Path
        Directory where the figure is saved.
    label_column : str, default="migration_type"
        Column used to separate classes in histograms.
    fname : str, default="find_coherent_mig_histograms.png"
        Output figure filename.
    legend_suffix : str, default=""
        Optional text appended to class names in the figure legend.

    Returns
    -------
    None
        This function saves a figure and returns nothing.
    """
    n_pcs = len(ranking)
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
    suffix_text = f" {legend_suffix}" if legend_suffix else ""
    legend_labels = [
        f"Coherent Migration{suffix_text} (N={n_coherent})",
        f"Mixed Migration{suffix_text} (N={n_mixed})",
    ]
    fig.legend(legend_labels, loc="lower right", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, fname)
    plt.close()
