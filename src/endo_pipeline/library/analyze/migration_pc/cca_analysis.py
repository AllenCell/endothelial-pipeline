import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import CCA

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.manifests import load_dataframe_manifest

logger = logging.getLogger(__name__)


def calculate_cca_results(
    df: pd.DataFrame,
    target_feature: str,
    input_features: list[str],
    output_dir: Path,
    n_components: int = 1,
    max_iter: int = 5000,
    tol: float = 1e-12,
    dataset_column: str = "dataset",
    scale_cca: bool = False,
) -> tuple[pd.DataFrame, Path]:

    datasets_used = df[dataset_column].unique().tolist()
    x = df[input_features].to_numpy()
    y = df[target_feature].to_numpy().reshape(-1, 1)

    cca = CCA(n_components=n_components, max_iter=max_iter, tol=tol, scale=scale_cca)
    x_c, y_c = cca.fit_transform(x, y)
    corr = float(np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1])
    corr_original = float(np.corrcoef(x_c[:, 0], y.flatten())[0, 1])

    df_result = pd.DataFrame({"input_feature": input_features, "weight": cca.x_weights_[:, 0]})
    df_result.attrs["target_feature"] = target_feature
    df_result.attrs["corr"] = corr
    df_result.attrs["corr_original"] = corr_original
    df_result.attrs["x_canonical"] = x_c[:, 0]
    df_result.attrs["y_canonical"] = y_c[:, 0]
    df_result.attrs["y_original"] = y.flatten()
    df_result.attrs["datasets_used"] = datasets_used

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"cca_weights_{target_feature}.csv"
    df_result.to_csv(csv_path, index=False)
    logger.info("Saved CCA weights to %s", csv_path)

    return df_result, csv_path


def plot_cca_results(
    cca_results: pd.DataFrame,
    output_dir: Path,
    xtick_fontsize: int = 8,
) -> None:
    target_feature = str(cca_results.attrs["target_feature"])
    corr = float(cca_results.attrs["corr"])
    x_c = np.asarray(cca_results.attrs["x_canonical"])
    y_c = np.asarray(cca_results.attrs["y_canonical"])
    weights = cca_results["weight"].to_numpy()
    features = cca_results["input_feature"].tolist()
    datasets_used = cca_results.attrs.get("datasets_used", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 2]})

    # Density scatter of canonical variables
    xmin, xmax = np.percentile(x_c, [1, 99])
    ymin, ymax = np.percentile(y_c, [1, 99])
    hb = ax1.hexbin(x_c, y_c, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(hb, ax=ax1, pad=0.01).set_label("Point density (log scale)")
    ax1.set(
        xlabel="Canonical Variable 1 (PCs)",
        ylabel=f"Canonical Variable 1 ({target_feature})",
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )

    # Weight bar chart
    ax2.bar(range(len(features)), weights)
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha="right", fontsize=xtick_fontsize)
    ax2.legend(
        handles=[Line2D([], [], linestyle="none", label=d) for d in datasets_used],
        title="Datasets used",
        loc="upper right",
        frameon=True,
        fontsize=8,
        handlelength=0,
        handletextpad=0,
    )

    fig.suptitle(f"CCA Projection vs {target_feature}, Canonical Correlation = {corr:.3f}")
    fig.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, f"cca_projection_{target_feature}.png")
    plt.close(fig)


def plot_cca_projection_validation(
    df: pd.DataFrame,
    cca_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Scatter manual projection vs CCA canonical variable with a unity line.

    Points falling on the unity line confirm that the same data was used
    for PCA and CCA fitting, becuase pc scores are mean-centered and CCA projection is also mean-centered.
    Deviations from the unity line may indicate a mismatch in data used for PCA and CCA.
    """
    if cca_results.empty:
        return

    features = cca_results["input_feature"].tolist()
    weights = cca_results["weight"].to_numpy()
    x_canonical = np.asarray(cca_results.attrs["x_canonical"])
    target_feature = str(cca_results.attrs.get("target_feature", "cca"))

    projected = df[features].to_numpy() @ weights

    assert projected.shape == x_canonical.shape, (
        f"Shape mismatch: projected {projected.shape} vs x_canonical {x_canonical.shape}. "
        "Ensure the same rows (after dropna) are used for both."
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(projected, x_canonical, alpha=0.3, s=4)

    lims = [
        min(projected.min(), x_canonical.min()),
        max(projected.max(), x_canonical.max()),
    ]
    ax.plot(lims, lims, "--", color="red", label="Unity line (slope=1)")
    ax.set_xlabel("Manual projection (X @ weights)")
    ax.set_ylabel("CCA canonical variable")
    ax.set_title(f"CCA projection validation ({target_feature})")
    ax.legend()
    fig.tight_layout()
    plt.show()
    save_plot_to_path(fig, output_dir, f"cca_projection_validation_{target_feature}.png")
    plt.close(fig)


def apply_cca_projection(
    df: pd.DataFrame,
    manifest_name: str = "cca_weights",
    location_key: str = "80_pcs",
    sparse_axes: list[float] | None = [0.1, 0.2, 0.3],
    return_column_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load CCA weights from a manifest and project onto a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the CCA input features.
    manifest_name : str
        Name of the dataframe manifest containing the CCA weights.
    location_key : str
        Key within the manifest's locations dict (e.g. ``"80_pcs"``).
    sparse_axes : list[float] or None, default=[0.1, 0.2, 0.3]
        If provided, sparse projections are computed at fixed thresholds.
    return_column_info : bool, default=False
        If True, also return a dict mapping each added column name to
        the list of input features used.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, dict[str, list[str]]]
        The dataframe with appended CCA projection columns. If
        *return_column_info* is True, a tuple of (dataframe, column_info)
        is returned instead.
    """

    manifest = load_dataframe_manifest(manifest_name)
    location = manifest.locations[location_key]
    cca_weights_df = load_dataframe(location)

    features = cca_weights_df["input_feature"].tolist()
    weights = cca_weights_df["weight"].to_numpy()

    df_features = df[features]
    df_result = pd.DataFrame(index=df_features.index)
    column_info: dict[str, list[str]] = {}

    # Full CCA projection
    df_result["cca"] = df_features.to_numpy() @ weights
    column_info["cca"] = features

    # Top-3 PCs
    top3_features = ["pc_1", "pc_2", "pc_3"]
    top3_weights = (
        cca_weights_df.loc[cca_weights_df["input_feature"].isin(top3_features)]
        .set_index("input_feature")
        .loc[top3_features, "weight"]
        .to_numpy()
    )
    df_result["cca_top3"] = df[top3_features].to_numpy() @ top3_weights
    column_info["cca_top3"] = top3_features

    # Threshold-based sparse projections
    if sparse_axes is not None:
        for minimal_weight in sparse_axes:
            sparse_axis = np.where(np.abs(weights) >= minimal_weight, weights, 0)
            nonzero_idx = np.where(np.abs(sparse_axis) > 0)[0]
            sparse_features = [features[i] for i in nonzero_idx]

            projected_data_sparse = df_features.to_numpy() @ sparse_axis

            weight_str = str(minimal_weight).replace(".", "")
            col_name = f"cca_sp{weight_str}"
            df_result[col_name] = projected_data_sparse
            column_info[col_name] = sparse_features

    df_result = df.merge(df_result, left_index=True, right_index=True)
    if return_column_info:
        return df_result, column_info
    return df_result


def plot_feature_correlations(
    df: pd.DataFrame,
    features: list[str],
    target_feature: str,
    output_dir: Path | None = None,
) -> None:
    """Plot scatter subplots of features vs a target with correlation labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing all columns in *features* and *target_feature*.
    features : list[str]
        Column names to plot on the x-axes.
    target_feature : str
        Column name for the shared y-axis.
    output_dir : Path or None, default=None
        If provided, the figure is saved to this directory.
    """
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), sharey=True)
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features, strict=False):
        corr = df[[feature, target_feature]].corr().iloc[0, 1]
        ax.scatter(df[feature], df[target_feature], alpha=0.5, s=4)
        ax.set_title(f"corr={corr:.2f}")
        ax.set_xlabel(feature)
    axes[0].set_ylabel(target_feature)
    fig.tight_layout()
    plt.show()
    if output_dir is not None:
        save_plot_to_path(fig, output_dir, f"feature_correlations_{'_'.join(features[:3])}.png")
    plt.close(fig)
