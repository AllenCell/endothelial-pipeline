import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import CCA

from endo_pipeline.io import save_plot_to_path

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
    if cca_results.empty:
        return

    target_feature = str(cca_results.attrs["target_feature"])
    corr = float(cca_results.attrs["corr"])
    xdata = np.asarray(cca_results.attrs["x_canonical"])
    ydata = np.asarray(cca_results.attrs["y_canonical"])
    x_weights = cca_results["weight"].to_numpy()
    input_features = cca_results["input_feature"].tolist()
    datasets_used = list(cca_results.attrs.get("datasets_used", []))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13, 4),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    xmin, xmax = np.percentile(xdata, [1, 99])
    ymin, ymax = np.percentile(ydata, [1, 99])

    density = ax1.hexbin(
        xdata,
        ydata,
        gridsize=80,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    cbar = fig.colorbar(density, ax=ax1, pad=0.01)
    cbar.set_label("Point density (log scale)")
    ax1.set_xlabel("Canonical Variable 1 (PCs)")
    ax1.set_ylabel(f"Canonical Variable 1 ({target_feature})")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax2.bar(range(len(input_features)), x_weights)
    ax2.set_xticks(range(len(input_features)))
    ax2.set_xticklabels(input_features, rotation=45, ha="right", fontsize=xtick_fontsize)
    if datasets_used:
        dataset_handles = [
            Line2D([], [], linestyle="none", marker=None, label=dataset)
            for dataset in datasets_used
        ]
        ax2.legend(
            handles=dataset_handles,
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
    save_plot_to_path(
        fig,
        output_dir,
        f"cca_projection_{target_feature}.png",
    )
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
    cca_results: pd.DataFrame,
    sparse_axes: list[float] | None = None,
) -> pd.DataFrame:
    """Apply stored CCA coefficients to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the CCA input features.
    cca_results : pandas.DataFrame
        CCA weights DataFrame returned by ``calculate_cca_results``,
        with columns ``input_feature`` and ``weight``.
    sparse_axes : list[float] or None, default=None
        If provided, sparse projections are computed at fixed thresholds.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with appended CCA projection columns.
    """
    features_in_rank = cca_results["input_feature"].tolist()
    cca_weights = cca_results["weight"].to_numpy()
    target_feature = cca_results.attrs.get("target_feature", "cca")

    df_features = df[features_in_rank]
    df_result = pd.DataFrame(index=df_features.index)

    # CCA projection (center then project)
    df_result[f"CCA_{target_feature}"] = (df_features - df_features.mean()).to_numpy() @ cca_weights

    # Sparse projections
    if sparse_axes is not None:
        for minimal_weight in sparse_axes:
            sparse_axis = np.where(np.abs(cca_weights) >= minimal_weight, cca_weights, 0)
            logger.info(
                f"Highly contributing features at minimal weight threshold of {minimal_weight}"
            )
            logger.info([features_in_rank[i] for i in np.where(np.abs(sparse_axis) > 0)[0]])
            projected_data_sparse = (df_features - df_features.mean()).to_numpy() @ sparse_axis
            df_result[f"CCA_SP_{int(minimal_weight)}_{target_feature}"] = projected_data_sparse

    df_result = df.merge(df_result, left_index=True, right_index=True)
    return df_result
