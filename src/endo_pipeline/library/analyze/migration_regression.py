"""Assemble per-fixed-point tables for migration-coherence regression analysis."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_shear_stress,
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.summary_plot import _process_bootstrap_dataframe_for_plot
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.unicode import UnicodeCharacters

logger = logging.getLogger(__name__)


def assemble_fixed_points_dataframe(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_bootstrap_dataframe_manifest: DataframeManifest,
    bootstrap_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Assemble per-fixed-point dataframe across datasets and flow conditions.

    For each ``(dataset, flow_condition)`` tuple this loads the per-dataset
    feature dataframe, adds optical-flow features, loads the corresponding
    bootstrapped fixed points, filters them by ``bootstrap_threshold``, and
    enriches each fixed point with binned-mean optical-flow values (migration
    coherence and speed) and the nematic-order column. The resulting rows are
    concatenated across all datasets/conditions.

    Replicates the data-assembly portion of
    :func:`endo_pipeline.library.visualize.summary_plot.plot_cross_dataset_summaries`
    so the same per-fixed-point table can be used for downstream regression.

    Parameters
    ----------
    dataset_names
        Datasets to include.
    feature_dataframe_manifest
        Manifest of per-dataset feature dataframes (PCA-filtered).
    fixed_points_bootstrap_dataframe_manifest
        Manifest of per-dataset bootstrapped-fixed-points dataframes.
    bootstrap_threshold
        Minimum bootstrap detection rate to retain a fixed point.

    Returns
    -------
    pandas.DataFrame
        One row per high-confidence fixed point per flow condition, with
        cluster-mean structural columns (``polar_angle_cluster_mean``,
        ``polar_radius_cluster_mean``, ``pc3_flipped_cluster_mean``,
        ``nematic_order_cluster_mean``), confidence-interval columns, and
        binned-mean optical-flow features (``mean_optical_flow_*``).
    """
    column_names = [
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
        ColumnName.DiffAEData.PC3_FLIPPED,
    ]
    optical_flow_features = [
        ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
        ColumnName.OpticalFlow.SPEED_MEAN,
    ]

    df_fp_all_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning("No feature dataframe for [ %s ]. Skipping.", dataset_name)
            continue
        if dataset_name not in fixed_points_bootstrap_dataframe_manifest.locations:
            logger.warning("No fixed-point bootstrap dataframe for [ %s ]. Skipping.", dataset_name)
            continue

        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
        df = df_[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
        df_of = add_optical_flow_features(df_steady_state, datasets=[dataset_name])

        df_bootstrap = load_dataframe(
            fixed_points_bootstrap_dataframe_manifest.locations[dataset_name], delay=False
        )

        for flow_condition in dataset_config.flow_conditions:
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_of, dataset_config, flow_condition
            )
            df_bootstrap_flow = filter_dataframe_by_shear_stress(
                df_bootstrap, flow_condition.shear_stress
            )
            df_fp = _process_bootstrap_dataframe_for_plot(
                df_bootstrap_flow,
                df_flow,
                bootstrap_threshold,
                dataset_name,
                flow_condition,
                optical_flow_features,
                convert_angle_to_nematic=True,
                column_names=column_names,
                x_axis_mode="dataset",
                dataset_config=dataset_config,
            )
            if not df_fp.empty:
                df_fp_all_list.append(df_fp)

    if not df_fp_all_list:
        return pd.DataFrame()
    return pd.concat(df_fp_all_list, ignore_index=True)


# Coherence column produced by add_binned_mean_to_fixed_points for the
# UNIT_VECTOR_MEAN optical-flow feature. Defined as a module-level constant so
# both regression and plotting helpers stay in sync.
COHERENCE_COLUMN = f"mean_{ColumnName.OpticalFlow.UNIT_VECTOR_MEAN}"
COHERENCE_CI_LOWER = f"{COHERENCE_COLUMN}_{ColumnName.BootstrapAnalysis.CI_LOWER}"
COHERENCE_CI_UPPER = f"{COHERENCE_COLUMN}_{ColumnName.BootstrapAnalysis.CI_UPPER}"


def build_feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build named feature matrices for regressing migration coherence.

    Polar angle is encoded as ``(cos theta, sin theta)`` to respect angular
    periodicity; nematic order is included as the head-tail-symmetric scalar
    ``cos(2*theta)`` already produced by the assembly step.

    Parameters
    ----------
    df
        Per-fixed-point dataframe produced by
        :func:`assemble_fixed_points_dataframe`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from feature-set name to feature matrix (rows aligned with
        ``df``).
    """
    theta = df[f"{ColumnName.DiffAEData.POLAR_ANGLE}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"]
    r = df[f"{ColumnName.DiffAEData.POLAR_RADIUS}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"]
    rho = df[f"{ColumnName.DiffAEData.PC3_FLIPPED}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"]
    nematic = df[
        f"{ColumnName.DiffAEData.NEMATIC_ORDER}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"
    ]
    theta_x = np.cos(theta)
    theta_y = np.sin(theta)

    return {
        "theta_only": pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y}),
        "theta_r": pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y, "polar_r": r}),
        "theta_rho": pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y, "rho": rho}),
        "full_3d": pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y, "polar_r": r, "rho": rho}),
        "nematic_only": pd.DataFrame({"nematic_order": nematic}),
        "n_r_rho": pd.DataFrame({"nematic_order": nematic, "polar_r": r, "rho": rho}),
    }


def estimate_noise_floor_mse(df: pd.DataFrame) -> float:
    """
    Estimate per-point measurement-noise variance from coherence bootstrap CIs.

    Treats the bootstrap 95%-style CI half-width as ``~1.96 * sigma`` and
    averages ``sigma**2`` across fixed points. This gives a lower bound on the
    held-out MSE achievable by any coherence predictor.
    """
    half_width = 0.5 * (df[COHERENCE_CI_UPPER] - df[COHERENCE_CI_LOWER])
    sigma = half_width / 1.96
    return float(np.mean(sigma**2))


def leave_one_dataset_out_regression(
    df: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame] | None = None,
    target_column: str = COHERENCE_COLUMN,
    dataset_column: str = ColumnName.DATASET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run leave-one-dataset-out linear regression of coherence on each feature set.

    Each fold trains a linear regression on all datasets except one and
    predicts the held-out dataset's stable points. Predictions are pooled
    across folds and metrics (R², MSE, RMSE) are reported on the pooled
    held-out predictions.

    Parameters
    ----------
    df
        Per-fixed-point dataframe (already filtered to stable points and to
        rows with a non-NaN target).
    feature_sets
        Mapping from feature-set name to feature matrix. If ``None``, uses
        :func:`build_feature_sets`.
    target_column
        Name of the coherence column.
    dataset_column
        Name of the dataset-id column used to define folds.

    Returns
    -------
    benchmark : pandas.DataFrame
        One row per feature set with columns
        ``["feature_set", "n_features", "n_points", "n_folds", "cv_r2",
        "cv_mse", "cv_rmse"]``.
    predictions : pandas.DataFrame
        Per-point held-out predictions with columns
        ``["feature_set", dataset_column, "y_true", "y_pred"]``.
    """
    if feature_sets is None:
        feature_sets = build_feature_sets(df)

    datasets = df[dataset_column].unique()
    if len(datasets) < 2:
        raise ValueError(f"Leave-one-dataset-out CV requires >=2 datasets, got {len(datasets)}.")

    benchmark_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    y_full = df[target_column].to_numpy()

    for name, X in feature_sets.items():
        y_pred = np.full_like(y_full, np.nan, dtype=float)
        X_arr = X.to_numpy()
        for held in datasets:
            train_mask = (df[dataset_column] != held).to_numpy()
            test_mask = ~train_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            model = LinearRegression()
            model.fit(X_arr[train_mask], y_full[train_mask])
            y_pred[test_mask] = model.predict(X_arr[test_mask])

        valid = ~np.isnan(y_pred)
        cv_mse = float(mean_squared_error(y_full[valid], y_pred[valid]))
        benchmark_rows.append(
            {
                "feature_set": name,
                "n_features": X.shape[1],
                "n_points": int(valid.sum()),
                "n_folds": len(datasets),
                "cv_r2": float(r2_score(y_full[valid], y_pred[valid])),
                "cv_mse": cv_mse,
                "cv_rmse": float(np.sqrt(cv_mse)),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_set": name,
                    dataset_column: df[dataset_column].to_numpy()[valid],
                    "shear_stress": df["flow_condition_shear_stress_bin"].to_numpy()[valid],
                    "y_true": y_full[valid],
                    "y_pred": y_pred[valid],
                }
            )
        )

    return pd.DataFrame(benchmark_rows), pd.concat(prediction_frames, ignore_index=True)


def _plot_scatter_ax(
    ax: plt.Axes,
    sub: pd.DataFrame,
    lim: tuple[float, float],
    vmin: float,
    vmax: float,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> plt.PathCollection:
    """
    Draw a single true-vs-predicted scatter panel onto *ax*.

    Parameters
    ----------
    ax
        Axes to draw on.
    sub
        Rows of the predictions dataframe for one feature set.
    lim
        Shared (min, max) axis limits.
    vmin, vmax
        Colormap range for the shear-stress colour encoding.
    title
        Axes title string (typically feature-set name plus R² annotation).
    show_ylabel
        Whether to label the y axis ``"predicted coherence"``.
    show_xlabel
        Whether to label the x axis ``"true coherence"``.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter artist, needed by the caller to attach a colorbar.
    """
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    scatter = ax.scatter(
        sub["y_true"],
        sub["y_pred"],
        c=sub["shear_stress"],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=20,
        edgecolor="black",
        linewidths=0.4,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(alpha=0.3)
    if show_ylabel:
        ax.set_ylabel("predicted coherence")
    if show_xlabel:
        ax.set_xlabel("true coherence")
    return scatter


def plot_single_feature_scatter(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    feature_set: str,
) -> plt.Figure:
    """
    Plot true vs predicted coherence for a single feature set.

    Parameters
    ----------
    predictions
        Per-point held-out predictions returned by
        :func:`leave_one_dataset_out_regression`.
    benchmark
        Benchmark dataframe returned by
        :func:`leave_one_dataset_out_regression`. Used for R² annotations.
    feature_set
        Name of the feature set to plot (must appear in ``benchmark``).
    """
    row = benchmark.loc[benchmark["feature_set"] == feature_set].iloc[0]
    title = (
        f"{feature_set}\nR{UnicodeCharacters.SQUARED} = {row['cv_r2']:.3f}  (n = {row['n_points']})"
    )

    sub = predictions[predictions["feature_set"] == feature_set]
    y_min = float(min(sub["y_true"].min(), sub["y_pred"].min()))
    y_max = float(max(sub["y_true"].max(), sub["y_pred"].max()))
    pad = 0.05 * (y_max - y_min)
    lim = (y_min - pad, y_max + pad)
    vmin = float(predictions["shear_stress"].min())
    vmax = float(predictions["shear_stress"].max())

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.set_aspect("equal")
    scatter = _plot_scatter_ax(ax, sub, lim, vmin, vmax, title, show_ylabel=True, show_xlabel=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(f"shear stress (dyn/cm{UnicodeCharacters.SQUARED})")
    return fig


def plot_predictions_scatter(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> plt.Figure:
    """
    Plot held-out true vs predicted migration coherence per feature set.

    One subplot per feature set, with a y = x reference line and the CV R^2
    annotated in the title. Axes are shared so feature sets are visually
    comparable.

    Parameters
    ----------
    predictions
        Per-point held-out predictions returned by
        :func:`leave_one_dataset_out_regression`.
    benchmark
        Benchmark dataframe returned by
        :func:`leave_one_dataset_out_regression`. Used for R² annotations.
    """
    feature_sets = list(benchmark["feature_set"])
    n = len(feature_sets)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols + 0.6, 3.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    y_min = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    y_max = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    pad = 0.05 * (y_max - y_min)
    lim = (y_min - pad, y_max + pad)
    vmin = float(predictions["shear_stress"].min())
    vmax = float(predictions["shear_stress"].max())

    r2_by_set = dict(zip(benchmark["feature_set"], benchmark["cv_r2"], strict=False))
    n_pts_by_set = dict(zip(benchmark["feature_set"], benchmark["n_points"], strict=False))

    for i, fs in enumerate(feature_sets):
        ax = axs[i // ncols][i % ncols]
        sub = predictions[predictions["feature_set"] == fs]
        title = (
            f"{fs}\nR{UnicodeCharacters.SQUARED} = {r2_by_set[fs]:.3f}  (n = {n_pts_by_set[fs]})"
        )
        scatter = _plot_scatter_ax(
            ax,
            sub,
            lim,
            vmin,
            vmax,
            title,
            show_ylabel=(i % ncols == 0),
            show_xlabel=(i // ncols == nrows - 1),
        )

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")

    cbar = fig.colorbar(scatter, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(f"shear stress (dyn/cm{UnicodeCharacters.SQUARED})")
    return fig
