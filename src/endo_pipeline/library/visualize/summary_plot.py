"""Methods for visualizing migration coherence metrics and their relationships to morphology dynamics."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_flow_condition,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
    add_shear_stress_to_df,
)
from endo_pipeline.library.visualize.diffae_features.fixed_points import (
    make_legend_handles_for_fixed_pts,
)
from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
)
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLOR_DICT, STABILITY_MARKER_DICT
from endo_pipeline.settings.summary_plot import CELL_LINE_LABEL_MAP, COLOR_PALETTE

logger = logging.getLogger(__name__)

# Unique color per dataset — colorblind-friendly palette (Wong 2011 + extensions)
_COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#882255",  # wine
    "#AA4499",  # magenta
]


# --- Build jitter map (shared by numeric and categorical shear-stress modes) ---
def _build_jitter_map(df: pd.DataFrame, jitter_width: float = 0.1) -> dict[tuple, float]:
    jmap: dict[tuple, float] = {}
    for ss in df["shear_stress_numeric"].unique():
        datasets_at_ss = df.loc[df["shear_stress_numeric"] == ss, "dataset"].unique()
        n = len(datasets_at_ss)
        if n <= 1:
            offsets = [0.0]
        else:
            offsets = [jitter_width * (i / (n - 1) - 0.5) for i in range(n)]
        for ds, off in zip(datasets_at_ss, offsets, strict=False):
            jmap[(ds, ss)] = off
    return jmap


def _compute_yerr(
    row: pd.Series,  # type: ignore[type-arg]
    y_val: float,
    ci_lower_col: str,
    ci_upper_col: str,
) -> list[list[float]] | None:
    """Return asymmetric *yerr* for :func:`~matplotlib.axes.Axes.errorbar`, or ``None``."""
    lo = row.get(ci_lower_col)
    hi = row.get(ci_upper_col)
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return None
    return [[max(0.0, y_val - lo)], [max(0.0, hi - y_val)]]


def plot_fixed_points_vs_shear_stress(
    df_fp: pd.DataFrame,
    variable: str,
    label: str,
    dataset_order: list[str] | None = None,
    ylimits: tuple[float, float] | None = None,
    x_axis_mode: Literal[
        "dataset", "shear_stress_numeric", "shear_stress_categorical", "cell_line"
    ] = "dataset",
    marker_size_scatter: int = 15,
    marker_size_legend: int = 5,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    stable_only: bool = True,
    ax: plt.Axes | None = None,
    jitter_width: float = 0.1,
) -> plt.Figure:
    """Make and save plot of one component of fixed points vs shear stress.

    Parameters
    ----------
    df_fp
        Concatenated fixed-points dataframe with a ``"shear_stress"`` column
        (e.g. from :func:`add_shear_stress_to_df`).
    variable
        Column name to plot on the y-axis.
    label
        Display label for the y-axis.
    dataset_order
        Optional list of dataset names specifying the desired x-axis order.
        If ``None``, falls back to sorting by shear stress.
    ylimits
        Optional ``(ymin, ymax)`` limits for the y-axis.
    x_axis_mode
        Controls x-axis layout:

        - ``"dataset"`` (default): one categorical tick per dataset, ordered by
          ``dataset_order`` or shear stress. Labels show
          ``"dataset_name (shear_stress)"``.
        - ``"shear_stress_numeric"``: x positions are the actual numeric
          shear-stress values, with jitter for datasets sharing the same value.
        - ``"shear_stress_categorical"``: one evenly-spaced tick per unique
          shear-stress value (so 6 and 21 are adjacent), with jitter for
          datasets sharing the same value.
        - ``"cell_line"``: one categorical tick per cell-line label
          (e.g. WT, Control, KD), ordered as WT → Control → KD, with
          jitter for datasets sharing the same shear-stress value.
    marker_size_scatter
        Size of the scatter markers for fixed points.
    marker_size_legend
        Size of the markers in the legend for fixed points.


    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    # Convert shear stress to numeric values
    df_fp = df_fp.copy()
    # Parse raw shear stress to the max numeric value in the label
    df_fp["shear_stress_numeric"] = df_fp["shear_stress"].apply(
        lambda s: max(float(v) for v in str(s).split("-"))
    )
    # Snap to ±1 bins; values outside any bin keep their rounded value
    _SHEAR_STRESS_BINS: dict[int, tuple[int, int]] = {
        6: (5, 7),
        9: (8, 10),
        12: (11, 13),
        15: (14, 16),
        21: (20, 22),
    }

    def _snap_to_bin(val: float) -> int:
        for center, (lo, hi) in _SHEAR_STRESS_BINS.items():
            if lo <= val <= hi:
                return center
        return round(val)

    df_fp["shear_stress_numeric"] = df_fp["shear_stress_numeric"].apply(_snap_to_bin)

    if stable_only:
        df_fp = df_fp[df_fp[ColumnName.VectorField.STABILITY] == "stable"]
    else:
        legend_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=df_fp[ColumnName.VectorField.STABILITY].unique().tolist(),
            marker_size=marker_size_legend,
        )

    # Order by specified dataset list, or fall back to shear stress sorting
    if dataset_order is not None:
        dataset_cat = pd.CategoricalDtype(categories=dataset_order, ordered=True)
        df_fp["dataset"] = df_fp["dataset"].astype(dataset_cat)
        df_fp = df_fp.sort_values("dataset")
    else:
        df_fp = df_fp.sort_values("shear_stress_numeric")

    row_to_x: Any  # noqa: E731
    tick_positions: list[float]
    if x_axis_mode == "dataset":
        # Categorical x-axis: one tick per dataset
        unique_datasets = df_fp["dataset"].unique()
        row_to_x = lambda row: {d: i for i, d in enumerate(unique_datasets)}[
            row["dataset"]
        ]  # noqa: E731
        tick_positions = list(range(len(unique_datasets)))
        tick_labels = [
            f"{d} ({df_fp.loc[df_fp['dataset'] == d, 'shear_stress_numeric'].iloc[0]})"
            for d in unique_datasets
        ]
    elif x_axis_mode == "shear_stress_numeric":
        # Numeric x-axis: position by shear stress value, jittered by dataset
        unique_shear = sorted(df_fp["shear_stress_numeric"].unique())
        tick_positions = unique_shear
        tick_labels = [str(round(s)) for s in unique_shear]
        jitter_map = _build_jitter_map(df_fp, jitter_width=jitter_width)
        row_to_x = lambda row: row["shear_stress_numeric"] + jitter_map.get(  # noqa: E731
            (row["dataset"], row["shear_stress_numeric"]), 0.0
        )
    elif x_axis_mode == "shear_stress_categorical":
        # Evenly-spaced categorical ticks for each unique shear stress value,
        # with jitter so datasets sharing a value are visible individually.
        unique_shear = sorted(df_fp["shear_stress_numeric"].unique())
        tick_spacing = 0.5  # compress horizontal spacing between categories
        ss_to_pos = {ss: i * tick_spacing for i, ss in enumerate(unique_shear)}
        tick_positions = [i * tick_spacing for i in range(len(unique_shear))]
        tick_labels = [str(round(s)) for s in unique_shear]
        jitter_map = _build_jitter_map(df_fp, jitter_width=jitter_width)
        row_to_x = lambda row: ss_to_pos[  # noqa: E731
            row["shear_stress_numeric"]
        ] + jitter_map.get((row["dataset"], row["shear_stress_numeric"]), 0.0)
    elif x_axis_mode == "cell_line":
        cell_line_catagories = df_fp["cell_line_label"].unique()
        # order by Parental line, then Control, then VE-Cad KD
        cell_line_order = sorted(
            cell_line_catagories,
            key=lambda x: (0 if x == "Parental" else 1 if x == "Control" else 2),
        )
        cell_line_dtype = pd.CategoricalDtype(categories=cell_line_order, ordered=True)
        df_fp["cell_line_label"] = df_fp["cell_line_label"].astype(cell_line_dtype)
        df_fp = df_fp.sort_values("cell_line_label")
        jitter_map = _build_jitter_map(df_fp, jitter_width=jitter_width)
        cell_line_to_code = {label: i for i, label in enumerate(cell_line_order)}
        row_to_x = lambda row: cell_line_to_code[
            row["cell_line_label"]
        ] + jitter_map.get(  # noqa: E731
            (row["dataset"], row["shear_stress_numeric"]), 0.0
        )
        tick_positions = list(range(len(cell_line_order)))
        tick_labels = list(cell_line_order)

    else:
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if stable_only:
        unique_datasets_list = df_fp["dataset"].unique()
        dataset_color_map = {
            ds: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, ds in enumerate(unique_datasets_list)
        }

        for _, row in df_fp.iterrows():
            y_val = row[variable]
            ci_lower_col = f"{variable}_{ColumnName.BootstrapAnalysis.CI_LOWER}"
            ci_upper_col = f"{variable}_{ColumnName.BootstrapAnalysis.CI_UPPER}"
            yerr = _compute_yerr(row, y_val, ci_lower_col, ci_upper_col)
            if yerr is not None:
                ax.errorbar(
                    row_to_x(row),
                    y_val,
                    yerr=yerr,
                    fmt="o",
                    color=dataset_color_map[row["dataset"]],
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=marker_size_scatter**0.5,
                    capsize=2,
                    elinewidth=0.8,
                    ecolor=dataset_color_map[row["dataset"]],
                    zorder=3,
                )
            else:
                ax.scatter(
                    row_to_x(row),
                    y_val,
                    marker="o",
                    color=dataset_color_map[row["dataset"]],
                    edgecolor="black",
                    linewidths=0.5,
                    s=marker_size_scatter,
                    zorder=3,
                )

    else:
        # Color by stability
        for _, row in df_fp.iterrows():
            stability = row[ColumnName.VectorField.STABILITY]
            mk = STABILITY_MARKER_DICT.get(stability, "o")
            clr = STABILITY_COLOR_DICT.get(stability, "gray")
            is_gray = stability not in STABILITY_COLOR_DICT
            y_val = row[variable]
            yerr = _compute_yerr(row, y_val, ci_lower_col, ci_upper_col)
            if yerr is not None:
                ax.errorbar(
                    row_to_x(row),
                    y_val,
                    yerr=yerr,
                    fmt=mk,
                    color=clr,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    markersize=marker_size_scatter**0.5,
                    capsize=2,
                    elinewidth=0.8,
                    ecolor=clr,
                    alpha=0.35 if is_gray else 1.0,
                    zorder=1 if is_gray else 3,
                )
            else:
                ax.scatter(
                    row_to_x(row),
                    y_val,
                    marker=mk,
                    color=clr,
                    edgecolor="black",
                    linewidths=0.8,
                    s=marker_size_scatter,
                    alpha=0.35 if is_gray else 1.0,
                    zorder=1 if is_gray else 3,
                )
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="stability",
        )

    ax.set_xticks(tick_positions)
    if x_axis_mode in ("dataset", "cell_line"):
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(tick_labels)
    # Add edge padding so jittered points aren't clipped
    if tick_positions and x_axis_mode != "dataset":
        pad = 0.5
        ax.set_xlim(tick_positions[0] - pad, tick_positions[-1] + pad)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)

    return fig


def plot_cross_dataset_summaries(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_bootstrap_dataframe_manifest: DataframeManifest,
    output_dir: Path,
    bootstrap_threshold: float = 0.4,
    column_names: list[ColumnName.DiffAEData | ColumnName.OpticalFlow | StrEnum] | None = None,
    x_axis_mode: Literal[
        "dataset", "shear_stress_numeric", "shear_stress_categorical", "cell_line"
    ] = "dataset",
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    dataset_order: list[str] | None = None,
    stable_only: bool = True,
    jitter_width: float = 0.1,
) -> None:
    """Create a plot of cross-dataset summary visualizations for
    observable fixed-point locations vs shear-stress plots
        - polar angle, polar radius, rho
        - maps mean optical-flow features onto fixed points as binned means for given
            polar angle/radius/rho bins

    Parameters
    ----------
    dataset_names
        List of dataset names to include in the summaries.
    feature_dataframe_manifest
        Manifest containing per-dataset feature dataframe locations.
    fixed_points_bootstrap_dataframe_manifest
        Manifest containing per-dataset fixed-points and confidence intervals found from
        precomputed bootstrapping.
    output_dir
        Directory where the figures are saved.
    column_names
        List of column names to plot in the fixed-point vs shear-stress plot.
        If None, defaults to polar angle, polar radius, PC3 flipped, migration coherence, and speed.
    x_axis_mode
        Controls x-axis layout of the fixed-point vs shear-stress plot.
        See :func:`plot_fixed_points_vs_shear_stress` for details.
    figure_size
        Size of the output figure for the fixed point vs shear stress plot.
    dataset_order
        Optional list of dataset names specifying the desired x-axis order in the fixed point vs shear stress plot.
        If ``None``, falls back to sorting by shear stress.
    stable_only
        If ``True``, only fixed points classified as stable are included in the
        fixed point vs shear stress plot.
    jitter_width
        Horizontal jitter applied to overlapping points sharing the same
        x-axis position.  Larger values spread points further apart.
    """
    if column_names is None:
        column_names = [
            ColumnName.DiffAEData.POLAR_ANGLE,
            ColumnName.DiffAEData.POLAR_RADIUS,
            ColumnName.DiffAEData.PC3_FLIPPED,
            ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
            ColumnName.OpticalFlow.SPEED_MEAN,
        ]

    optical_flow_features = [
        ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
        ColumnName.OpticalFlow.SPEED_MEAN,
    ]

    summary_stats: dict[str, list[dict[str, float | str]]] = {f: [] for f in optical_flow_features}
    df_fp_all_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found for dataset [ %s ]. Skipping.",
                dataset_name,
            )
            continue

        # Load, filter, and enrich the feature dataframe
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
        df = df_[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
        df_of = add_optical_flow_features(df_steady_state, datasets=[dataset_name])

        for flow_condition in dataset_config.flow_conditions:
            df_flow = filter_dataframe_by_flow_condition(df_of, dataset_config, flow_condition)
            plot_label = f"{dataset_name} ({round(flow_condition.shear_stress)} dyn/cm$^2$)"

            # Summary stats per optical flow feature
            for feature_key in optical_flow_features:
                mean_ft = df_flow[feature_key].mean()
                summary_stats[feature_key].append(
                    {
                        "label": plot_label,
                        "shear_stress": round(flow_condition.shear_stress),
                        "mean": mean_ft,
                    }
                )

            # Fixed points with binned means for each feature
            try:
                fp_bootstrap_location = get_dataframe_location_for_dataset(
                    fixed_points_bootstrap_dataframe_manifest, dataset_name
                )
                df_bootstrap = load_dataframe(fp_bootstrap_location, delay=False)

                n_total = len(df_bootstrap)
                high_confidence_df = df_bootstrap[
                    df_bootstrap[ColumnName.BootstrapAnalysis.DETECTION_RATE] >= bootstrap_threshold
                ].copy()

                if high_confidence_df.empty:
                    logger.warning(
                        "No fixed points with bootstrap_detection_rate >= %.2f for dataset "
                        "[ %s ] (%d fixed points in total). Skipping plot for this dataset.",
                        bootstrap_threshold,
                        dataset_name,
                        n_total,
                    )
                    continue

                for feature_key in optical_flow_features:
                    df_flow_no_nan = df_flow.dropna(subset=[feature_key])
                    high_confidence_df = add_binned_mean_to_fixed_points(
                        high_confidence_df,
                        df_flow_no_nan,
                        fp_x_col=f"{ColumnName.DiffAEData.POLAR_ANGLE}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
                        fp_y_col=f"{ColumnName.DiffAEData.POLAR_RADIUS}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
                        fp_z_col=f"{ColumnName.DiffAEData.PC3_FLIPPED}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
                        binned_col=feature_key,
                        of_x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                        of_y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                        of_z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                    )

                if x_axis_mode == "cell_line":
                    cell_line_label = CELL_LINE_LABEL_MAP.get(
                        dataset_config.cell_lines[0], dataset_config.cell_lines[0]
                    )
                    high_confidence_df["cell_line_label"] = cell_line_label

                df_fp_all_list.append(high_confidence_df)
            except KeyError:
                logger.warning(
                    "No fixed point dataframe found for dataset [ %s ]. Skipping fixed points.",
                    dataset_name,
                )

    # --- Fixed-points vs shear stress ---
    df_fp_all = pd.concat(df_fp_all_list, ignore_index=True)
    df_fp_all = add_shear_stress_to_df(df_fp_all)

    # Plot all fixed-point variables in a single 1-row subplot
    n_panels = len(column_names)
    fig, axs = plt.subplots(
        1,
        n_panels,
        figsize=(figure_size[0], figure_size[1]),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    all_column_info = get_seg_feat_plot_args()
    for ax_i, var in zip(axs[0], column_names, strict=False):
        column_info = all_column_info.get(var)
        var_label: str = column_info["label"] if column_info else str(var)
        col_name: str = f"mean_{var}" if var in optical_flow_features else str(var)
        plot_fixed_points_vs_shear_stress(
            df_fp_all,
            col_name,
            var_label,
            dataset_order=dataset_order,
            x_axis_mode=x_axis_mode,
            figure_size=figure_size,
            stable_only=stable_only,
            ax=ax_i,
            jitter_width=jitter_width,
        )
    if x_axis_mode == "cell_line":
        fig.supxlabel("Cell Line", fontsize=FONTSIZE_MEDIUM, fontweight="bold")
    else:
        fig.supxlabel("Shear Stress (dyn/cm\u00b2)", fontsize=FONTSIZE_MEDIUM, fontweight="bold")

    # reduce spacing between axis labels and tick labels
    for ax in axs[0]:
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

    # add variables being used to fname
    fname = f"{'_'.join(column_names)}_fp_vs_shear_stress"

    save_plot_to_path(
        fig,
        output_dir,
        fname,
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )
