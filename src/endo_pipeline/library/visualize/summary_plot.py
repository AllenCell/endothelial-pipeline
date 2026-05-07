"""Methods for visualizing migration coherence metrics and their relationships to morphology dynamics."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig, FlowCondition, load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_shear_stress,
    filter_dataframe_by_stability,
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
    add_shear_stress_to_df,
)
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    make_legend_handles_for_fixed_pts,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.summary_plot import (
    CELL_LINE_LABEL_MAP,
    COLOR_PALETTE,
    DATASET_COLOR_MAP,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


# --- Build jitter map (shared by numeric and categorical shear-stress modes) ---
def _build_jitter_map(df: pd.DataFrame, jitter_width: float = 0.1) -> dict[tuple, float]:
    jmap: dict[tuple, float] = {}
    for ss in df["flow_condition_shear_stress_bin"].unique():
        datasets_at_ss = df.loc[
            df["flow_condition_shear_stress_bin"] == ss, ColumnName.DATASET
        ].unique()
        n = len(datasets_at_ss)
        if n <= 1:
            offsets = [0.0]
        else:
            offsets = [jitter_width * (i / (n - 1) - 0.5) for i in range(n)]
        for ds, off in zip(datasets_at_ss, offsets, strict=False):
            jmap[(ds, ss)] = off
    return jmap


def _build_categorical_axis(
    df_fp: pd.DataFrame,
    group_col: str,
    jitter_width: float,
    tick_spacing: float = 1.0,
) -> tuple[Any, list[float], list[str]]:
    """Build row_to_x, tick_positions, and tick_labels for a categorical x-axis.

    Parameters
    ----------
    df_fp
        Dataframe with ``group_col`` already populated and sorted in desired order.
    group_col
        Column whose unique values define the x-axis categories.
    jitter_width
        Horizontal jitter spread for datasets sharing the same category.
    tick_spacing
        Distance between adjacent categories on the x-axis.

    Returns
    -------
    row_to_x
        Callable mapping a row to its x position.
    tick_positions
        List of x positions for axis ticks.
    tick_labels
        List of labels for each tick.
    """
    unique_categories = df_fp[group_col].unique()
    cat_to_pos = {cat: i * tick_spacing for i, cat in enumerate(unique_categories)}
    tick_positions = [i * tick_spacing for i in range(len(unique_categories))]
    tick_labels = [str(cat) for cat in unique_categories]
    jitter_map = _build_jitter_map(df_fp, jitter_width=jitter_width)
    row_to_x = lambda row: cat_to_pos[row[group_col]] + jitter_map.get(  # noqa: E731
        (row[ColumnName.DATASET], row["flow_condition_shear_stress_bin"]), 0.0
    )
    return row_to_x, tick_positions, tick_labels


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
        "dataset", "shear_stress_numeric", "shear_stress_categorical", "cell_line", "flow_switch"
    ] = "dataset",
    marker_size_scatter: int = 15,
    marker_size_legend: int = 5,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    stable_only: bool = True,
    ax: plt.Axes | None = None,
    jitter_width: float = 0.1,
    x_padding: float = 0.5,
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
        - ``"shear_stress_numeric"``: x positions are the numeric
          shear-stress values (binned), with jitter for datasets sharing the same value.
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
    figure_size
        Size of the output figure.
    stable_only
        If ``True``, only fixed points classified as stable are included in the
        plot, and colored by dataset.  If ``False``, all fixed points are
        included and colored by stability classification.
    ax
        Optional matplotlib Axes to plot on.  If ``None``, a new figure
        and axes are created.
    jitter_width
        Horizontal jitter applied to overlapping points sharing the same
        x-axis position.  Larger values spread points further apart.
    x_padding
        Additional horizontal padding added to the left and right edges of the plot
        to ensure jittered points aren't clipped.  Only applied for non-categorical
        x-axis modes.


    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    if stable_only:
        df_fp = filter_dataframe_by_stability(df_fp, stability_label=StabilityLabel.STABLE)
    else:
        legend_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=df_fp[ColumnName.VectorField.STABILITY].unique().tolist(),
            marker_size=marker_size_legend,
        )

    # Order by specified dataset list, or fall back to shear stress sorting
    if dataset_order is not None:
        dataset_cat = pd.CategoricalDtype(categories=dataset_order, ordered=True)
        df_fp[ColumnName.DATASET] = df_fp[ColumnName.DATASET].astype(dataset_cat)
        df_fp = df_fp.sort_values(ColumnName.DATASET)
    else:
        df_fp = df_fp.sort_values("flow_condition_shear_stress_bin")

    row_to_x: Any  # noqa: E731
    tick_positions: list[float]
    if x_axis_mode == "dataset":
        # Categorical x-axis: one tick per dataset, custom labels
        unique_datasets = df_fp[ColumnName.DATASET].unique()
        df_fp["_dataset_label"] = df_fp[ColumnName.DATASET].map(
            {
                d: f"{load_dataset_config(d).date} ({df_fp.loc[df_fp[ColumnName.DATASET] == d, 'flow_condition_shear_stress_bin'].iloc[0]})"
                for d in unique_datasets
            }
        )
        row_to_x, tick_positions, tick_labels = _build_categorical_axis(
            df_fp, "_dataset_label", jitter_width=jitter_width
        )
    elif x_axis_mode == "shear_stress_numeric":
        # Numeric x-axis: position by shear stress value, jittered by dataset
        unique_shear = sorted(df_fp["flow_condition_shear_stress_bin"].unique())
        tick_positions = unique_shear
        tick_labels = list(map(str, unique_shear))
        jitter_map = _build_jitter_map(df_fp, jitter_width=jitter_width)
        row_to_x = lambda row: row[
            "flow_condition_shear_stress_bin"
        ] + jitter_map.get(  # noqa: E731
            (row[ColumnName.DATASET], row["flow_condition_shear_stress_bin"]), 0.0
        )
    elif x_axis_mode == "shear_stress_categorical":
        # Evenly-spaced categorical ticks for each unique shear stress value
        row_to_x, tick_positions, tick_labels = _build_categorical_axis(
            df_fp, "flow_condition_shear_stress_bin", jitter_width=jitter_width, tick_spacing=0.5
        )
    elif x_axis_mode == "cell_line":
        # Order by Parental → Control → VE-Cad KD
        cell_line_order = sorted(
            df_fp["cell_line_label"].unique(),
            key=lambda x: (0 if x == "Parental" else 1 if x == "Control" else 2),
        )
        cell_line_dtype = pd.CategoricalDtype(categories=cell_line_order, ordered=True)
        df_fp["cell_line_label"] = df_fp["cell_line_label"].astype(cell_line_dtype)
        df_fp = df_fp.sort_values("cell_line_label")
        row_to_x, tick_positions, tick_labels = _build_categorical_axis(
            df_fp, "cell_line_label", jitter_width=jitter_width
        )
    elif x_axis_mode == "flow_switch":
        # Label: "<shear_stress_bin> (single flow)" or "<shear_stress_bin> (flow switch)"
        df_fp["flow_switch_label"] = df_fp.apply(
            lambda row: (
                f"{int(row['flow_condition_shear_stress_bin'])} (single flow)"
                if row["n_shear_stress_conditions"] == 1
                else f"{int(row['flow_condition_shear_stress_bin'])} (flow switch)"
            ),
            axis=1,
        )
        row_to_x, tick_positions, tick_labels = _build_categorical_axis(
            df_fp, "flow_switch_label", jitter_width=jitter_width
        )
    else:
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if stable_only:
        # Use global dataset color map for consistent colors across plots;
        # fall back to palette cycling for unknown datasets.
        unique_datasets_list = df_fp[ColumnName.DATASET].unique()
        dataset_color_map = {
            ds: DATASET_COLOR_MAP.get(ds, COLOR_PALETTE[i % len(COLOR_PALETTE)])
            for i, ds in enumerate(unique_datasets_list)
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
                    color=dataset_color_map[row[ColumnName.DATASET]],
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=marker_size_scatter**0.5,
                    capsize=2,
                    elinewidth=0.8,
                    ecolor=dataset_color_map[row[ColumnName.DATASET]],
                    zorder=3,
                )
            else:
                ax.scatter(
                    row_to_x(row),
                    y_val,
                    marker="o",
                    color=dataset_color_map[row[ColumnName.DATASET]],
                    edgecolor="black",
                    linewidths=0.5,
                    s=marker_size_scatter,
                    zorder=3,
                )

    else:
        # Color by stability
        for _, row in df_fp.iterrows():
            stability = row[ColumnName.VectorField.STABILITY]
            mk = FIXED_POINT_PLOT_STYLE[stability].marker
            clr = FIXED_POINT_PLOT_STYLE[stability].color
            is_gray = stability not in FIXED_POINT_PLOT_STYLE
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
    if x_axis_mode in ("dataset", "cell_line", "flow_switch"):
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(tick_labels)
    # Add edge padding so jittered points aren't clipped
    if tick_positions and x_axis_mode != "dataset":
        x_padding = x_padding
        ax.set_xlim(tick_positions[0] - x_padding, tick_positions[-1] + x_padding)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    if variable == ColumnName.DiffAEData.POLAR_ANGLE:
        ax.set_yticks(
            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
            labels=[
                f"0={Unicode.PI}",
                f"{Unicode.PI}/4",
                f"{Unicode.PI}/2",
                f"3{Unicode.PI}/4",
                f"{Unicode.PI}=0",
            ],
        )

    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)

    return fig


def _convert_polar_angle_to_nematic_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert polar angle to nematic order in the given dataframe.

    Converts polar angle columns (e.g. `"polar_angle"`,
    `"polar_angle_cluster_mean"`) to nematic order using the transformation:

    ..math::
        S = cos(2*theta)

    Also applies the chain rule to approximate confidence intervals for the
    nematic order based on the polar angle confidence intervals. For example,
    the upper CI for the nematic order is approximated as:

    ..math::
        CI_{upper}^S = S_{mean} + f'(theta_{mean}) * (theta_{CI_{upper}} - theta_{mean})
    """
    for column_suffix in [
        "",
        f"_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
    ]:
        df[f"{ColumnName.DiffAEData.NEMATIC_ORDER}{column_suffix}"] = df[
            f"{ColumnName.DiffAEData.POLAR_ANGLE}{column_suffix}"
        ].apply(lambda theta: (np.cos(2 * theta)))

    # Use chain rule to approximate transformed confidence intervals for nematic
    # order based on polar angle CIs:
    #    S_CI_upper = S_mean + f'(theta_mean) * (theta_CI_upper - theta_mean)
    #    S_CI_lower = S_mean + f'(theta_mean) * (theta_CI_lower - theta_mean)
    # where S_mean = cos(2*theta_mean) is the nematic order at the mean angle, and
    # f'(theta) = -2*sin(2*theta) is the derivative of the nematic order function.
    for ci_type in [ColumnName.BootstrapAnalysis.CI_LOWER, ColumnName.BootstrapAnalysis.CI_UPPER]:
        angle_mean_col = (
            f"{ColumnName.DiffAEData.POLAR_ANGLE}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"
        )
        angle_ci_col = f"{ColumnName.DiffAEData.POLAR_ANGLE}_{ci_type}"
        nematic_mean_col = (
            f"{ColumnName.DiffAEData.NEMATIC_ORDER}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}"
        )
        nematic_ci_col = f"{ColumnName.DiffAEData.NEMATIC_ORDER}_{ci_type}"

        for idx, row in df.iterrows():
            theta_mean = row[angle_mean_col]
            theta_ci = row[angle_ci_col]
            nematic_mean = row[nematic_mean_col]
            # Compute the derivative f'(theta) at the mean angle
            f_prime = -2 * np.sin(2 * theta_mean)
            # Approximate the nematic order CI using the chain rule
            df.at[idx, nematic_ci_col] = nematic_mean + f_prime * (theta_ci - theta_mean)

    return df


def _process_bootstrap_dataframe_for_plot(
    df_bootstrap: pd.DataFrame,
    df_features: pd.DataFrame,
    bootstrap_threshold: float,
    dataset_name: str,
    flow_condition: FlowCondition,
    optical_flow_features: list[ColumnName.OpticalFlow],
    convert_angle_to_nematic: bool,
    column_names: list[ColumnName.DiffAEData | ColumnName.OpticalFlow | StrEnum],
    x_axis_mode: Literal[
        "dataset", "shear_stress_numeric", "shear_stress_categorical", "cell_line", "flow_switch"
    ],
    dataset_config: DatasetConfig,
) -> pd.DataFrame:
    # Fixed points with binned means for each feature
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
        return pd.DataFrame()

    for feature_key in optical_flow_features:
        df_flow_no_nan = df_features.dropna(subset=[feature_key])
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

    high_confidence_df["n_shear_stress_conditions"] = len(dataset_config.flow_conditions)
    high_confidence_df["flow_condition_shear_stress_bin"] = flow_condition.shear_stress_bin

    if convert_angle_to_nematic and ColumnName.DiffAEData.POLAR_ANGLE in column_names:
        # Add nematic order columns to the dataframe
        high_confidence_df = _convert_polar_angle_to_nematic_order(high_confidence_df)

    if x_axis_mode == "cell_line":
        cell_line_label = CELL_LINE_LABEL_MAP.get(
            dataset_config.cell_lines[0], dataset_config.cell_lines[0]
        )
        high_confidence_df["cell_line_label"] = cell_line_label

    return high_confidence_df


def plot_cross_dataset_summaries(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_bootstrap_dataframe_manifest: DataframeManifest,
    output_dir: Path,
    bootstrap_threshold: float = 0.4,
    column_names: list[ColumnName.DiffAEData | ColumnName.OpticalFlow | StrEnum] | None = None,
    x_axis_mode: Literal[
        "dataset", "shear_stress_numeric", "shear_stress_categorical", "cell_line", "flow_switch"
    ] = "dataset",
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    dataset_order: list[str] | None = None,
    stable_only: bool = True,
    jitter_width: float = 0.1,
    x_padding: float = 0.5,
    subplot_layout: Literal["horizontal", "vertical"] = "horizontal",
    convert_angle_to_nematic: bool = True,
) -> None:
    """
    Create a plot of cross-dataset summary visualizations for observable
    fixed-point locations vs shear-stress plots
        - polar angle, polar radius, rho
        - maps mean optical-flow features onto fixed points as binned means for
          given polar angle/radius/rho bins

    **Subplot layout specification**

    The `subplot_layout` parameter controls the arrangement of multiple panels
    when plotting multiple variables. The options are:
        - `"horizontal"`: panels side-by-side in a single row (1xn).
        - `"vertical"`: panels stacked vertically with a shared x-axis (nx1).
           Only the bottom panel shows x-axis tick labels.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in the summaries.
    feature_dataframe_manifest
        Manifest containing per-dataset feature dataframe locations.
    fixed_points_bootstrap_dataframe_manifest
        Manifest containing per-dataset fixed-points and confidence intervals
        found from precomputed bootstrapping.
    output_dir
        Directory where the figures are saved.
    column_names
        List of column names to plot in the fixed-point vs shear-stress plot. If
        None, defaults to polar angle, polar radius, PC3 flipped, migration
        coherence, and speed.
    x_axis_mode
        Controls x-axis layout of the fixed-point vs shear-stress plot. See
        :func:`plot_fixed_points_vs_shear_stress` for details.
    figure_size
        Size of the output figure for the fixed point vs shear stress plot.
    dataset_order
        Optional list of dataset names specifying the desired x-axis order in
        the fixed point vs shear stress plot. If `None`, falls back to sorting
        by shear stress.
    stable_only
        If `True`, only fixed points classified as stable are included in the
        fixed point vs shear stress plot.
    jitter_width
        Horizontal jitter applied to overlapping points sharing the same x-axis
        position.  Larger values spread points further apart.
    x_padding
        Additional horizontal padding added to the left and right edges of the
        fixed point vs shear stress plot to ensure jittered points aren't
        clipped.  Only applied for non-categorical x-axis modes.
    subplot_layout
        Layout direction for multiple column panels.
    convert_angle_to_nematic
        If `True`, converts polar angle to nematic order (cos(2*theta)) before
        plotting.
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

    df_fp_all_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found for dataset [ %s ]. Skipping.",
                dataset_name,
            )
            continue
        elif dataset_name not in fixed_points_bootstrap_dataframe_manifest.locations:
            logger.warning(
                "No fixed point bootstrap dataframe found for dataset [ %s ]. Skipping.",
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

        # Load bootstrap results
        df_bootstrap = load_dataframe(
            fixed_points_bootstrap_dataframe_manifest.locations[dataset_name], delay=False
        )

        # For flow_switch mode with multiple conditions, skip the first (pre-switch) condition
        if x_axis_mode == "flow_switch" and len(dataset_config.flow_conditions) > 1:
            flow_conditions_to_process = dataset_config.flow_conditions[1:]
        else:
            flow_conditions_to_process = dataset_config.flow_conditions

        for flow_condition in flow_conditions_to_process:
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
                convert_angle_to_nematic,
                column_names,
                x_axis_mode,
                dataset_config,
            )
            if not df_fp.empty:
                df_fp_all_list.append(df_fp)

    # update column names to pass to plotting function based on whether we're
    # plotting polar angle or nematic order
    if convert_angle_to_nematic and ColumnName.DiffAEData.POLAR_ANGLE in column_names:
        column_names = [
            ColumnName.DiffAEData.NEMATIC_ORDER if col == ColumnName.DiffAEData.POLAR_ANGLE else col
            for col in column_names
        ]

    # --- Fixed-points vs shear stress ---
    df_fp_all = pd.concat(df_fp_all_list, ignore_index=True)
    df_fp_all = add_shear_stress_to_df(df_fp_all)

    # Plot all fixed-point variables
    n_panels = len(column_names)
    if subplot_layout == "vertical":
        fig, axs = plt.subplots(
            n_panels,
            1,
            figsize=(figure_size[0], figure_size[1] * n_panels),
            layout="constrained",
            squeeze=False,
        )
        axes_list = [axs[i][0] for i in range(n_panels)]
    else:
        fig, axs = plt.subplots(
            1,
            n_panels,
            figsize=(figure_size[0], figure_size[1]),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        axes_list = list(axs[0])
    for ax_i, var in zip(axes_list, column_names, strict=False):
        var_label = COLUMN_METADATA[var].label or str(var)
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
            x_padding=x_padding,
        )
    if x_axis_mode == "cell_line":
        fig.supxlabel("Cell Line", fontsize=FONTSIZE_MEDIUM, fontweight="bold")
    elif x_axis_mode == "dataset":
        fig.supxlabel(
            f"Dataset Date (Shear Stress dyn/cm{Unicode.SQUARED})",
            fontsize=FONTSIZE_MEDIUM,
            fontweight="bold",
        )
    else:
        fig.supxlabel(
            f"Shear Stress (dyn/cm{Unicode.SQUARED})", fontsize=FONTSIZE_MEDIUM, fontweight="bold"
        )

    # reduce spacing between axis labels and tick labels
    for ax in axes_list:
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

    # For vertical layout, sync x-axes and hide tick labels on all but the bottom panel
    if subplot_layout == "vertical":
        # Match xlims across all panels
        all_xlims = [ax.get_xlim() for ax in axes_list]
        shared_xlim = (min(lo for lo, _ in all_xlims), max(hi for _, hi in all_xlims))
        for ax in axes_list:
            ax.set_xlim(shared_xlim)
        # Hide tick labels (but keep tick marks) on upper panels
        for ax in axes_list[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

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
