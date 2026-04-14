"""Methods for visualizing migration coherence metrics and their relationships to morphology dynamics."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_flow_condition,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
    add_shear_stress_to_df,
)
from endo_pipeline.library.visualize.diffae_features.pplane import make_legend_handles_for_fixed_pts
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
from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_COLUMN_NAME,
    STABILITY_MARKER_DICT,
)

logger = logging.getLogger(__name__)


def plot_fixed_points_vs_shear_stress(
    df_fp: pd.DataFrame,
    variable: str,
    label: str,
    dataset_order: list[str] | None = None,
    ylimits: tuple[float, float] | None = None,
    x_axis_mode: Literal["dataset", "shear_stress_numeric", "shear_stress_categorical"] = "dataset",
    marker_size_scatter: int = 15,
    marker_size_legend: int = 5,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    stable_only: bool = True,
    ax: plt.Axes | None = None,
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
    df_fp["shear_stress_numeric"] = df_fp["shear_stress"].apply(
        lambda s: min(max(round(max(float(v) for v in str(s).split("-"))), 6), 21)
    )

    if stable_only:
        df_fp = df_fp[df_fp[STABILITY_COLUMN_NAME] == "stable"]
    else:
        legend_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=df_fp[STABILITY_COLUMN_NAME].unique().tolist(),
            marker_size=marker_size_legend,
        )

    # Order by specified dataset list, or fall back to shear stress sorting
    if dataset_order is not None:
        dataset_cat = pd.CategoricalDtype(categories=dataset_order, ordered=True)
        df_fp["dataset"] = df_fp["dataset"].astype(dataset_cat)
        df_fp = df_fp.sort_values("dataset")
    else:
        df_fp = df_fp.sort_values("shear_stress_numeric")

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
        jitter_map = _build_jitter_map(df_fp)
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
        jitter_map = _build_jitter_map(df_fp)
        row_to_x = lambda row: ss_to_pos[
            row["shear_stress_numeric"]
        ] + jitter_map.get(  # noqa: E731
            (row["dataset"], row["shear_stress_numeric"]), 0.0
        )
    else:
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure

    if stable_only:
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
        unique_datasets = list(df_fp["dataset"].unique())
        dataset_color_map = {
            ds: _COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)]
            for i, ds in enumerate(unique_datasets)
        }

        for _, row in df_fp.iterrows():
            ax.scatter(
                row_to_x(row),
                row[variable],
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
            stability = row[STABILITY_COLUMN_NAME]
            mk = STABILITY_MARKER_DICT.get(stability, "o")
            clr = STABILITY_COLOR_DICT.get(stability, "gray")
            is_gray = stability not in STABILITY_COLOR_DICT
            ax.scatter(
                row_to_x(row),
                row[variable],
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
    if x_axis_mode == "dataset":
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(tick_labels)
        # Add edge padding so jittered points aren't clipped
        if tick_positions:
            pad = 0.25
            ax.set_xlim(tick_positions[0] - pad, tick_positions[-1] + pad)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)

    return fig


def plot_cross_dataset_summaries(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_dataframe_manifest: DataframeManifest,
    output_dir: Path,
    column_names: list[ColumnName.DiffAEData | ColumnName.OpticalFlow] | None = None,
    x_axis_mode: Literal["dataset", "shear_stress_numeric", "shear_stress_categorical"] = "dataset",
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    dataset_order: list[str] | None = None,
    stable_only: bool = True,
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
    fixed_points_dataframe_manifest
        Manifest containing per-dataset fixed-point dataframe locations.
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
                fp_location = get_dataframe_location_for_dataset(
                    fixed_points_dataframe_manifest, dataset_name
                )
                fp_df = load_dataframe(fp_location, delay=False)
                check_required_columns_in_dataframe(
                    fp_df,
                    required_columns=[
                        *DYNAMICS_COLUMN_NAMES,
                        ColumnName.DATASET,
                        STABILITY_COLUMN_NAME,
                    ],
                )
                for feature_key in optical_flow_features:
                    df_flow_no_nan = df_flow.dropna(subset=[feature_key])
                    fp_df = add_binned_mean_to_fixed_points(
                        fp_df,
                        df_flow_no_nan,
                        x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                        y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                        z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                        binned_col=feature_key,
                    )
                df_fp_all_list.append(fp_df)
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
    )
    all_column_info = get_seg_feat_plot_args()
    for ax_i, var in zip(axs, column_names, strict=False):
        column_info = all_column_info.get(var)
        label = column_info["label"] if column_info else var
        if var in optical_flow_features:
            var = f"mean_{var}"
        limits = column_info["lims"] if column_info else None
        if limits is not None and limits[0] is not None and limits[1] is not None:
            # Add 5% padding to y-limits
            padding = 0.05 * (limits[1] - limits[0])
            limits = (limits[0] - padding, limits[1] + padding)
        else:
            limits = None
        plot_fixed_points_vs_shear_stress(
            df_fp_all,
            var,
            label,
            dataset_order=dataset_order,
            ylimits=limits,
            x_axis_mode=x_axis_mode,
            figure_size=figure_size,
            stable_only=stable_only,
            ax=ax_i,
        )
    fig.supxlabel("Shear Stress (dyn/cm\u00b2)", fontsize=FONTSIZE_MEDIUM, fontweight="bold")

    # reduce spacing between axis labels and tick labels
    for ax in axs:
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

    save_plot_to_path(
        fig,
        output_dir,
        "fixed_points_vs_shear_stress",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )
