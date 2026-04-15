"""Methods for visualizing migration coherence metrics and their relationships to morphology dynamics."""

import logging
from pathlib import Path

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
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
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
    ylim: tuple[float, float] | None = None,
    by_dataset: bool = True,
    marker_size_scatter: int = 10,
    marker_size_legend: int = 5,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
) -> plt.Figure:
    """Make and save plot of one component of fixed points vs shear stress.

    Plot is either categorical by dataset (default) or numeric by shear stress
    value, as described in the `by_dataset` parameter.

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
    ylim
        Optional ``(ymin, ymax)`` limits for the y-axis.
    by_dataset
        If ``True`` (default), each dataset gets its own categorical x position,
        ordered by ``dataset_order`` (or by shear stress if not provided).
        Tick labels show ``"dataset_name (shear_stress)"``.
        If ``False``, x positions are the numeric shear-stress values and
        datasets with the same shear stress overlap.
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
        lambda s: round(float(s.split("-")[0])) if isinstance(s, str) else round(float(s))
    )

    # Order by specified dataset list, or fall back to shear stress sorting
    if dataset_order is not None:
        dataset_cat = pd.CategoricalDtype(categories=dataset_order, ordered=True)
        df_fp["dataset"] = df_fp["dataset"].astype(dataset_cat)
        df_fp = df_fp.sort_values("dataset")
    else:
        df_fp = df_fp.sort_values("shear_stress_numeric")

    # Build legend handles for fixed-point stability markers
    legend_handles = make_legend_handles_for_fixed_pts(
        fpt_stabilities=df_fp[STABILITY_COLUMN_NAME].unique().tolist(),
        marker_size=marker_size_legend,
    )

    if by_dataset:
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
    else:
        # Numeric x-axis: position by shear stress value
        row_to_x = lambda row: row["shear_stress_numeric"]  # noqa: E731
        unique_shear = sorted(df_fp["shear_stress_numeric"].unique())
        tick_positions = unique_shear
        tick_labels = [str(int(s)) for s in unique_shear]

    fig, ax = plt.subplots(figsize=figure_size)

    # Plot fixed points — draw gray (unknown stability) behind others
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

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(label, fontsize=10)
    ax.set_xlabel("shear stress (dyn/cm\u00b2)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="stability",
    )

    return fig


def plot_cross_dataset_summaries(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_dataframe_manifest: DataframeManifest,
    output_dir: Path,
    by_dataset: bool = True,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    dataset_order: list[str] | None = None,
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
    by_dataset
        If ``True`` (default), each dataset gets its own categorical x position in the fixed
        point vs shear stress plot, with tick labels showing dataset name and shear stress.
        If ``False``, x positions are the numeric shear-stress values and datasets with the
        same shear stress overlap.
    figure_size
        Size of the output figure for the fixed point vs shear stress plot.
    dataset_order
        Optional list of dataset names specifying the desired x-axis order in the fixed point vs shear stress plot.
        If ``None``, falls back to sorting by shear stress.
    """
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
            plot_label = f"{dataset_name} ({int(flow_condition.shear_stress)} dyn/cm$^2$)"

            # Summary stats per optical flow feature
            for feature_key in optical_flow_features:
                mean_ft = df_flow[feature_key].mean()
                summary_stats[feature_key].append(
                    {
                        "label": plot_label,
                        "shear_stress": int(flow_condition.shear_stress),
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

    # Plot feature-independent fixed-point variables once
    for var, label in [
        (ColumnName.DiffAEData.POLAR_ANGLE, "\u03b8"),
        (ColumnName.DiffAEData.POLAR_RADIUS, "r"),
        (ColumnName.DiffAEData.PC3_FLIPPED, "\u03c1"),
        (f"mean_{ColumnName.OpticalFlow.UNIT_VECTOR_MEAN}", "Migration Coherence"),
        (f"mean_{ColumnName.OpticalFlow.SPEED_MEAN}", "Mean Speed"),
    ]:
        fig = plot_fixed_points_vs_shear_stress(
            df_fp_all,
            var,
            label,
            dataset_order=dataset_order,
            ylim=None,
            by_dataset=by_dataset,
            figure_size=figure_size,
        )
        save_plot_to_path(
            fig, output_dir, f"fixed_points_{var}_vs_shear_stress", file_format=".svg"
        )
