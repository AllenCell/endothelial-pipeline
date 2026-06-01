"""Methods for plotting summaries of values across datasets."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import join_sorted_strings, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_shear_stress,
    filter_dataframe_by_stability,
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    make_legend_handles_for_fixed_pts,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA, ColumnMetadata
from endo_pipeline.settings.column_names import ColumnName, ColumnNameType
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
    POLAR_ANGLE_PERIOD,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
from endo_pipeline.settings.summary_plot import (
    CELL_LINE_LABEL_MAP,
    COLOR_PALETTE,
    DATASET_COLOR_MAP,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)

SummaryPlotAxisMode = Literal["dataset", "shear_stress", "cell_line"]
"""Type hint for summary plot axis modes."""

SummaryPlotStyleMode = Literal["dataset", "stability"]
"""Type hint for summary plot style modes."""

DEFAULT_SUMMARY_COLUMN_NAMES: list[ColumnNameType] = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ColumnName.OpticalFlow.SPEED_MEAN,
]
"""List of default summary plot column names."""

BINNED_MEAN_FEATURES = [
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ColumnName.OpticalFlow.SPEED_MEAN,
]
"""List of columns where mean is calculated for bin around the fixed point."""

SUMMARY_MODE_X_AXIS_SUP_LABELS: dict[SummaryPlotAxisMode, str] = {
    "dataset": f"Dataset Date (Shear Stress dyn/cm{Unicode.SQUARED})",
    "shear_stress": f"Shear Stress (dyn/cm{Unicode.SQUARED})",
    "cell_line": "Cell Line",
}
"""Mapping of summary plot axis mode to X axis super labels."""

SUMMARY_MODE_COLUMN_NAME: dict[SummaryPlotAxisMode, str] = {
    "dataset": ColumnName.DATASET,
    "shear_stress": "_shear_stress_category",
    "cell_line": "_cell_line_category",
}
"""Mapping of summary plot axis mode to column names."""


def _convert_polar_angle_to_nematic_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert polar angle to nematic order in the given dataframe.

    Converts polar angle columns (e.g. `"polar_angle"`,
    `"polar_angle_cluster_mean"`) to nematic order using the transformation:

    ..math::
        S = cos(2*theta)

    Also applies the chain rule for error propagation to approximate confidence
    intervals for the nematic order based on the polar angle confidence
    intervals. For example, the upper CI for the nematic order is approximated
    as:

    ..math::
        CI_{upper}^S = S_{mean} + |f'(theta_{mean})| * (theta_{CI_{upper}} -
        theta_{mean})
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
    #    S_CI_upper = S_mean + |f'(theta_mean)| * (theta_CI_upper - theta_mean)
    #    S_CI_lower = S_mean + |f'(theta_mean)| * (theta_CI_lower - theta_mean)
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
            f_prime = np.abs(-2 * np.sin(2 * theta_mean))
            # Compute the circular/unwrapped difference between the CI angle and
            # the mean angle to avoid issues with angle wrapping around the periodic boundary.
            circ_diff = np.diff(np.unwrap([theta_mean, theta_ci], period=POLAR_ANGLE_PERIOD))[-1]
            # Approximate the nematic order CI using the chain rule
            nematic_bound = nematic_mean + f_prime * circ_diff
            # clip the nematic CI to the valid range of [-1, 1]
            if ci_type == ColumnName.BootstrapAnalysis.CI_LOWER:
                df.at[idx, nematic_ci_col] = max(nematic_bound, -1)
            else:
                df.at[idx, nematic_ci_col] = min(nematic_bound, 1)

    return df


def _build_color_by_column_mappable(
    df: pd.DataFrame,
    color_by_column: ColumnNameType,
) -> tuple[cm.ScalarMappable | None, str, ColumnMetadata | None]:
    """
    Build a ScalarMappable for continuous coloring by a dataframe column.

    Returns the scalar mappable (or None if the column isn't in df), the
    resolved column name (with ``mean_`` prefix if needed), and the column
    metadata object for labeling the colorbar.
    """
    color_column_metadata = COLUMN_METADATA.get(color_by_column)
    if color_by_column in BINNED_MEAN_FEATURES:
        color_by_column = f"mean_{color_by_column}"
    if color_by_column in df.columns:
        cmap = mcolors.LinearSegmentedColormap.from_list("cyan_magenta", ["cyan", "magenta"])
        norm = mcolors.Normalize(vmin=df[color_by_column].min(), vmax=df[color_by_column].max())
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
    else:
        scalar_mappable = None
    return scalar_mappable, color_by_column, color_column_metadata


def _get_tick_labels(
    axis_mode: SummaryPlotAxisMode,
    unique_categories: list[str],
    dataset_configs: dict,
) -> list[str]:
    """Get tick labels for the x axis based on the selected axis mode."""
    if axis_mode == "dataset":
        return [
            f"{dataset_configs[cat].date} ({dataset_configs[cat].flow_conditions[-1].shear_stress_bin})"
            for cat in unique_categories
        ]
    elif axis_mode == "cell_line":
        return [CELL_LINE_LABEL_MAP[cat] for cat in unique_categories]
    return unique_categories


def _plot_cross_dataset_summary_for_column(
    df: pd.DataFrame,
    ax: Axes,
    column_name: ColumnNameType,
    category_order: list[str] | None = None,
    axis_mode: SummaryPlotAxisMode = "dataset",
    style_mode: SummaryPlotStyleMode = "dataset",
    marker_size_plot: float = 4.2,
    marker_size_legend: int = 4,
    jitter_width: float = 0.05,
    set_y_lims: bool = False,
    color_by_column: ColumnNameType | None = None,
) -> None:
    """
    Plot cross dataset summary for given column name and summary mode.

    Parameters
    ----------
    df
        Dataframe containing features for summary plot.
    ax
        Axes instance for plotting summary.
    column_name
        Feature column name.
    category_order
        Optional order for the categories in the plot.
    axis_mode
        Select the axis mode for the summary plot.
    style_mode
        Select the style mode for the summary plot.
    marker_size_plot
        Size of the markers in the plot.
    marker_size_legend
        Size of the markers in the legend.
    jitter_width
        Width of the jitter applied to points in the same category bin.
    set_y_lims
        True to set y limits based on column metadata, False otherwise.
    color_by_column
        Optional column name whose values are mapped to a continuous
        cyan-to-magenta colormap. When provided, overrides the discrete
        coloring from ``style_mode`` while preserving all other behavior
        (axis mode, markers, error bars, etc.).
    """

    # Load dataset configs for all unique datasets in summary data
    unique_datasets = df[ColumnName.DATASET].unique()
    dataset_configs = {dataset: load_dataset_config(dataset) for dataset in unique_datasets}

    # Create color and marker mapping based on selected style mode
    if style_mode == "dataset":
        style_column = ColumnName.DATASET
        color_map = {
            dataset: DATASET_COLOR_MAP.get(dataset, COLOR_PALETTE[i % len(COLOR_PALETTE)])
            for i, dataset in enumerate(unique_datasets)
        }
        marker_map = dict.fromkeys(unique_datasets, "o")
    elif style_mode == "stability":
        style_column = ColumnName.VectorField.STABILITY
        color_map = {key: style.color for key, style in FIXED_POINT_PLOT_STYLE.items()}
        marker_map = {key: style.marker for key, style in FIXED_POINT_PLOT_STYLE.items()}
    else:
        raise ValueError(f"Summary plot style mode '{style_mode}' is not supported")

    # Build continuous colormap if color_by_column is provided (overrides style_mode colors)
    if color_by_column is not None:
        scalar_mappable, color_by_column, color_column_metadata = _build_color_by_column_mappable(
            df, color_by_column
        )
    else:
        scalar_mappable = None
        color_column_metadata = None

    # Get the category column name for the selected axis mode
    category_column = SUMMARY_MODE_COLUMN_NAME.get(axis_mode)
    if category_column is None:
        raise ValueError(f"Summary plot axis mode '{axis_mode}' is not supported")

    # Add category column for selected axis mode based on dataset (if needed)
    if axis_mode == "shear_stress":
        shear_stress_map = {
            dataset_config.name: dataset_config.flow_conditions[-1].shear_stress_bin
            for dataset_config in dataset_configs.values()
        }
        df[category_column] = df[ColumnName.DATASET].map(shear_stress_map)
    elif axis_mode == "cell_line":
        cell_line_map = {
            dataset_config.name: dataset_config.cell_lines[0]
            for dataset_config in dataset_configs.values()
        }
        df[category_column] = df[ColumnName.DATASET].map(cell_line_map)

    # If category order is provided, remap the data type to preserve given order
    if category_order is not None:
        dataset_category = pd.CategoricalDtype(categories=category_order, ordered=True)
        df[category_column] = df[category_column].astype(dataset_category)

    # Sort the data by the category and get final order for categories
    df = df.sort_values(category_column)
    unique_categories = list(df[category_column].unique())

    # Get category labels for the selected axis mode
    tick_labels = _get_tick_labels(axis_mode, unique_categories, dataset_configs)

    # Get column metadata from base name and then adjust column name if the
    # column is a binned mean feature
    column_metadata = COLUMN_METADATA[column_name]
    column_name = f"mean_{column_name}" if column_name in BINNED_MEAN_FEATURES else column_name

    # Get column names for confidence interval
    ci_lower_col = f"{column_name}_{ColumnName.BootstrapAnalysis.CI_LOWER}"
    ci_upper_col = f"{column_name}_{ColumnName.BootstrapAnalysis.CI_UPPER}"

    # If both confidence interval columns exist, calculate upper and lower bounds
    if ci_lower_col in df and ci_upper_col in df:
        df["_lower_bound"] = (df[column_name] - df[ci_lower_col]).clip(lower=0)
        df["_upper_bound"] = (df[ci_upper_col] - df[column_name]).clip(lower=0)
    else:
        df["_lower_bound"] = 0
        df["_upper_bound"] = 0

    # Iterate through each category to plot points and (if available) error
    # bars. Points are colored based on selected style mode while position on
    # the x axis is determined by the selected axis mode
    for category, category_df in df.groupby(category_column, observed=True):
        # Get position for the dataset based on index in category list
        index = unique_categories.index(category)

        # Assign jitter offsets per-dataset so points from the same dataset
        # share the same horizontal position within a category bin
        unique_datasets_in_category = category_df[ColumnName.DATASET].unique()
        num_datasets = len(unique_datasets_in_category)
        dataset_offsets = (
            {unique_datasets_in_category[0]: 0}
            if num_datasets == 1
            else dict(
                zip(
                    unique_datasets_in_category,
                    np.linspace(-jitter_width, jitter_width, num_datasets),
                    strict=True,
                )
            )
        )
        x_values = [index + dataset_offsets[ds] for ds in category_df[ColumnName.DATASET]]

        # Get y values based on feature column
        y_values = category_df[column_name]

        # Get marker and color based on style column or color_by_column
        if scalar_mappable is not None:
            colors = [scalar_mappable.to_rgba(val) for val in category_df[color_by_column]]
        else:
            colors = list(category_df[style_column].map(color_map))
        markers = [marker_map[col] for col in category_df[style_column]]

        # Get lower and upper bounds for points in the category
        lower_bounds = category_df["_lower_bound"]
        upper_bounds = category_df["_upper_bound"]

        for x, y, color, marker, lower, upper in zip(
            x_values, y_values, colors, markers, lower_bounds, upper_bounds, strict=True
        ):
            ax.errorbar(
                x,
                y,
                yerr=[[lower], [upper]],
                fmt=marker,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=marker_size_plot,
                capsize=2.3,
                elinewidth=0.8,
                ecolor="black",
                zorder=3,
            )

    # Add colorbar if color_by_column override is active
    if scalar_mappable is not None:
        assert color_by_column is not None
        cbar_label = (
            color_column_metadata.label
            if color_column_metadata and color_column_metadata.label
            else str(color_by_column)
        )
        cbar = ax.figure.colorbar(scalar_mappable, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=FONTSIZE_SMALL)

    # Include legend if using stability style mode (only when not overridden)
    if style_mode == "stability" and scalar_mappable is None:
        legend_handles = make_legend_handles_for_fixed_pts(
            fpt_stabilities=df[ColumnName.VectorField.STABILITY].unique().tolist(),
            marker_size=marker_size_legend,
        )
        ax.legend(handles=legend_handles, fontsize=FONTSIZE_SMALL)

    # Set x axis ticks to category positions
    category_positions = range(len(unique_categories))
    ax.set_xticks(category_positions)

    # Rotate x axis category labels if the label is more than 10 characters
    if len(str(tick_labels[0])) > 5:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(tick_labels)

    # Set the x axis limits with padding to avoid cutting of jittered points
    x_padding = 0.5
    ax.set_xlim(category_positions[0] - x_padding, category_positions[-1] + x_padding)

    # Set y ticks if they are available for the given column
    if column_metadata.ticks is not None:
        ax.set_yticks(column_metadata.ticks)

    # Set y labels if they are available for the given column
    if column_metadata.tick_labels is not None:
        ax.set_yticklabels(column_metadata.tick_labels)

    # Set y limits if they are available for the given column
    if set_y_lims and column_metadata.limits is not None:
        y_min = df[category_column].min() if column_metadata.min == "min" else column_metadata.min
        y_max = df[category_column].max() if column_metadata.max == "max" else column_metadata.max
        ax.set_ylim(y_min, y_max)

    # Add y axis label and grid lines
    y_axis_label = column_metadata.label or str(column_name)
    ax.set_ylabel(y_axis_label)
    ax.grid(axis="y", alpha=0.3)


def plot_cross_dataset_summaries(
    df: pd.DataFrame,
    output_dir: Path,
    column_names: list[ColumnNameType] | None = None,
    axis_mode: SummaryPlotAxisMode = "dataset",
    style_mode: SummaryPlotStyleMode = "dataset",
    category_order: list[str] | None = None,
    subplot_layout: Literal["horizontal", "vertical"] = "horizontal",
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 3),
    jitter_width: float = 0.05,
    convert_angle_to_nematic: bool = True,
    set_y_lims: bool = False,
    color_by_column: ColumnNameType | None = None,
) -> Path:
    """
    Plot cross dataset summaries for given columns in selected plot mode.

    **Summary plot axis modes**

    The `axis_mode` parameter controls the categories used on the x axis of the
    summary plot. The options are:

    - `dataset` = each dataset is a separate category
    - `shear_stress` = each shear stress bin is a separate category (multiple
      datasets may fall into the same shear stress bin)
    - `cell_line` = each cell line is a separate category (multiple datasets may
      fall into the same cell line bin)

    **Summary plot style modes**

    The `style_mode` parameter controls how the points in the summary plot are
    styled. The options are:

    - `dataset` = each point is uniquely colored by dataset
    - `stability` = each point is assigned a color and marker shape based on
      stability (dataframe must a `ColumnName.VectorField.STABILITY` column)

    **Subplot layout specification**

    The `subplot_layout` parameter controls the arrangement of multiple panels
    when plotting multiple columns. The options are:

    - `horizontal` = panels stacked horizontally in a single row (1 x n)
    - `vertical` = panels stacked vertically with a shared x-axis (n x 1) where
      only the bottom panel shows x-axis tick labels

    Parameters
    ----------
    df
        Dataframe containing features for summary plot.
    output_dir
        Output directory to save plot.
    column_names
        List of feature column names.
    axis_mode
        Select the axis mode for the summary plot.
    style_mode
        Select the style mode for the summary plot.
    category_order
        Optional order for the categories in the plot.
    subplot_layout
        Layout direction for summary plots of multiple columns.
    figure_size
        Size of output figure.
    jitter_width
        Width of the jitter applied to points in the same category bin.
    convert_angle_to_nematic
        True to swap polar angle column to nemetic order column.
    color_by_column
        Optional column name whose values are mapped to a continuous
        cyan-to-magenta colormap. When provided, overrides the discrete
        coloring from ``style_mode``.

    Returns
    -------
    :
        Path to saved summary plot.
    """

    # If specific column names are not provided, use defaults
    if column_names is None:
        column_names = DEFAULT_SUMMARY_COLUMN_NAMES

    # If converting polar angle to nematic order, swap the column name
    if convert_angle_to_nematic:
        column_names = [
            ColumnName.DiffAEData.NEMATIC_ORDER if col == ColumnName.DiffAEData.POLAR_ANGLE else col
            for col in column_names
        ]

    # Build figure layout with one subplot for each column name
    n_panels = len(column_names)
    if subplot_layout == "vertical":
        fig, axes_ = plt.subplots(
            n_panels,
            1,
            figsize=(figure_size[0], figure_size[1] * n_panels),
            layout="constrained",
            squeeze=False,
        )
        axes = [axes_[i][0] for i in range(n_panels)]
    elif subplot_layout == "horizontal":
        fig, axes_ = plt.subplots(
            1,
            n_panels,
            figsize=(figure_size[0], figure_size[1]),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        axes = list(axes_[0])
    else:
        raise ValueError(f"Subplot layout '{subplot_layout}' is not supported")

    # Iterate through each column and plot dataset summary
    for ax, column_name in zip(axes, column_names, strict=True):
        _plot_cross_dataset_summary_for_column(
            df=df,
            ax=ax,
            category_order=category_order,
            column_name=column_name,
            axis_mode=axis_mode,
            style_mode=style_mode,
            jitter_width=jitter_width,
            set_y_lims=set_y_lims,
            color_by_column=color_by_column,
        )

    # Add super x axis label
    fig.supxlabel(
        SUMMARY_MODE_X_AXIS_SUP_LABELS[axis_mode],
        fontsize=FONTSIZE_MEDIUM,
        fontweight="bold",
    )

    # Reduce spacing between axis labels and tick labels
    for ax in axes:
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

    # For vertical layout, match the x limits across all panels and hide the
    # x tick labels for all but the bottom panel.
    if subplot_layout == "vertical":
        all_xlims = [ax.get_xlim() for ax in axes]
        shared_xlim = (min(lo for lo, _ in all_xlims), max(hi for _, hi in all_xlims))
        for ax in axes:
            ax.set_xlim(shared_xlim)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

    # Save figure with name including all column names
    column_name_str = [str(column_name) for column_name in column_names]
    figure_name = f"summary_{axis_mode}_{join_sorted_strings(column_name_str)}"
    save_plot_to_path(fig, output_dir, figure_name, file_format=".svg", tight_layout=False)

    return output_dir / f"{figure_name}.svg"


def build_dataframe_for_fixed_point_dataset_summary(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    bootstrap_dataframe_manifest: DataframeManifest,
    column_names: list[ColumnNameType] | None = None,
    bootstrap_threshold: float = 0.4,
    convert_angle_to_nematic: bool = True,
    stable_only: bool = True,
) -> pd.DataFrame:
    """
    Build dataframe for plotting fixed point dataset summary.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in summary.
    feature_dataframe_manifest
        Dataframe manifest for feature values.
    bootstrap_dataframe_manifest
        Dataframe manifest for bootstrapped fixed points.
    column_names
        List of feature column names to include in summary.
    bootstrap_threshold
        Threshold for high confidence fixed points.
    convert_angle_to_nematic
        True to convert polar angle to nematic order.
    stable_only
        True to only include stable fixed points.

    Returns
    -------
    :
        Dataframe to use for making cross dataset summary plots.
    """

    if column_names is None:
        column_names = DEFAULT_SUMMARY_COLUMN_NAMES

    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
    columns_to_bin = {
        "fp_x_col": f"{ColumnName.DiffAEData.POLAR_ANGLE}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
        "fp_y_col": f"{ColumnName.DiffAEData.POLAR_RADIUS}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
        "fp_z_col": f"{ColumnName.DiffAEData.PC3_FLIPPED}_{ColumnName.BootstrapAnalysis.CLUSTER_MEAN}",
        "of_x_col": ColumnName.DiffAEData.POLAR_ANGLE,
        "of_y_col": ColumnName.DiffAEData.POLAR_RADIUS,
        "of_z_col": ColumnName.DiffAEData.PC3_FLIPPED,
    }

    df_fixed_points_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        # Skip including dataset if no feature dataframe is found
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found for dataset [ %s ]. Skipping.",
                dataset_name,
            )
            continue

        # Skip including dataset if no fixed point bootstrap dataframe is found
        if dataset_name not in bootstrap_dataframe_manifest.locations:
            logger.warning(
                "No fixed point bootstrap dataframe found for dataset [ %s ]. Skipping.",
                dataset_name,
            )
            continue

        # Load dataset config for dataset
        dataset_config = load_dataset_config(dataset_name)

        # Load selected columns from feature dataframe
        df_delay = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_features = df_delay[columns_to_compute].compute()
        df_features = filter_dataframe_to_steady_state(df_features, dataset_config)
        df_features = add_optical_flow_features(df_features, datasets=[dataset_name])

        # Load bootstrap results dataframe
        df_bootstrap = load_dataframe(
            bootstrap_dataframe_manifest.locations[dataset_name], delay=False
        )

        # If the dataset has multiple flow conditions, only use final condition
        flow_condition = dataset_config.flow_conditions[-1]
        df_features_flow = filter_dataframe_to_flow_condition_by_timepoint(
            df_features, dataset_config, flow_condition
        )
        df_bootstrap_flow = filter_dataframe_by_shear_stress(
            df_bootstrap, flow_condition.shear_stress
        )

        # Filter bootstrap dataframe to only include high confidence values
        df_fixed_points = df_bootstrap_flow[
            df_bootstrap_flow[ColumnName.BootstrapAnalysis.DETECTION_RATE] >= bootstrap_threshold
        ].copy()

        if df_fixed_points.empty:
            logger.warning(
                "No fixed points with bootstrap_detection_rate >= %.2f for dataset "
                "[ %s ] with shear stress [ %.2f ] (%d fixed points in total). "
                "Skipping plot for this dataset.",
                bootstrap_threshold,
                dataset_name,
                flow_condition.shear_stress,
                len(df_bootstrap_flow),
            )
            return pd.DataFrame()

        for feature_key in BINNED_MEAN_FEATURES:
            df_features_flow_no_nan = df_features_flow.dropna(subset=[feature_key])
            df_fixed_points = add_binned_mean_to_fixed_points(
                df_fixed_points,
                df_features_flow_no_nan,
                binned_col=feature_key,
                **columns_to_bin,  # type: ignore[arg-type]
            )

        if convert_angle_to_nematic and ColumnName.DiffAEData.POLAR_ANGLE in column_names:
            df_fixed_points = _convert_polar_angle_to_nematic_order(df_fixed_points)

        if df_fixed_points.empty:
            continue

        if stable_only:
            df_fixed_points = filter_dataframe_by_stability(
                df_fixed_points, stability_label=StabilityLabel.STABLE
            )

        df_fixed_points_list.append(df_fixed_points)

    return pd.concat(df_fixed_points_list, ignore_index=True)


def build_dataframe_for_first_passage_time_dataset_summary(
    dataset_names: list[str],
    first_passage_time_manifest: DataframeManifest,
) -> pd.DataFrame:
    """
    Build dataframe for plotting first passage time dataset summary.

    Parameters
    ----------
    dataset_names
        List of dataset names to include in summary.
    first_passage_time_manifest
        Dataframe manifest for first passage times.

    Returns
    -------
    :
        Dataframe to use for making cross dataset summary plots.
    """

    from endo_pipeline.library.analyze.track_integration import get_line_fit_and_filtered_df

    line_fit_df, _ = get_line_fit_and_filtered_df(first_passage_time_manifest, dataset_names)

    return line_fit_df
