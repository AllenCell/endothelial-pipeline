"""Methods for plotting flow switch fixed points over a cross-dataset summary."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from endo_pipeline.configs import DatasetConfig, load_dataset_config
from endo_pipeline.io import join_sorted_strings, save_plot_to_path
from endo_pipeline.library.visualize.summary_plot import (
    BINNED_MEAN_FEATURES,
    DEFAULT_SUMMARY_COLUMN_NAMES,
    SUMMARY_MODE_X_AXIS_SUP_LABELS,
    add_error_bar_bounds,
    build_color_by_column_mappable,
    compute_positions_from_group_keys,
    draw_error_bar_markers,
    draw_shear_stress_brackets,
    format_value_axis,
    sort_and_categorize_datasets,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL, MAX_FIGURE_WIDTH

if TYPE_CHECKING:
    from endo_pipeline.settings.column_names import ColumnNameType

logger = logging.getLogger(__name__)

BACKGROUND_ALPHA = 0.25
"""Default opacity of the background (steady-state) summary."""

FLOW_SWITCH_BAND_COLOR = "#F4C542"
"""Color of the band highlighting flow switch slots on the x axis."""

SLOT_WIDTH = 0.6
"""Spacing between x axis slots (matches summary plot replicate mode)."""

ARROW_Y_START = 0.92
"""Height (axes fraction) of the first flow switch arrow."""

ARROW_Y_STEP = 0.1
"""Vertical spacing (axes fraction) between arrows of different datasets."""


@dataclass(frozen=True)
class _Slot:
    """One position on the x axis of the flow switch summary plot."""

    key: str
    dataset: str
    shear_stress_bin: int
    stability: str
    tick_label: str
    shear_stress: float | None = None
    """Shear stress of the flow condition (only for flow switch slots)."""
    flow_condition_index: int | None = None
    """Index of the flow condition within the dataset (only for flow switch slots)."""

    @property
    def is_flow_switch(self) -> bool:
        """True if the slot holds the fixed points of a flow switch flow condition."""
        return self.shear_stress is not None

    @property
    def group_key(self) -> tuple[int, str]:
        """Key used to group neighboring slots under a shear stress bracket."""
        return (self.shear_stress_bin, self.stability)


def _build_background_slots(
    df_background: pd.DataFrame, dataset_configs: dict[str, DatasetConfig]
) -> list[_Slot]:
    """Build one slot per background dataset, ordered as in replicate mode."""

    _, ordered_datasets, _ = sort_and_categorize_datasets(
        df_background.copy(),
        "replicate",
        ColumnName.DATASET,
        None,
        dataset_configs,
        df_background[ColumnName.DATASET].unique(),
    )

    return [
        _Slot(
            key=str(dataset),
            dataset=str(dataset),
            shear_stress_bin=dataset_configs[str(dataset)].flow_conditions[-1].shear_stress_bin,
            stability=dataset_configs[str(dataset)].stability_category or "monostable",
            tick_label=str(dataset_configs[str(dataset)].replicate_number or ""),
        )
        for dataset in ordered_datasets
    ]


def _build_flow_switch_slots(
    df_flow_switch: pd.DataFrame, dataset_configs: dict[str, DatasetConfig]
) -> list[_Slot]:
    """Build one slot per flow condition of each flow switch dataset with fixed points."""

    slots = []
    for dataset in sorted(df_flow_switch[ColumnName.DATASET].unique()):
        df_dataset = df_flow_switch[df_flow_switch[ColumnName.DATASET] == dataset]
        for index, flow_condition in enumerate(dataset_configs[dataset].flow_conditions):
            if (df_dataset[ColumnName.SHEAR_STRESS] == flow_condition.shear_stress).any():
                slots.append(
                    _Slot(
                        key=f"{dataset}|{index}",
                        dataset=dataset,
                        shear_stress_bin=flow_condition.shear_stress_bin,
                        stability="monostable",
                        tick_label=f"{dataset_configs[dataset].replicate_number or ''}*",
                        shear_stress=flow_condition.shear_stress,
                        flow_condition_index=index,
                    )
                )
    return slots


def _combine_slots(background_slots: list[_Slot], flow_switch_slots: list[_Slot]) -> list[_Slot]:
    """
    Insert flow switch slots into the background slots.

    Each flow switch slot is placed after the last slot with the same shear
    stress bin and stability, so that it sits next to the steady-state
    datasets at the same shear stress. If no such group exists, the slot starts
    a new group before the first group with a higher shear stress bin.
    """

    slots = list(background_slots)
    for flow_switch_slot in flow_switch_slots:
        matching = [
            i for i, slot in enumerate(slots) if slot.group_key == flow_switch_slot.group_key
        ]
        if matching:
            insert_at = matching[-1] + 1
        else:
            higher = [
                i
                for i, slot in enumerate(slots)
                if slot.shear_stress_bin > flow_switch_slot.shear_stress_bin
            ]
            insert_at = higher[0] if higher else len(slots)
        slots.insert(insert_at, flow_switch_slot)
    return slots


def _get_points_for_slot(df: pd.DataFrame, slot: _Slot) -> pd.DataFrame:
    """Get the fixed points belonging to a slot."""

    df_slot = df[df[ColumnName.DATASET] == slot.dataset]
    if slot.is_flow_switch:
        df_slot = df_slot[df_slot[ColumnName.SHEAR_STRESS] == slot.shear_stress]
    return df_slot


def _draw_flow_switch_arrows(
    ax: Axes, flow_switch_slots: list[_Slot], positions: dict[str, float]
) -> None:
    """
    Draw a dashed arrow for each dataset from its first to its second flow condition.

    Arrows run from the edge of the first flow condition band to the nearest
    edge of the second flow condition band. Arrows for different datasets are
    staggered vertically so that they do not overlap.
    """

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    slots_by_dataset: dict[str, dict[int, _Slot]] = {}
    for slot in flow_switch_slots:
        if slot.flow_condition_index is not None:
            slots_by_dataset.setdefault(slot.dataset, {})[slot.flow_condition_index] = slot

    arrow_index = 0
    for conditions in slots_by_dataset.values():
        if 0 not in conditions or 1 not in conditions:
            continue

        start_x = positions[conditions[0].key]
        end_x = positions[conditions[1].key]
        direction = 1 if end_x > start_x else -1
        y = ARROW_Y_START - ARROW_Y_STEP * arrow_index
        ax.annotate(
            "",
            xy=(end_x - direction * SLOT_WIDTH / 2, y),
            xytext=(start_x + direction * SLOT_WIDTH / 2, y),
            xycoords=transform,
            textcoords=transform,
            arrowprops={
                "arrowstyle": "-|>",
                "linestyle": "--",
                "color": "black",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 6,
            },
            zorder=5,
        )
        arrow_index += 1


def _plot_points(
    ax: Axes,
    df: pd.DataFrame,
    slots: list[_Slot],
    positions: dict[str, float],
    column_name: "ColumnNameType | str",
    scalar_mappable,
    plotting_column: str,
    alpha: float | None,
    zorder: float,
    marker_size_plot: float,
) -> None:
    """Plot fixed points with error bars for the given slots."""

    capsize_scale = add_error_bar_bounds(df, column_name)

    for slot in slots:
        df_slot = _get_points_for_slot(df, slot)
        colors = (
            [scalar_mappable.to_rgba(value) for value in df_slot[plotting_column]]
            if scalar_mappable is not None
            else ["steelblue"] * len(df_slot)
        )
        draw_error_bar_markers(
            ax,
            [positions[slot.key]] * len(df_slot),
            df_slot[column_name],
            colors,
            ["o"] * len(df_slot),
            df_slot["_lower_bound"],
            df_slot["_upper_bound"],
            marker_size_plot=marker_size_plot,
            capsize_scale=capsize_scale,
            alpha=alpha,
            zorder=zorder,
        )


def plot_flow_switch_summaries(
    df_flow_switch: pd.DataFrame,
    df_background: pd.DataFrame,
    output_path: Path,
    regime_label: str,
    column_names: list["ColumnNameType"] | None = None,
    color_by_column: "ColumnNameType" = ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    background_alpha: float = BACKGROUND_ALPHA,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 1.5),
    marker_size_plot: float = 4.2,
    highlight_bistable: bool = True,
    ylabel_rotation: float = 90,
) -> Path:
    """
    Plot fixed points of flow switch datasets over a faded cross-dataset summary.

    The plot uses the summary plot "replicate" style: datasets are placed along
    the x axis grouped by shear stress bin, and fixed points are colored by
    migration coherence (``color_by_column``). The background shows all
    steady-state datasets in ``df_background`` at low opacity. Flow switch
    fixed points are drawn at full opacity, with a separate x axis slot for each
    flow condition of each dataset. Each slot is placed in the group matching
    the shear stress bin of its flow condition, so that flow switch fixed
    points can be compared against steady-state datasets at the same shear
    stress. A dashed arrow on the top panel points from the first to the second
    flow condition of each dataset. Slots are highlighted with a colored band and labeled with the
    replicate number from the dataset config followed by an asterisk (datasets
    without a replicate number are labeled with only an asterisk).

    Parameters
    ----------
    df_flow_switch
        Dataframe of flow switch fixed points, with all flow conditions
        included (see ``build_dataframe_for_fixed_point_dataset_summary``).
    df_background
        Dataframe of steady-state fixed points for the background summary.
    output_path
        Output path to save plot.
    regime_label
        Label for the shear stress regime of the flow switch datasets (e.g.
        ``"min_to_max"``) used in the legend and output file name.
    column_names
        List of feature column names, one panel per column.
    color_by_column
        Column whose values are mapped to point color.
    background_alpha
        Opacity of the background summary.
    figure_size
        Size of each panel in the output figure.
    marker_size_plot
        Size of the markers in the plot.
    highlight_bistable
        True to add a grey background spanning all bistable background datasets.
    ylabel_rotation
        Rotation angle for y axis label.

    Returns
    -------
    :
        Path to saved summary plot.
    """

    if df_flow_switch.empty:
        raise ValueError("No flow switch fixed points to plot")

    column_names = column_names or DEFAULT_SUMMARY_COLUMN_NAMES

    df_flow_switch = df_flow_switch.copy()
    df_background = df_background.copy()

    dataset_names = [
        *df_background[ColumnName.DATASET].unique(),
        *df_flow_switch[ColumnName.DATASET].unique(),
    ]
    dataset_configs = {str(name): load_dataset_config(str(name)) for name in dataset_names}

    background_slots = _build_background_slots(df_background, dataset_configs)
    flow_switch_slots = _build_flow_switch_slots(df_flow_switch, dataset_configs)
    slots = _combine_slots(background_slots, flow_switch_slots)

    slot_keys = [slot.key for slot in slots]
    positions = compute_positions_from_group_keys(slot_keys, [slot.group_key for slot in slots])

    # Use a single color scale across background and flow switch points
    scalar_mappable, plotting_column, color_metadata = build_color_by_column_mappable(
        pd.concat([df_background, df_flow_switch], ignore_index=True), color_by_column
    )

    n_panels = len(column_names)
    fig, axes_ = plt.subplots(
        n_panels,
        1,
        figsize=(figure_size[0], figure_size[1] * n_panels),
        layout="constrained",
        squeeze=False,
    )
    axes = [axes_[i][0] for i in range(n_panels)]

    for ax, base_column_name in zip(axes, column_names, strict=True):
        column_name = (
            f"mean_{base_column_name}"
            if base_column_name in BINNED_MEAN_FEATURES
            else base_column_name
        )

        for slot in flow_switch_slots:
            ax.axvspan(
                positions[slot.key] - SLOT_WIDTH / 2,
                positions[slot.key] + SLOT_WIDTH / 2,
                color=FLOW_SWITCH_BAND_COLOR,
                alpha=0.25,
                linewidth=0,
                zorder=0,
            )

        if highlight_bistable:
            bistable_positions = [
                positions[slot.key] for slot in background_slots if slot.stability == "bistable"
            ]
            if bistable_positions:
                ax.axvspan(
                    min(bistable_positions) - 0.4,
                    max(bistable_positions) + 0.4,
                    color="lightgrey",
                    alpha=0.4,
                    zorder=0,
                )

        for df, layer_slots, alpha, zorder in [
            (df_background, background_slots, background_alpha, 2),
            (df_flow_switch, flow_switch_slots, None, 4),
        ]:
            _plot_points(
                ax,
                df,
                layer_slots,
                positions,
                column_name,
                scalar_mappable,
                plotting_column,
                alpha,
                zorder,
                marker_size_plot,
            )

        format_value_axis(
            ax,
            column_name,
            COLUMN_METADATA[base_column_name],
            ylabel_rotation=ylabel_rotation,
        )

        ax.set_xticks([positions[key] for key in slot_keys])
        ax.set_xlim(positions[slot_keys[0]] - 0.5, positions[slot_keys[-1]] + 0.5)
        ax.tick_params(axis="x", labelsize=FONTSIZE_XSMALL, pad=2)
        ax.tick_params(axis="y", pad=2)
        ax.yaxis.labelpad = 8

    # Indicate the direction of each flow switch on the top panel only
    _draw_flow_switch_arrows(axes[0], flow_switch_slots, positions)

    # Label ticks and add brackets on the bottom panel only
    bottom_ax = axes[-1]
    bottom_ax.set_xticklabels([slot.tick_label for slot in slots])
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    draw_shear_stress_brackets(bottom_ax, slot_keys, [slot.group_key for slot in slots], positions)

    fig.align_ylabels(axes)
    fig.supxlabel(
        SUMMARY_MODE_X_AXIS_SUP_LABELS["replicate"],
        fontsize=FONTSIZE_MEDIUM,
        fontweight="bold",
    )

    axes[0].legend(
        handles=[
            Patch(
                facecolor=FLOW_SWITCH_BAND_COLOR,
                alpha=0.4,
                label=f"* Flow switch ({regime_label.replace('_', ' ')})",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=FONTSIZE_XSMALL,
    )

    if scalar_mappable is not None:
        label = (
            color_metadata.label
            if color_metadata and color_metadata.label
            else str(color_by_column)
        )
        cbar = fig.colorbar(scalar_mappable, ax=axes, pad=0.02, aspect=30)
        cbar.set_label(label.replace("\n", " "), fontsize=FONTSIZE_XSMALL)
        if color_metadata and color_metadata.ticks is not None:
            cbar.set_ticks(list(color_metadata.ticks))

    figure_name = (
        f"summary_flow_switch_{regime_label}_"
        f"{join_sorted_strings([str(column_name) for column_name in column_names])}"
    )
    save_plot_to_path(fig, output_path, figure_name, file_format=".svg", tight_layout=False)

    return output_path / f"{figure_name}.svg"
