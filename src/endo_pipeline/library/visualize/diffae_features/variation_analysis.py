from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _initialize_figure_for_variation_analysis(n_cols: int) -> tuple[Figure, list[Axes]]:
    """
    Helper function to initialize a matplotlib figure with the specified number of columns.

    Parameters
    ----------
    n_cols
        Number of columns (subplots) in the figure.

    Returns
    -------
    fig
        The created matplotlib Figure object.
    axs
        List of Axes objects corresponding to each subplot.
    """
    fig, ax = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    axs: list[Axes]
    axs = [ax] if n_cols == 1 else ax
    return fig, axs


def _build_handles_seen_for_legend(
    data_entry: list[tuple],
    alpha: float = 0.45,
    handle_type: Literal["line", "patch"] = "patch",
) -> list[Artist]:
    """
    Helper function to build a mapping of unique labels to matplotlib Artist
    handles for legend creation.

    **Input data format**

    Each tuple in the input list be such that the last two elements are (color, label), e.g.:

    .. code-block:: python
        data_entry = (time_values, cov_series, color, label)


    Parameters
    ----------
    data_entry
        A list of tuples containing data for plotting.
    alpha
        Transparency level for the legend handles.
    handle_type
        Type of matplotlib Artist to create for the legend handles, either "line" or "patch".
    """
    handles_seen: list[Artist] = []
    for data_tuple in data_entry:
        color, label = data_tuple[-2], data_tuple[-1]
        if label not in handles_seen:
            if handle_type == "line":
                handles_seen.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        label=label,
                        alpha=alpha,
                    )
                )
            elif handle_type == "patch":
                handles_seen.append(Patch(facecolor=color, alpha=alpha, label=label))
            else:
                raise ValueError(f"Invalid handle_type: {handle_type}. Must be 'line' or 'patch'.")
    return handles_seen


def _format_fig_with_legend_and_title(
    fig: Figure,
    suptitle: str,
    legend_handles: list[Artist],
    legend_ncol: int,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.18),
    legend_fontsize: int = 7,
    suptitle_y: float = 1.01,
    tight_layout: bool = True,
) -> None:
    """Helper function to format a matplotlib figure with a suptitle and shared legend."""
    fig.suptitle(suptitle, y=suptitle_y)
    fig.legend(
        handles=legend_handles,
        loc=legend_loc,
        ncol=legend_ncol,
        bbox_to_anchor=legend_bbox_to_anchor,
        fontsize=legend_fontsize,
    )
    if tight_layout:
        fig.tight_layout()


def plot_population_cov_vs_time(
    pop_cov_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    title: str,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot population CoV vs time for all dataset / flow conditions on a shared figure.

    Each dataset-condition is drawn as a separate line and coloured by shear stress
    regime.  A single shared legend is placed below the subplots.

    Parameters
    ----------
    pop_cov_data
        Mapping from feature column name to a list of
        ``(time_values, cov_series, color, label)`` tuples — one per
        dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    ylim_dict
        Optional mapping from feature column name to ``(ymin, ymax)`` y-axis
        limits.  When ``None`` (default), matplotlib auto-scales the y-axis.
        Columns absent from the dict also fall back to auto-scaling.
    """
    column_names = list(pop_cov_data.keys())
    n_cols = len(column_names)

    fig, axs = _initialize_figure_for_variation_analysis(n_cols)

    for col, ax in zip(column_names, axs, strict=False):
        for time_values, cov_series, color, label in pop_cov_data[col]:
            ax.plot(time_values, cov_series, color=color, label=label, alpha=0.75, linewidth=1.2)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{variable_labels_dict[col]} CoV")
        ax.set_title(variable_labels_dict[col])
        if ylim_dict is not None and col in ylim_dict:
            ax.set_ylim(ylim_dict[col])

    # shared legend below subplots — collect unique label / color pairs from the last column
    handles_seen = _build_handles_seen_for_legend(
        pop_cov_data[column_names[-1]],
        alpha=1.0,
        handle_type="line",
    )
    _format_fig_with_legend_and_title(
        fig,
        suptitle=title,
        legend_handles=handles_seen,
        legend_ncol=min(3, len(handles_seen)),
    )

    return fig, axs


def plot_ergodicity_test(
    erg_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    title: str,
) -> tuple[Figure, list[Axes]]:
    """
    Visualize the ergodicity test by comparing temporal and ensemble CoV.

    For an ergodic system the time-average of an observable computed from a
    single trajectory (= per-crop temporal CoV) should equal the ensemble
    average across all crops at a single snapshot (= mean population CoV).
    Deviations between the two indicate non-ergodic behaviour.

    Each subplot shows a violin plot of the distribution of per-crop temporal
    CoV values (one violin per dataset / flow condition), with a diamond marker
    overlaid at the corresponding mean population CoV.  If the system is
    ergodic the diamond should fall near the centre of the violin.

    Parameters
    ----------
    erg_data
        Mapping from feature column name to a list of
        ``(crop_temporal_cov, mean_pop_cov, color, label)`` tuples — one per
        dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    """
    column_names = list(erg_data.keys())
    n_cols = len(column_names)

    fig, axs = _initialize_figure_for_variation_analysis(n_cols)

    for col, ax in zip(column_names, axs, strict=False):
        entries = erg_data[col]
        crop_cov_arrays = [e[0] for e in entries]
        mean_pop_covs = [e[1] for e in entries]
        colors = [e[2] for e in entries]
        labels = [e[3] for e in entries]

        # build long-format DataFrame for seaborn violinplot
        records = [
            (label, float(val))
            for label, cov_arr in zip(labels, crop_cov_arrays, strict=False)
            for val in cov_arr
        ]
        df_viol = pd.DataFrame(records, columns=["condition", "CoV"])
        palette = dict(zip(labels, colors, strict=False))

        # violin plot of per-crop temporal CoV distributions
        sns.violinplot(
            data=df_viol,
            x="condition",
            y="CoV",
            hue="condition",
            legend=False,
            palette=palette,
            inner="quart",
            alpha=0.6,
            linewidth=0.8,
            ax=ax,
            cut=0,
        )

        # diamond marker for mean population (ensemble) CoV
        ax.scatter(
            np.arange(len(labels)),
            mean_pop_covs,
            color=colors,
            marker="D",
            s=60,
            zorder=5,
            edgecolors="k",
            linewidths=0.7,
        )

        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("CoV")
        ax.set_ylim(bottom=0)
        ax.set_title(variable_labels_dict[col])

    # shared legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.45, label="Per-crop temporal CoV (distribution)"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=8,
            label="Mean population CoV",
        ),
    ]
    _format_fig_with_legend_and_title(
        fig,
        suptitle=title,
        legend_handles=legend_elements,
        legend_ncol=2,
        legend_bbox_to_anchor=(0.5, -0.06),
        legend_fontsize=8,
    )
    return fig, axs


def plot_variance_ratio_vs_time(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    title: str,
    ylabel_suffix: str = "",
) -> tuple[Figure, list[Axes]]:
    """
    Plot the ratio of individual to population variance as a function of time.

    Can be used to plot either the cumulative variance ratio (per-crop
    cumulative temporal variance vs population variance) or the binned variance
    ratio (per-crop variance within binned window of time vs population variance
    within same bins). A shaded band shows ± 1 SEM across crops. A dashed
    reference line at ratio = 1 marks the ergodic expectation.

    Each dataset-condition is drawn as a separate line and coloured by shear
    stress regime.

    Parameters
    ----------
    var_ratio_data
        Mapping from feature column name to a list of ``(time_values,
        ratio_mean, ratio_upper, ratio_lower, color, label)`` tuples — one per
        dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    ylabel_suffix
        Optional suffix appended to each y-axis label (e.g. " (cumulative)").
    """
    column_names = list(var_ratio_data.keys())
    n_cols = len(column_names)

    fig, axs = _initialize_figure_for_variation_analysis(n_cols)

    for col, ax in zip(column_names, axs, strict=False):
        for entry in var_ratio_data[col]:
            time_values, ratio_mean, ratio_upper, ratio_lower, color, label = entry
            ax.plot(
                time_values,
                ratio_mean,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=ratio_upper,
                y2=ratio_lower,
                alpha=0.25,
                color=color,
                label=label,
            )
        # reference line at ratio = 1
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(
            f"Var$_{{\\mathrm{{individual}}}}$ / Var$_{{\\mathrm{{population}}}}$ {ylabel_suffix}"
        )
        ax.set_ylim(0, 1.5)
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen = _build_handles_seen_for_legend(var_ratio_data[column_names[-1]])
    _format_fig_with_legend_and_title(
        fig,
        suptitle=title,
        legend_handles=handles_seen,
        legend_ncol=min(3, len(handles_seen)),
    )
    return fig, axs


def plot_binned_variance_ratio(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    title: str,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the ratio of individual to population variance computed within time bins.

    This is the non-cumulative counterpart of :func:`plot_variance_ratio`.
    At each time-bin centre the ratio is the mean per-crop variance *within
    the bin* divided by the population variance within the same bin.  A shaded
    band shows ± 1 SEM across crops.  A dashed reference line at ratio = 1
    marks the ergodic expectation.

    **How to interpret the plot**

    * **Ratio ≈ 1 everywhere** — individual crops fluctuate as much as the
      whole population within each short window → ergodic.
    * **Ratio ≪ 1** — crops occupy narrow, distinct niches in feature space →
      heterogeneous / non-ergodic.
    * **Ratio rising toward 1** — the system is mixing over time.
    * Comparing this binned plot with the cumulative version reveals whether
      ergodicity is driven by local fluctuations (binned ratio ≈ 1) or slow
      drift (cumulative ratio ≈ 1 but binned ratio < 1).

    Parameters
    ----------
    var_ratio_data
        Mapping from feature column name to a list of
        ``(time_values, ratio_mean, ratio_upper, ratio_lower, color, label)``
        tuples — one per dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    """
    column_names = list(var_ratio_data.keys())
    n_cols = len(column_names)

    fig, axs = _initialize_figure_for_variation_analysis(n_cols)

    for col, ax in zip(column_names, axs, strict=False):
        for entry in var_ratio_data[col]:
            time_values, ratio_mean, ratio_upper, ratio_lower, color, label = entry
            ax.plot(
                time_values,
                ratio_mean,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=ratio_upper,
                y2=ratio_lower,
                alpha=0.25,
                color=color,
                label=label,
            )
        # reference line at ratio = 1
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Var$_{\\mathrm{individual}}$ / Var$_{\\mathrm{population}}$ (binned)")
        ax.set_ylim(0, 1.5)
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen = _build_handles_seen_for_legend(var_ratio_data[column_names[-1]])
    _format_fig_with_legend_and_title(
        fig,
        suptitle=title,
        legend_handles=handles_seen,
        legend_ncol=min(3, len(handles_seen)),
    )
    return fig, axs


def plot_mean_feature_vs_time(
    mean_std_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    title: str,
    ylabel_suffix: str = "",
) -> tuple[Figure, list[Axes]]:
    """
    Plot population mean ± std of each feature as a function of time.

    Each dataset-condition is drawn as a line (mean) with a shaded band
    (± 1 std) and coloured by shear stress regime.

    Parameters
    ----------
    mean_std_data
        Mapping from feature column name to a list of
        ``(time_values, mean_array, std_array, color, label)`` tuples — one
        per dataset / flow condition.
    variable_labels_dict
        Human-readable label for each feature column name.
    fig_savedir
        Directory to save the figure.
    filename
        Filename (without directory) for the saved figure.
    title
        Figure suptitle.
    ylabel_suffix
        Optional suffix appended to each y-axis label (e.g. " (scaled)").
    """
    column_names = list(mean_std_data.keys())
    n_cols = len(column_names)

    fig, axs = _initialize_figure_for_variation_analysis(n_cols)

    for col, ax in zip(column_names, axs, strict=False):
        for entry in mean_std_data[col]:
            time_values, mean_arr, std_arr, color, label = entry
            ax.plot(
                time_values,
                mean_arr,
                color=color,
                alpha=0.9,
                linewidth=1.2,
            )
            ax.fill_between(
                time_values,
                y1=mean_arr + std_arr,
                y2=mean_arr - std_arr,
                alpha=0.25,
                color=color,
                label=label,
                edgecolor="none",
            )
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{variable_labels_dict[col]}{ylabel_suffix}")
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen = _build_handles_seen_for_legend(mean_std_data[column_names[-1]])
    _format_fig_with_legend_and_title(
        fig,
        suptitle=title,
        legend_handles=handles_seen,
        legend_ncol=min(3, len(handles_seen)),
    )
    return fig, axs
