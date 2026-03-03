from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_population_cov_vs_time(
    pop_cov_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
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

    axs: list[Axes]
    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

    for col, ax in zip(column_names, axs, strict=False):
        for time_values, cov_series, color, label in pop_cov_data[col]:
            ax.plot(time_values, cov_series, color=color, label=label, alpha=0.75, linewidth=1.2)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{variable_labels_dict[col]} CoV")
        ax.set_title(variable_labels_dict[col])
        if ylim_dict is not None and col in ylim_dict:
            ax.set_ylim(ylim_dict[col])

    # shared legend below subplots — collect unique label / color pairs from the last column
    handles_seen: dict[str, Artist] = {}
    for _, _, color, label in pop_cov_data[column_names[-1]]:
        if label not in handles_seen:
            (line,) = axs[-1].plot([], [], color=color, label=label)
            handles_seen[label] = line
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle("Population CoV vs time", y=1.01)
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "population_cov_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    return fig, axs


def plot_ergodicity_test(
    erg_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
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

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

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
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
    )
    fig.suptitle(
        "Ergodicity test: individual-crop temporal CoV vs population CoV",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "ergodicity_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_variance_ratio(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the ratio of individual to population variance as a function of time.

    At each timepoint the ratio is the mean per-crop cumulative temporal
    variance divided by the population (cross-sectional) variance.  A shaded
    band shows ± 1 SEM across crops.  A dashed reference line at ratio = 1
    marks the ergodic expectation.

    Each dataset-condition is drawn as a separate line and coloured by shear
    stress regime.

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

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

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
        ax.set_ylabel("Var$_{\\mathrm{individual}}$ / Var$_{\\mathrm{population}}$")
        ax.set_ylim(0, 1.5)
        ax.set_title(variable_labels_dict[col])

    # shared legend below subplots
    handles_seen: dict[str, Artist] = {}
    for entry in var_ratio_data[column_names[-1]]:
        _, _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(
        "Individual / population variance ratio vs time",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "variance_ratio_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_binned_variance_ratio(
    var_ratio_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
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

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

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
    handles_seen: dict[str, Artist] = {}
    for entry in var_ratio_data[column_names[-1]]:
        _, _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(
        "Individual / population variance ratio vs time (binned)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        fig_savedir / "binned_variance_ratio_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs


def plot_mean_feature_vs_time(
    mean_std_data: dict[str, list[tuple]],
    variable_labels_dict: dict[str, str],
    fig_savedir: Path,
    filename: str,
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

    fig, axs = plt.subplots(
        ncols=n_cols,
        figsize=(5 * n_cols, 5),
        dpi=300,
    )
    if n_cols == 1:
        axs = [axs]

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
    handles_seen: dict[str, Artist] = {}
    for entry in mean_std_data[column_names[-1]]:
        _, _, _, color, label = entry
        if label not in handles_seen:
            handles_seen[label] = Patch(facecolor=color, alpha=0.45, label=label)
    fig.legend(
        handles=list(handles_seen.values()),
        loc="lower center",
        ncol=min(3, len(handles_seen)),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
    )
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(
        fig_savedir / filename,
        dpi=300,
        bbox_inches="tight",
    )
    return fig, axs
