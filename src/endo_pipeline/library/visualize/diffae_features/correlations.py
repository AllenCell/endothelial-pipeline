import logging
import re
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import get_dataset_descriptions
from endo_pipeline.library.analyze.numerics import (
    double_exponential_decay,
    exponential_decay,
    fit_exp_decay_and_get_relaxation_timescale,
)
from endo_pipeline.library.analyze.numerics.correlations import CROSS_CORR_INDEX_COMBINATIONS
from endo_pipeline.library.visualize.viz_base import init_plot

logger = logging.getLogger(__name__)


def _plot_single_acf_curve(
    lags: np.ndarray,
    acf: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[int, int] = (12, 6),
    plot_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the autocorrelation function (ACF) curves for given lags and ACF values."""
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = init_plot(figsize=figsize)

    ax.plot(lags, acf, **kwargs)
    ax.set_title(plot_title if plot_title is not None else "Autocorrelation Function")
    ax.set_xlabel(xlabel if xlabel is not None else "Lag")
    ax.set_ylabel(ylabel if ylabel is not None else "ACF")
    return fig, ax


def _plot_acf_curves_together(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    bootstrap_samples: int = 0,
    figsize: tuple[int, int] = (12, 6),
    component_labels: list[str] | None = None,
    component_colors: list[str] | None = None,
    plot_title: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple ACF curves together for comparison."""
    fig, ax = init_plot(figsize=figsize)

    lags: np.ndarray = correlation_dict["lags"][dataset_name]
    acf_array: np.ndarray = correlation_dict["acf"][dataset_name]

    # plot only positive lags
    index_positive = lags > 0
    positive_lags = lags[index_positive]
    positive_lags_as_hours = 5 * positive_lags / 60  # convert from frames (5 minutes) to hours
    acf_positive_lags = acf_array[index_positive]

    num_dims = acf_array.shape[1]
    # loop over each dimension/component and plot its ACF curve
    for i in range(num_dims):
        fig, ax = _plot_single_acf_curve(
            positive_lags_as_hours,
            acf_positive_lags[:, i],
            fig_ax=(fig, ax),
            plot_title=plot_title,
            label=component_labels[i] if component_labels else f"Component {i + 1}",
            color=component_colors[i] if component_colors else list(TABLEAU_COLORS.keys())[i],
            **kwargs,
        )

        # add confidence intervals if available
        if bootstrap_samples > 0:
            acf_ci_lower = correlation_dict["acf_ci_lower"][dataset_name][index_positive]
            acf_ci_upper = correlation_dict["acf_ci_upper"][dataset_name][index_positive]
            ax.fill_between(
                positive_lags_as_hours,
                acf_ci_lower[:, i],
                acf_ci_upper[:, i],
                alpha=0.25,
                color=list(TABLEAU_COLORS.keys())[i],
                label="95% CI",
            )

    return fig, ax


def _parse_dataset_description(dataset_description: str) -> str:
    """Parse dataset description for better readability in plot titles."""
    # replace underscores with spaces for better readability
    description_parsed = dataset_description.replace("_", " ")
    # find [0-9]dyncm2, put comma and space before, put a space between number and unit,
    # and change dyncm2 to dyn/cm^2 for better readability
    description_parsed = re.sub(r"(\d+)dyncm2", r", \1 dyn/cm$^2$", description_parsed)
    # turn capital 'S' into lowercase 's' for shear stress
    description_parsed = description_parsed.replace(" Shear Stress", " shear stress")
    # remove unwanted space before comma
    description_parsed = description_parsed.replace(" ,", ",")
    return description_parsed


def _add_relaxation_timescale_to_plot(relaxation_timescales: list[float], ax: plt.Axes) -> plt.Axes:
    """Print relaxation timescales on plot of ACFs."""
    # using unicode because slurm nodes and A100s do not support LaTeX rendering
    tau_str = chr(964)  # Greek letter tau (τ)
    strings_per_pc = [
        f"PC{i+1}: {tau_str} = {tau:.2f} hrs" for i, tau in enumerate(relaxation_timescales)
    ]
    # use ax coordinates to place text in lower left corner of plot
    for i, string in enumerate(strings_per_pc):
        x_loc = 0.025
        # decrement y_loc for each PC to avoid overlap
        y_loc = 0.05 + 0.075 * (len(strings_per_pc) - 1 - i)
        ax.text(
            x_loc,
            y_loc,
            string,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8},
            color=list(TABLEAU_COLORS.keys())[i],  # use same color as ACF curve,
            weight="bold",
            transform=ax.transAxes,
        )

    return ax


def _add_delta_ccf_integral_to_plot(
    delta_ccf_integral: np.ndarray, max_lag_integrate: int, ci_bounds: tuple | None, ax: plt.Axes
) -> plt.Axes:
    """Print integral of delta CCF near zero on plot of delta CCFs."""
    integral_upper_bound_hrs = round(5 * max_lag_integrate / 60, 2)  # convert from frames to hours
    integral_srings = [
        rf"$|\int_{{0}}^{{{integral_upper_bound_hrs}}}\Delta C_{{{j+1}{k+1}}}(\tau) d\tau|$"
        for (j, k) in CROSS_CORR_INDEX_COMBINATIONS
    ]
    strings_per_pc = [
        rf"PC{j+1}, PC{k+1}: {string} = {integral:.2f}"
        for (j, k), string, integral in zip(
            CROSS_CORR_INDEX_COMBINATIONS, integral_srings, delta_ccf_integral, strict=True
        )
    ]
    x_loc = 0.7  # place on upper right corner of plot

    # add confidence intervals if available
    if ci_bounds is not None:
        x_loc = 0.5  # need more space for CI
        ci_lower, ci_upper = ci_bounds
        strings_per_pc = [
            f"{string} (95% CI: {ci_lower[i]:.2f}, {ci_upper[i]:.2f})"
            for i, string in enumerate(strings_per_pc)
        ]

    for i, string in enumerate(strings_per_pc):
        # print on upper right corner of plot
        # decrement y_loc for each PC combination to avoid overlap
        y_loc = 0.675 + 0.115 * (len(strings_per_pc) - 1 - i)
        # use transform to place text in axes coordinates
        ax.text(
            x_loc,
            y_loc,
            string,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8},
            color=list(TABLEAU_COLORS.keys())[i],  # use same color as delta CCF curve,
            weight="bold",
            transform=ax.transAxes,
        )

    return ax


def _add_exp_fit_to_plot(
    acf: np.ndarray,
    lags: np.ndarray,
    ax: plt.Axes,
    exp_decay_func: Literal["exponential_decay", "double_exponential_decay"],
) -> tuple[plt.Axes, list[float]]:
    """Fit exponential decay to ACF and add to existing plot."""
    # check to make sure valid function is provided
    if exp_decay_func not in ["exponential_decay", "double_exponential_decay"]:
        logger.error(
            "Invalid exp_decay_func provided: [ %s ]. "
            "Must be 'exponential_decay' or 'double_exponential_decay'.",
            exp_decay_func,
        )
        raise ValueError("Invalid exp_decay_func provided to _add_exp_fit_to_plot.")

    # fit exponential decay to each PC's ACF and plot on existing axes
    relaxation_timescales = []
    for i in range(3):
        try:
            exp_fit, relaxation_time = fit_exp_decay_and_get_relaxation_timescale(
                acf[:, i], lags, exp_decay_func=exp_decay_func
            )
            relaxation_timescales.append(relaxation_time)
        except RuntimeError:
            logger.warning(
                "Could not fit [ %s ] to ACF of PC%s, skipping plot step", exp_decay_func, i + 1
            )
            relaxation_timescales.append(np.nan)
            continue

        # get curve of fit exponential decay
        if exp_decay_func == "exponential_decay":
            acf_fit = exponential_decay(lags, *exp_fit)
            logger.debug(
                "Exponential fit for PC%s: [%.3f + %.3f exp(%.3f tau)]",
                i + 1,
                exp_fit[2],
                exp_fit[0],
                -exp_fit[1],
            )
        else:
            acf_fit = double_exponential_decay(lags, *exp_fit)
            which_weight_is_larger = np.argmax(np.abs(exp_fit[[0, 2]]))
            logger.debug(
                "Full double exponential fit for PC%s: "
                "[%.3f + %.3f exp(%.3f tau) + %.3f exp(%.3f tau) ]",
                i + 1,
                exp_fit[4],
                exp_fit[0],
                -exp_fit[1],
                exp_fit[2],
                -exp_fit[3],
            )
            logger.debug(
                "Dominant exponent in multi-exponential fit for PC%s: [ %.3f exp(%.3f tau) ]",
                i + 1,
                exp_fit[[0, 2][which_weight_is_larger]],
                -exp_fit[[1, 3][which_weight_is_larger]],
            )
        # plot fit curve on existing axes
        ax.plot(lags, acf_fit, "k--", linewidth=2.0, alpha=0.85, label="")

    ax.legend()
    ax.set_ylim(-0.25, 1.05)

    # add relaxation timescale to plot
    ax = _add_relaxation_timescale_to_plot(relaxation_timescales, ax)

    return ax, relaxation_timescales


def _make_all_acf_plots(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    dataset_description: str,
    output_path: Path,
    fit_double_exp: bool = True,
    bootstrap_samples: int = 0,
) -> dict[str, dict[str, Any]]:
    # unpack results
    lags: np.ndarray = correlation_dict["lags"][dataset_name]
    acf: np.ndarray = correlation_dict["acf"][dataset_name]

    # plot acf for positive lags
    # (acf is symmetric around zero)
    index_positive = lags > 0
    lags_ = lags[index_positive]
    lags_as_hours = 5 * lags_ / 60  # convert from frames (5 minutes) to hours
    acf_ = acf[index_positive]
    fig, ax = _plot_acf_curves_together(
        dataset_name,
        correlation_dict,
        bootstrap_samples=bootstrap_samples,
        component_labels=["PC1", "PC2", "PC3"],
        plot_title=f"Autocorrelation of PCA Components ({dataset_description})",
        xlabel="Lag (hours)",
        linewidth=2.75,
    )
    ax.legend()
    ax.set_ylim(-0.25, 1.05)
    save_plot_to_path(
        fig,
        output_path,
        f"autocorrelation_{dataset_name}",
    )

    # fit single exponential decay to ACF
    fig, ax = _plot_acf_curves_together(
        dataset_name,
        correlation_dict,
        bootstrap_samples=bootstrap_samples,
        component_labels=["PC1", "PC2", "PC3"],
        plot_title=f"Autocorrelation of PCA Components ({dataset_description})",
        xlabel="Lag (hours)",
        linewidth=2.75,
    )
    ax, relaxation_timescales = _add_exp_fit_to_plot(
        acf_, lags_as_hours, ax, exp_decay_func="exponential_decay"
    )
    # add relaxation timescales to correlation_dict for output
    correlation_dict["relaxation_timescales"][dataset_name] = relaxation_timescales
    save_plot_to_path(
        fig,
        output_path,
        f"autocorrelation_exp_fit_{dataset_name}",
    )

    if fit_double_exp:
        # fit double exponential decay to ACF
        fig, ax = _plot_acf_curves_together(
            dataset_name,
            correlation_dict,
            bootstrap_samples=bootstrap_samples,
            component_labels=["PC1", "PC2", "PC3"],
            plot_title=f"Autocorrelation of PCA Components ({dataset_description})",
            xlabel="Lag (hours)",
            linewidth=2.75,
            linestyle="-",
        )
        ax, _ = _add_exp_fit_to_plot(
            acf_, lags_as_hours, ax, exp_decay_func="double_exponential_decay"
        )
        save_plot_to_path(
            fig,
            output_path,
            f"autocorrelation_double_exp_fit_{dataset_name}",
        )

    return correlation_dict


def _make_all_ccf_plots(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    dataset_description: str,
    output_path: Path,
    bootstrap_samples: int = 0,
) -> None:
    # unpack results
    lags: np.ndarray = correlation_dict["lags"][dataset_name]
    num_lags = len(lags)
    ccf: np.ndarray = correlation_dict["ccf"][dataset_name]
    ccf_ci_lower: np.ndarray = correlation_dict["ccf_ci_lower"][dataset_name]
    ccf_ci_upper: np.ndarray = correlation_dict["ccf_ci_upper"][dataset_name]
    delta_ccf: np.ndarray = correlation_dict["delta_ccf"][dataset_name]
    delta_ccf_ci_lower: np.ndarray = correlation_dict["delta_ccf_ci_lower"][dataset_name]
    delta_ccf_ci_upper: np.ndarray = correlation_dict["delta_ccf_ci_upper"][dataset_name]
    delta_ccf_integral: np.ndarray = correlation_dict["delta_ccf_integral"][dataset_name]
    max_lag_integrate: int = correlation_dict["max_lag_integrate"][dataset_name]

    # plot ccf with confidence intervals if available
    fig, ax = init_plot(figsize=(12, 6))
    for i, (j, k) in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        lags_all_as_hours = 5 * lags / 60  # convert from frames (5 minutes) to hours
        ax.plot(lags_all_as_hours, ccf[:, i], label=f"(PC{j+1}, PC{k+1})")
        if bootstrap_samples > 0:
            ax.fill_between(
                lags_all_as_hours,
                ccf_ci_lower[:, i],
                ccf_ci_upper[:, i],
                alpha=0.25,
                color=list(TABLEAU_COLORS.keys())[i],
                label="95% CI",
            )
    ax.set_title(f"Cross-Correlation of PCA Components ({dataset_description})")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("CCF")
    ax.legend()
    ax.set_ylim(-0.25, 0.75)
    save_plot_to_path(
        fig,
        output_path,
        f"cross_correlation_{dataset_name}",
    )

    # plot delta ccf: difference between positive and negative lags
    fig, ax = init_plot(figsize=(12, 6))
    for i, (j, k) in enumerate(CROSS_CORR_INDEX_COMBINATIONS):
        # delta ccf is symmetric around zero
        lags_symmetric = lags[1 + num_lags // 2 :]
        lags_symmetric_as_hours = 5 * lags_symmetric / 60
        ax.plot(lags_symmetric_as_hours, delta_ccf[:, i], label=f"(PC{j+1}, PC{k+1})")
        if bootstrap_samples > 0:
            ax.fill_between(
                lags_symmetric_as_hours,
                delta_ccf_ci_lower[:, i],
                delta_ccf_ci_upper[:, i],
                alpha=0.25,
                color=list(TABLEAU_COLORS.keys())[i],
                label="95% CI",
            )
    ax.set_title(f"$C_{{ij}}(\\tau) - C_{{ij}}(-\\tau)$ ({dataset_description})")
    ax.set_xlabel("Lag $\\tau$ (hours)")
    ax.set_ylabel("$\Delta C_{ij}(\\tau)$")
    ax.legend()
    ax.set_ylim(-0.5, 0.75)
    # print integral of delta ccf near zero on plot
    ci_bounds = None
    if bootstrap_samples > 0:
        ci_bounds = (
            correlation_dict["delta_ccf_integral_ci_lower"][dataset_name],
            correlation_dict["delta_ccf_integral_ci_upper"][dataset_name],
        )
    ax = _add_delta_ccf_integral_to_plot(delta_ccf_integral, max_lag_integrate, ci_bounds, ax)
    save_plot_to_path(
        fig,
        output_path,
        f"cross_correlation_diff_{dataset_name}",
    )


def _plot_full_correlation_curves(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    dataset_descriptions: dict[str, str],
    output_path: Path,
    bootstrap_samples: int = 0,
) -> dict[str, dict[str, Any]]:
    """Plot full correlation curves for a single dataset."""
    # get string for dataset description
    dataset_description = _parse_dataset_description(dataset_descriptions[dataset_name])

    # plot acf and fit exponential decay
    # adds relaxation timescales to correlation_dict
    correlation_dict = _make_all_acf_plots(
        dataset_name,
        correlation_dict,
        dataset_description,
        output_path,
        bootstrap_samples=bootstrap_samples,
    )

    # plot ccf and difference between positive and negative lags
    _make_all_ccf_plots(
        dataset_name,
        correlation_dict,
        dataset_description,
        output_path,
        bootstrap_samples=bootstrap_samples,
    )

    return correlation_dict


def _plot_single_correlation_metric_vs_shear_stress(
    metric_values: list[np.ndarray],
    shear_stresses: np.ndarray,
    ci_bounds: list[tuple] | None = None,
    labels: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    # init plot
    fig, ax = init_plot(figsize=(8, 6))

    # set default labels if none provided
    if labels is None:
        labels = [f"(PC{j+1}, PC{k+1})" for (j, k) in CROSS_CORR_INDEX_COMBINATIONS]

    # sort by ascending shear stress
    sorted_indices = np.argsort(shear_stresses)
    shear_sorted = shear_stresses[sorted_indices]

    # plot each metric value vs shear stress
    for i, label in enumerate(labels):
        # get values for this PC or PC combination across all datasets
        values = np.array([value[i] for value in metric_values])[sorted_indices]
        # sort by ascending shear stress
        ax.plot(
            shear_sorted,
            values,
            label=label,
            color=list(TABLEAU_COLORS.keys())[i],
            linewidth=2.75,
            linestyle="-",
        )
        ax.scatter(
            shear_sorted,
            values,
            color=list(TABLEAU_COLORS.keys())[i],
            s=100,
            edgecolor="k",
            label="",
        )
        # add error bars if ci_bounds provided
        if ci_bounds is not None:
            ci_bounds_array = np.array(ci_bounds)[sorted_indices]
            ci_lower = ci_bounds_array[:, 0, i]
            ci_upper = ci_bounds_array[:, 1, i]
            ax.errorbar(
                shear_sorted,
                values,
                yerr=[values - ci_lower, ci_upper - values],
                fmt="none",
                ecolor=list(TABLEAU_COLORS.keys())[i],
                elinewidth=1.5,
                capsize=5,
                label="",
            )

    return fig, ax


def _plot_correlation_metrics_vs_shear_stress(
    correlation_dict: dict[str, dict[str, Any]],
    list_of_datasets: list[str],
    output_path: Path,
) -> None:
    """Plot integral of delta CCF near zero as a function of shear stress."""

    def _get_shear_stress_from_dataset_name(dataset_name: str) -> float:
        flow_conditions = load_dataset_config(dataset_name).flow_conditions
        single_flow_condition = next(iter(flow_conditions))
        return single_flow_condition.shear_stress

    # plot correlation metrics vs shear stress
    # with error bars if available
    delta_ccf_integral_values = [
        correlation_dict["delta_ccf_integral"][dataset_name] for dataset_name in list_of_datasets
    ]
    delta_ccf_integral_ci_bounds = [
        (
            correlation_dict["delta_ccf_integral_ci_lower"][dataset_name],
            correlation_dict["delta_ccf_integral_ci_upper"][dataset_name],
        )
        for dataset_name in list_of_datasets
    ]
    # also plot mean over PCs
    mean_delta_ccf_integral = [
        np.array([np.mean(correlation_dict["delta_ccf_integral"][dataset_name])])
        for dataset_name in list_of_datasets
    ]

    # also plot relaxation timescales
    relaxation_timescale_values = [
        np.array(correlation_dict["relaxation_timescales"][dataset_name])
        for dataset_name in list_of_datasets
    ]
    relaxation_timescale_ci_bounds = [
        (
            correlation_dict["relaxation_timescales_ci_lower"][dataset_name],
            correlation_dict["relaxation_timescales_ci_upper"][dataset_name],
        )
        for dataset_name in list_of_datasets
    ]

    # get shear stresses for each dataset
    shear_stresses = np.array(
        [_get_shear_stress_from_dataset_name(dataset_name) for dataset_name in list_of_datasets]
    )

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        delta_ccf_integral_values, shear_stresses, ci_bounds=delta_ccf_integral_ci_bounds
    )
    ax.legend()
    ax.set_ylabel("$\\langle |\\Delta C_{ij} |\\rangle$")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "delta_ccf_integral_vs_shear_stress",
    )

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        mean_delta_ccf_integral, shear_stresses, labels=["Mean over PCs"]
    )
    ax.legend()
    ax.set_ylabel("$\\overline{|\\Delta C_{ij}|}$")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "mean_delta_ccf_integral_vs_shear_stress",
    )

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        relaxation_timescale_values,
        shear_stresses,
        ci_bounds=relaxation_timescale_ci_bounds,
        labels=["PC1", "PC2", "PC3"],
    )
    ax.legend()
    ax.set_ylabel("Relaxation timescale (hours)")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "relaxation_time_vs_shear_stress",
    )


def plot_correlation_workflow_outputs(
    correlation_dict: dict[str, dict[str, Any]], bootstrap_samples: int = 0
) -> None:
    """
    Plot correlation workflow outputs.

    **Workflow output**

    Creates and saves a series of summary plots for the correlation analysis results:
    - Plots the auto and cross-correlation functions for each dataset in
    ``correlation_dict``.
    - Fits exponential decay to ACFs to extract relaxation timescales.
    - Plots CCFs with confidence intervals if bootstrap samples were provided.
    - Plots difference between positive and negative lag CCFs and computes integrals near
        lag = zero, which are then printed on the plots.
    - Plots these integrals as a function of shear stress across datasets.

    Parameters
    ----------
    correlation_dict
        Dictionary containing correlation results for multiple datasets.

    bootstrap_samples
        Optional, number of bootstrap samples used to compute confidence intervals.
    """
    list_of_datasets = list(correlation_dict["lags"].keys())
    dataset_descriptions = get_dataset_descriptions(
        list_of_datasets, simple=True, include_duration=False, include_shear_stress=True
    )
    output_path = get_output_path("correlations")

    # plot full correlation curves for each dataset
    for dataset_name in list_of_datasets:
        logger.info("Plotting correlation curves for dataset [ %s ]", dataset_name)
        correlation_dict = _plot_full_correlation_curves(
            dataset_name,
            correlation_dict,
            dataset_descriptions,
            output_path,
            bootstrap_samples=bootstrap_samples,
        )

    # plot integrated difference between CCF for positive and
    # negative lags as a function of shear stress
    _plot_correlation_metrics_vs_shear_stress(
        correlation_dict,
        list_of_datasets,
        output_path,
    )
