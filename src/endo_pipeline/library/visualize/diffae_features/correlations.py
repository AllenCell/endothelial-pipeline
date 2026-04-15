"""Visualization methods for time series auto- and cross-correlation analyses."""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.figure import Figure

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.numerics.correlations import (
    double_exponential_decay,
    exponential_decay,
    fit_exp_decay_and_get_relaxation_timescale,
)

logger = logging.getLogger(__name__)


def _plot_single_acf_curve(
    lags: np.ndarray,
    acf: np.ndarray,
    fig_ax: tuple[Figure, Axes] | None = None,
    figsize: tuple[int, int] = (12, 6),
    plot_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the autocorrelation function (ACF) curves for given lags and ACF values."""
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lags, acf, **kwargs)
    ax.set_title(plot_title if plot_title is not None else "Autocorrelation Function")
    ax.set_xlabel(xlabel if xlabel is not None else "Lag")
    ax.set_ylabel(ylabel if ylabel is not None else "ACF")
    return fig, ax


def _plot_acf_curves_together(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    bootstrap_samples: int | None = None,
    figsize: tuple[int, int] = (12, 6),
    component_labels: list[str] | None = None,
    component_colors: list[str] | None = None,
    plot_title: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot multiple autocorrelation function (ACF) curves together for comparison."""
    fig, ax = plt.subplots(figsize=figsize)

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
        if bootstrap_samples is not None:
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


def _add_relaxation_timescale_to_plot(
    relaxation_timescales: list[float], feature_labels: list[str], ax: Axes
) -> Axes:
    """Print relaxation timescales on plot of autocorrelation functions (ACFs)."""
    # using unicode because slurm nodes and A100s do not support LaTeX rendering
    tau_str = chr(964)  # Greek letter tau (τ)
    strings_per_pc = [
        f"{feature_labels[i]}: {tau_str} = {tau:.2f} hrs"
        for i, tau in enumerate(relaxation_timescales)
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
    delta_ccf_integral: np.ndarray,
    max_lag_integrate: int,
    ci_bounds: tuple | None,
    feature_labels: list[str],
    ax: Axes,
) -> Axes:
    """Print summary of cross-correlation function (CCF) differences on plot.

    The summary metric is the integral of the difference between positive and
    negative lag CCFs near lag = zero, which quantifies the overall asymmetry of
    the CCF around zero lag. This metric is printed on the plot for each feature
    combination, along with confidence intervals if available.

    Parameters
    ----------
    delta_ccf_integral
        Array of integral values for each feature combination.
    max_lag_integrate
        Maximum lag (in frames) to integrate over around lag = zero.
    ci_bounds
        Tuple of (lower bounds, upper bounds) for confidence intervals of the integral values.
    feature_labels
        List of feature labels corresponding to the indices in the integral array.
    ax
        Matplotlib Axes object to add the text to.

    Returns
    -------
    :
        Matplotlib Axes object with the text added.

    """
    feature_indices = range(len(feature_labels))
    cross_corr_index_combinations = list(combinations(feature_indices, 2))
    integral_upper_bound_hrs = round(5 * max_lag_integrate / 60, 2)  # convert from frames to hours
    integral_strings = [
        rf"$|\int_{{0}}^{{{integral_upper_bound_hrs}}}\Delta C_{{{j+1}{k+1}}}(\tau) d\tau|$"
        for (j, k) in cross_corr_index_combinations
    ]
    strings_per_pc = [
        f"{feature_labels[j]}, {feature_labels[k]}: {string} = {integral:.2f}"
        for (j, k), string, integral in zip(
            cross_corr_index_combinations, integral_strings, delta_ccf_integral, strict=True
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
    ax: Axes,
    feature_labels: list[str],
    exp_decay_func: Literal["exponential_decay", "double_exponential_decay"],
) -> tuple[Axes, list[float]]:
    """Fit exponential decay to autocorrelation function (ACF) and add curve to existing plot.

    Parameters
    ----------
    acf
        Array of ACF values for a single feature/component across lags.
    lags
        Array of lag values corresponding to the ACF values.
    ax
        Matplotlib Axes object to add the fit curve to.
    feature_labels
        List of feature labels corresponding to the ACF curves being plotted.
    exp_decay_func
        String specifying which exponential decay function to fit (single or
        double exponential decay).

    Returns
    -------
    :
        Matplotlib Axes object with the fit curve added.
    :
        List of relaxation timescales extracted from the fit for each feature/component.

    """
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
    for i in range(len(feature_labels)):
        try:
            exp_fit, relaxation_time = fit_exp_decay_and_get_relaxation_timescale(
                acf[:, i], lags, exp_decay_func=exp_decay_func
            )
            relaxation_timescales.append(relaxation_time)
        except RuntimeError:
            logger.warning(
                "Could not fit [ %s ] to ACF of feature '%s', skipping plot step",
                exp_decay_func,
                feature_labels[i],
            )
            relaxation_timescales.append(np.nan)
            continue

        # get curve of fit exponential decay
        if exp_decay_func == "exponential_decay":
            acf_fit = exponential_decay(lags, *exp_fit)
            logger.debug(
                "Exponential fit for feature '%s': [%.3f + %.3f exp(%.3f tau)]",
                feature_labels[i],
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
    ax = _add_relaxation_timescale_to_plot(relaxation_timescales, feature_labels, ax)

    return ax, relaxation_timescales


def _make_all_acf_plots(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    output_path: Path,
    fit_double_exp: bool = True,
    bootstrap_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot autocorrelation function (ACF) curves and fits for a single dataset."""
    # unpack results
    lags: np.ndarray = correlation_dict["lags"][dataset_name]
    acf: np.ndarray = correlation_dict["acf"][dataset_name]

    # get string for shear stress to include in plot title
    dataset_config = load_dataset_config(dataset_name)
    if len(dataset_config.flow_conditions) > 1:
        logger.warning(
            "Multiple flow conditions found for dataset [ %s ]. "
            "Using shear stress from first flow condition for plot titles.",
            dataset_name,
        )
    shear_stress = dataset_config.flow_conditions[0].shear_stress

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
        component_labels=correlation_dict["features"][dataset_name],
        plot_title=f"Autocorrelation of PCA Components ({shear_stress} dyn/cm$^2$)",
        xlabel="Lag (hours)",
        linewidth=2.75,
    )
    ax.legend()
    ax.set_ylim(-0.25, 1.05)
    save_plot_to_path(
        fig,
        output_path,
        f"autocorrelation_{dataset_name}",
        show_and_close=False,
    )
    plt.close(fig)

    # fit single exponential decay to ACF
    fig, ax = _plot_acf_curves_together(
        dataset_name,
        correlation_dict,
        bootstrap_samples=bootstrap_samples,
        component_labels=correlation_dict["features"][dataset_name],
        plot_title=f"Autocorrelation of PCA Components ({shear_stress} dyn/cm$^2$)",
        xlabel="Lag (hours)",
        linewidth=2.75,
    )
    feats = correlation_dict["features"][dataset_name]
    ax, relaxation_timescales = _add_exp_fit_to_plot(
        acf_, lags_as_hours, ax, feats, exp_decay_func="exponential_decay"
    )
    # add relaxation timescales to correlation_dict for output
    correlation_dict["relaxation_timescales"][dataset_name] = relaxation_timescales
    save_plot_to_path(
        fig,
        output_path,
        f"autocorrelation_exp_fit_{dataset_name}",
        show_and_close=False,
    )
    plt.close(fig)

    if fit_double_exp:
        # fit double exponential decay to ACF
        fig, ax = _plot_acf_curves_together(
            dataset_name,
            correlation_dict,
            bootstrap_samples=bootstrap_samples,
            component_labels=correlation_dict["features"][dataset_name],
            plot_title=f"Autocorrelation of PCA Components ({shear_stress} dyn/cm$^2$)",
            xlabel="Lag (hours)",
            linewidth=2.75,
            linestyle="-",
        )
        ax, _ = _add_exp_fit_to_plot(
            acf_,
            lags_as_hours,
            ax,
            correlation_dict["features"][dataset_name],
            exp_decay_func="double_exponential_decay",
        )
        save_plot_to_path(
            fig,
            output_path,
            f"autocorrelation_double_exp_fit_{dataset_name}",
            show_and_close=False,
        )
        plt.close(fig)

    return correlation_dict


def _make_all_ccf_plots(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    output_path: Path,
    bootstrap_samples: int | None = None,
) -> None:
    """Plot cross-correlation function (CCF) curves and differences for a single dataset."""
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
    feature_labels: list[str] = correlation_dict["features"][dataset_name]

    # get string for shear stress to include in plot title
    dataset_config = load_dataset_config(dataset_name)
    shear_stress = dataset_config.flow_conditions[0].shear_stress

    # get the combinations of features for the cross-correlation plots
    # (we use the indices of the feature labels here because the features
    # themselves are stored in an array)
    feature_indices = range(len(feature_labels))
    cross_corr_index_combinations = list(combinations(feature_indices, r=2))
    # in `combinations` "r" is the number of elements to include in a combination

    # plot ccf with confidence intervals if available
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (j, k) in enumerate(cross_corr_index_combinations):
        lags_all_as_hours = 5 * lags / 60  # convert from frames (5 minutes) to hours
        ax.plot(lags_all_as_hours, ccf[:, i], label=f"({feature_labels[j]}, {feature_labels[k]})")
        if bootstrap_samples is not None:
            ax.fill_between(
                lags_all_as_hours,
                ccf_ci_lower[:, i],
                ccf_ci_upper[:, i],
                alpha=0.25,
                color=list(TABLEAU_COLORS.keys())[i],
                label="95% CI",
            )
    ax.set_title(f"Cross-Correlation of PCA Components ({shear_stress} dyn/cm$^2$)")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("CCF")
    ax.legend()
    ax.set_ylim(-0.25, 2)
    save_plot_to_path(
        fig,
        output_path,
        f"cross_correlation_{dataset_name}",
        show_and_close=False,
    )
    plt.close(fig)

    # plot delta ccf: difference between positive and negative lags
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (j, k) in enumerate(cross_corr_index_combinations):
        # delta ccf is symmetric around zero
        lags_symmetric = lags[1 + num_lags // 2 :]
        lags_symmetric_as_hours = 5 * lags_symmetric / 60
        ax.plot(
            lags_symmetric_as_hours,
            delta_ccf[:, i],
            label=f"({feature_labels[j]}, {feature_labels[k]})",
        )
        if bootstrap_samples is not None:
            ax.fill_between(
                lags_symmetric_as_hours,
                delta_ccf_ci_lower[:, i],
                delta_ccf_ci_upper[:, i],
                alpha=0.25,
                color=list(TABLEAU_COLORS.keys())[i],
                label="95% CI",
            )
    ax.set_title(f"$C_{{ij}}(\\tau) - C_{{ij}}(-\\tau)$ ({shear_stress} dyn/cm$^2$)")
    ax.set_xlabel("Lag $\\tau$ (hours)")
    ax.set_ylabel("$\Delta C_{ij}(\\tau)$")
    ax.legend()
    ax.set_ylim(-2, 2)
    # print integral of delta ccf near zero on plot
    ci_bounds = None
    if bootstrap_samples is not None:
        ci_bounds = (
            correlation_dict["delta_ccf_integral_ci_lower"][dataset_name],
            correlation_dict["delta_ccf_integral_ci_upper"][dataset_name],
        )
    ax = _add_delta_ccf_integral_to_plot(
        delta_ccf_integral, max_lag_integrate, ci_bounds, feature_labels, ax
    )
    save_plot_to_path(
        fig,
        output_path,
        f"cross_correlation_diff_{dataset_name}",
        show_and_close=False,
    )
    plt.close(fig)


def _plot_full_correlation_curves(
    dataset_name: str,
    correlation_dict: dict[str, dict[str, Any]],
    output_path: Path,
    bootstrap_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot correlation curves (auto- and cross-correlations) for a single dataset."""
    # plot acf and fit exponential decay
    # adds relaxation timescales to correlation_dict
    correlation_dict = _make_all_acf_plots(
        dataset_name,
        correlation_dict,
        output_path,
        bootstrap_samples=bootstrap_samples,
    )

    # plot ccf and difference between positive and negative lags
    _make_all_ccf_plots(
        dataset_name,
        correlation_dict,
        output_path,
        bootstrap_samples=bootstrap_samples,
    )

    return correlation_dict


def _plot_single_correlation_metric_vs_shear_stress(
    metric_values: list[np.ndarray],
    shear_stresses: np.ndarray,
    features: list[str],
    ci_bounds: list[tuple] | None = None,
    labels: list[str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single correlation summary metric as a function of shear stress across datasets.

    Example metrics include:

    - The integral of the difference between positive and negative lag CCFs
        for a given feature combination, integrated from lag = zero to some
        maximum lag. This quantifies the overall asymmetry of the CCF around
        zero lag, which is a signature of non-equilibrium dynamics.
    - The average of this integral across all feature combinations.
    - The relaxation timescales extracted from fitting exponential decay to
        the ACFs.

    Parameters
    ----------
    metric_values
        List of arrays of summary metric values for each dataset.
    shear_stresses
        Array of shear stress values corresponding to each dataset.
    features
        List of feature names corresponding to the indices in the metric arrays.
    ci_bounds
        Optional list of tuples of (lower bounds, upper bounds) for confidence
        intervals of the metric values for each dataset.
    labels
        Optional list of labels for each feature or feature combination.

    Returns
    -------
    :
        Matplotlib Figure and Axes objects containing the resulting plot.

    """
    # init plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # set default labels if none provided
    feature_indices = range(len(features))
    cross_corr_index_combinations = list(combinations(feature_indices, 2))
    if labels is None:
        labels = [f"({features[j]}, {features[k]})" for (j, k) in cross_corr_index_combinations]

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
    """Make and save plots of all correlation metrics as a function of shear stress.

    Wrapper method to plot multiple correlation summary metrics (e.g. delta CCF
    integral, relaxation timescales) as a function of shear stress across
    datasets by calling `_plot_single_correlation_metric_vs_shear_stress` for each metric.

    Parameters
    ----------
    correlation_dict
        Dictionary containing correlation results for multiple datasets, including
        the summary metrics to plot.
    list_of_datasets
        List of dataset names corresponding to the keys in `correlation_dict` to
        include in the plot.
    output_path
        Path to save the resulting plots.

    """

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

    feature_names = correlation_dict["features"][list_of_datasets[0]]

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        delta_ccf_integral_values,
        shear_stresses,
        feature_names,
        ci_bounds=delta_ccf_integral_ci_bounds,
    )
    ax.legend()
    ax.set_ylabel("$\\langle |\\Delta C_{ij} |\\rangle$")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "delta_ccf_integral_vs_shear_stress",
        show_and_close=False,
    )
    plt.close(fig)

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        mean_delta_ccf_integral,
        shear_stresses,
        feature_names,
        labels=["Mean over features"],
    )
    ax.legend()
    ax.set_ylabel("$\\overline{|\\Delta C_{ij}|}$")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "mean_delta_ccf_integral_vs_shear_stress",
        show_and_close=False,
    )
    plt.close(fig)

    fig, ax = _plot_single_correlation_metric_vs_shear_stress(
        relaxation_timescale_values,
        shear_stresses,
        features=feature_names,
        ci_bounds=relaxation_timescale_ci_bounds,
        labels=feature_names,
    )
    ax.set_ylim(0, None)
    ax.legend()
    ax.set_ylabel("Relaxation timescale (hours)")
    ax.set_xlabel("Shear Stress (dyn/cm$^2$)")
    save_plot_to_path(
        fig,
        output_path,
        "relaxation_time_vs_shear_stress",
        show_and_close=False,
    )
    plt.close(fig)


def _plot_relaxation_timescale_histogram(
    dataset_name: str, correlation_dict: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """Plot histograms of relaxation timescales from ACF fits to individual crop indices."""
    feature_labels = correlation_dict["features"][dataset_name]
    relaxation_timescales = correlation_dict["relaxation_timescale_per_crop"][dataset_name].copy()
    df_relaxations = pd.DataFrame(data=relaxation_timescales, columns=feature_labels)
    for feature in feature_labels:
        relax_max = df_relaxations[feature].quantile(0.75)  # use 75th percentile as upper bound
        df_relaxations[df_relaxations[feature] > relax_max] = np.nan
    df_relaxations = df_relaxations.melt(var_name="Feature", value_name="Relaxation timescale")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(
        data=df_relaxations,
        x="Relaxation timescale",
        hue="Feature",
        binwidth=1,
        stat="probability",
        discrete=True,
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlim(0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Relaxation timescale (hours)")
    save_plot_to_path(
        fig,
        output_path,
        f"relaxation_timescale_histogram_{dataset_name}",
        show_and_close=False,
    )
    plt.close(fig)


def plot_correlation_workflow_outputs(
    correlation_dict: dict[str, dict[str, Any]],
    bootstrap_samples: int | None = None,
    crop_pattern: Literal["grid", "tracked"] = "grid",
) -> None:
    """Make and save plots for correlation workflow outputs.

    **Workflow output**

    Creates and saves a series of summary plots for the correlation analysis
    results:
        - Plots of the autocorrelation functions (ACFs) for each datset in
          `correlation_dict`.
            - Fits exponential decay to ACFs to extract relaxation timescales
              and adds these to the ACF plots.
        - Plots of the cross-correlation functions (CCFs) for each dataset in
          `correlation_dict` with confidence intervals if bootstrap samples were
          provided.
        - Plots of the difference between positive and negative lag CCFs.
            - Computes integrals near lag = zero and adds to the plot.
        - Plots of these integrals as a function of shear stress across
          datasets.
        - Plots of the relaxation timescales extracted from the ACFs as a
          function of shear stress across datasets.

    Parameters
    ----------
    correlation_dict
        Dictionary containing correlation results for multiple datasets.
    bootstrap_samples
        Optional, number of bootstrap samples used to compute confidence
        intervals.

    """
    list_of_datasets = list(correlation_dict["lags"].keys())

    output_path = get_output_path("correlations", crop_pattern)

    # plot full correlation curves for each dataset
    for dataset_name in list_of_datasets:
        logger.info("Plotting correlation curves for dataset [ %s ]", dataset_name)
        correlation_dict = _plot_full_correlation_curves(
            dataset_name,
            correlation_dict,
            output_path,
            bootstrap_samples=bootstrap_samples,
        )

        _plot_relaxation_timescale_histogram(dataset_name, correlation_dict, output_path)

    # plot integrated difference between CCF for positive and
    # negative lags as a function of shear stress
    _plot_correlation_metrics_vs_shear_stress(
        correlation_dict,
        list_of_datasets,
        output_path,
    )
