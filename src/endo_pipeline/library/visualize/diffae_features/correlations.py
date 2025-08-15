import logging
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from scipy.optimize import curve_fit

from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest import get_dataset_descriptions
from src.endo_pipeline.library.analyze.numerics import (
    exponential_decay,
    get_3d_index_combinations,
    power_law_decay,
)
from src.endo_pipeline.library.visualize.viz_base import init_plot

logger = logging.getLogger(__name__)


def plot_single_acf_curve(
    lags: np.ndarray,
    acf: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    figsize: tuple[int, int] = (12, 6),
    plot_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs,
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


def plot_acf_curves_together(
    lags: np.ndarray,
    acf_array: np.ndarray,
    figsize: tuple[int, int] = (12, 6),
    component_labels: list[str] | None = None,
    component_colors: list[str] | None = None,
    plot_title: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple ACF curves together for comparison."""
    fig, ax = init_plot(figsize=figsize)

    num_dims = acf_array.shape[1]
    for i in range(num_dims):
        fig, ax = plot_single_acf_curve(
            lags,
            acf_array[:, i],
            fig_ax=(fig, ax),
            plot_title=plot_title,
            label=component_labels[i] if component_labels else f"Component {i + 1}",
            color=component_colors[i] if component_colors else list(TABLEAU_COLORS.keys())[i],
            **kwargs,
        )
    return fig, ax


def _parse_dataset_description(dataset_description: str) -> str:
    """Parse dataset description for better readability in plot titles."""
    # replace underscores with spaces for better readability
    description_parsed = dataset_description.replace("_", " ")
    # find [0-9]dyncm2, put comma and space before, put a space between number and unit,
    # and change dyncm2 to dyn/cm^2 for better readability
    description_parsed = re.sub(r"(\d+)dyncm2", r", \1 dyn/cm^2", description_parsed)
    # turn capital 'S' into lowercase 's' for shear stress
    description_parsed = description_parsed.replace(" Shear Stress", " shear stress")
    return description_parsed


def plot_correlation_workflow_outputs(correlation_dict: dict[str, dict]) -> None:
    """Plot correlation workflow outputs."""
    list_of_datasets = list(correlation_dict["lags"].keys())
    dataset_descriptions = get_dataset_descriptions(
        list_of_datasets, simple=True, include_duration=False, include_shear_rate=True
    )
    output_path = get_output_path("correlations")

    for dataset_name in list_of_datasets:
        # unpack results
        lags = correlation_dict["lags"][dataset_name]
        num_lags = len(lags)
        acf = correlation_dict["acf"][dataset_name]
        ccf = correlation_dict["ccf"][dataset_name]
        delta_ccf = correlation_dict["delta_ccf"][dataset_name]

        # get string for dataset description
        dataset_description = _parse_dataset_description(dataset_descriptions[dataset_name])

        # plot acf for positive lags
        # (acf is symmetric around zero)
        index_positive = lags > 0
        lags_ = lags[index_positive]
        lags_as_hours = 5 * lags_ / 60  # convert from frames (5 minutes) to hours
        acf_ = acf[index_positive]
        fig, ax = plot_acf_curves_together(
            lags_as_hours,
            acf_,
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

        # fit exponential decay to ACF
        fig, ax = plot_acf_curves_together(
            lags_as_hours,
            acf_,
            component_labels=["PC1", "PC2", "PC3"],
            plot_title=f"Autocorrelation of PCA Components ({dataset_description})",
            xlabel="Lag (hours)",
            linewidth=2.75,
        )
        for i in range(3):
            acf_where_positive = acf_[:, i] > 0
            lags_pos = lags_as_hours[acf_where_positive]
            acf_pos = acf_[acf_where_positive, i]
            exp_fit, _ = curve_fit(exponential_decay, lags_pos, acf_pos, p0=(1, 0.01))
            relaxation_time = 5 * (1 / exp_fit[1]) / 60  # convert to hours
            logger.info(
                "PC %d relaxation timescale (exponential): %.2f hrs.",
                i + 1,
                relaxation_time,
            )
            acf_fit = exponential_decay(lags_as_hours, *exp_fit)
            ax.plot(lags_as_hours, acf_fit, "k--", linewidth=2.0, alpha=0.85, label="")
        ax.legend()
        ax.set_ylim(-0.25, 1.05)
        save_plot_to_path(
            fig,
            output_path,
            f"autocorrelation_exp_fit_{dataset_name}",
        )

        # fit power law decay to ACF
        fig, ax = plot_acf_curves_together(
            lags_as_hours,
            acf_,
            component_labels=["PC1", "PC2", "PC3"],
            plot_title=f"Autocorrelation of PCA Components ({dataset_description})",
            xlabel="Lag (hours)",
            linewidth=2.75,
        )
        for i in range(3):
            # only fit power law to positive lags
            acf_where_positive = acf_[:, i] > 0
            lags_pos = lags_as_hours[acf_where_positive]
            acf_pos = acf_[acf_where_positive, i]
            # fit power law decay by fitting linear decay to log-log transformed data
            power_fit, _ = curve_fit(power_law_decay, lags_pos, acf_pos)
            relaxation_time = 5 * (1 / power_fit[1]) / 60
            logger.info("PC %d relaxation timescale (power law): %.2f hrs.", i + 1, relaxation_time)
            acf_fit = power_law_decay(lags_as_hours, *power_fit)
            ax.plot(lags_as_hours, acf_fit, "k:", linewidth=2.5, label="")
        ax.legend()
        ax.set_ylim(-0.25, 1.05)
        save_plot_to_path(
            fig,
            output_path,
            f"autocorrelation_power_fit_{dataset_name}",
        )

        # plot ccf
        index_combinations = get_3d_index_combinations()
        fig, ax = init_plot(figsize=(12, 6))
        for i, (j, k) in enumerate(index_combinations):
            lags_all_as_hours = 5 * lags / 60  # convert from frames (5 minutes) to hours
            ax.plot(lags_all_as_hours, ccf[:, i], label=f"(PC{j+1}, PC{k+1})")

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

        # plot delta ccf
        fig, ax = init_plot(figsize=(12, 6))
        for i, (j, k) in enumerate(index_combinations):
            # delta ccf is symmetric around zero
            lags_symmetric = lags[1 + num_lags // 2 :]
            lags_symmetric_as_hours = 5 * lags_symmetric / 60
            ax.plot(lags_symmetric_as_hours, delta_ccf[:, i], label=f"(PC{j+1}, PC{k+1})")
        ax.set_title("$C_{ij}(\\tau) - C_{ij}(-\\tau)$" + f" ({dataset_description})")
        ax.set_xlabel("Lag $\\tau$ (hours)")
        ax.set_ylabel("$\Delta C_{ij}(\\tau)$")
        ax.legend()
        ax.set_ylim(-0.75, 0.75)
        save_plot_to_path(
            fig,
            output_path,
            f"cross_correlation_diff_{dataset_name}",
        )
        # log summary statistics
        logger.info(
            "Minimum, maximum, and mean of delta CCF for dataset [ %s ]:" " [ %s, %s, %s ]",
            dataset_name,
            np.min(delta_ccf, axis=0),
            np.max(delta_ccf, axis=0),
            np.mean(delta_ccf, axis=0),
        )
