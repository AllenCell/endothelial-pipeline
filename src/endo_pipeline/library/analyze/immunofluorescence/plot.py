from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colormaps

from endo_pipeline.io import save_plot_to_path


def get_shear_stress_label(df):
    """
    Generate a label for shear stress based on the dataframe.

    Args:
        df (pd.DataFrame): Subset of the dataframe for a specific dataset.

    Returns:
        str: Formatted label for the shear stress.
    """
    dataset = df["dataset"].iloc[0]
    shear_regime = df["shear_stress_regime"].iloc[0]
    duration_1 = df["duration_at_ss_1_hr"].iloc[0]
    shear_stress_value_1 = df["shear_stress_1"].iloc[0]
    duration_2 = df["duration_at_ss_2_hr"].iloc[0]
    shear_stress_value_2 = df["shear_stress_2"].iloc[0]

    data_label = f"{dataset}\n{shear_regime} shear stress"
    duration_label1 = f"{duration_1:.2f} hrs @ {shear_stress_value_1} dyn/cm²\n"
    duration_label2 = (
        f"{duration_2:.2f} hrs @ {shear_stress_value_2} dyn/cm²\n"
        if not np.isnan(shear_stress_value_2)
        else ""
    )
    return f"{data_label}\n{duration_label1}{duration_label2}"


def bootstrap_confidence_cov(
    df: pd.DataFrame, feature: str, n_bootstraps: int = 100
) -> tuple[float, float]:
    """
    Calculate the confidence interval for the coefficient of variation of a feature using
    bootstrapping.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with track level features
    feature: String
        The feature to calculate the coefficient of variation for
    n_bootstraps: Int
        The number of bootstraps to perform
    """
    covs = []

    for i in range(n_bootstraps):
        np.random.seed(i)
        sample = df[feature].sample(frac=1, replace=True)
        cov = np.std(sample) / np.mean(sample)
        covs.append(cov)
    return np.percentile(covs, 5), np.percentile(covs, 95)


def feature_density(
    df_all: pd.DataFrame,
    dataset_name_list: list[str],
    feature: str,
    feature_name: str,
    save_dir: Path,
    positions: list | None = None,
    xlim: int | None = None,
    ylim: int | None = None,
    pool_positions: bool = False,
    per_dataset: bool = False,
) -> None:
    """
    Plot feature density for multiple datasets, optionally looping through positions or
    pooling all positions.

    Parameters
    ----------
    df_all: pd.DataFrame
        The dataframe containing all datasets.
    dataset_name_list: list of strings
        A list of datasets, each corresponding to a dataset.
    feature: str
        The feature to plot.
    feature_name: str
        The name of the feature to use in the plot title.
    save_dir: Path
        Directory to save the plot.
    positions: list, optional
        A list of positions to loop through and plot densities for. If None, all positions are used.
    xlim: int, optional
        The x-axis limit for the plot. If None, no limit is applied.
    ylim: int, optional
        The y-axis limit for the plot. If None, no limit is applied.
    pool_positions: bool, optional
        If True, pool all positions together for each dataset. Default is False.
    per_dataset: bool, optional
        If True, plot densities for each dataset separately. Default is False.
    """
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(6, 6))

    datasets = df_all["dataset"].unique()
    cmap = colormaps.get_cmap("tab20")
    colors = {dataset: cmap(i / len(datasets)) for i, dataset in enumerate(datasets)}

    def calc_stats(df: pd.DataFrame, feature: str) -> tuple:
        mean = np.mean(df[feature])
        cov = np.std(df[feature]) / mean
        low, high = bootstrap_confidence_cov(df, feature)
        return mean, cov, low, high

    # Define line styles for positions
    line_styles = ["-", "--", "-.", ":"]  # Add more styles if needed

    for dataset_name in dataset_name_list:
        color = colors[dataset_name]

        if per_dataset:
            # Create a new figure for each dataset
            fig = plt.figure(figsize=(6, 6))

        if pool_positions:
            # Pool all positions together for the dataset
            df = df_all[df_all["dataset"] == dataset_name]
            shear_stress_label = get_shear_stress_label(df)

            total_nuclei = len(df)
            number_pos = df["position"].nunique()
            average_nuclei_per_pos = int(total_nuclei / number_pos)

            mean, cov, low, high = calc_stats(df, feature)
            label = (
                f"{shear_stress_label}All positions, "
                f"Avg N={average_nuclei_per_pos} per position\n"
                f"Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
            )
            ax = sb.kdeplot(
                df[feature], color=color, label=label, alpha=0.85, linestyle="-", linewidth=5
            )
        else:
            # Plot densities for individual positions
            dataset_positions = (
                positions or df_all[df_all["dataset"] == dataset_name]["position"].unique()
            )

            for idx, position in enumerate(dataset_positions):
                df = df_all[(df_all["dataset"] == dataset_name) & (df_all["position"] == position)]
                shear_stress_label = get_shear_stress_label(df)
                mean, cov, low, high = calc_stats(df, feature)
                label = (
                    f"{shear_stress_label}Pos={position}, "
                    f"N={len(df)}\nMean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
                )
                line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
                ax = sb.kdeplot(
                    df[feature],
                    color=color,
                    label=label,
                    alpha=0.85,
                    linestyle=line_style,
                    linewidth=5,
                )

        if per_dataset:
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")
            ax.legend(
                loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10, ncol=2, frameon=False
            )
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            plt.show()
            fname = f"{feature}_{dataset_name}_poolpos{pool_positions}_density_plot"
            save_plot_to_path(fig, save_dir, fname, transparent=True)
            plt.close(fig)

    if not per_dataset:
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10, ncol=2, frameon=False)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        plt.show()

        fname = f"{feature}_poolpos{pool_positions}_all_datasets_density_plot"
        save_plot_to_path(fig, save_dir, fname, transparent=True)


def plot_channel_intensity_histograms(
    df: pd.DataFrame,
    df_all: pd.DataFrame,
    column_names: list,
    dataset: str,
    positions: list,
    save_dir: Path,
) -> None:
    """
    Plot a 1 x n grid of histograms for the given column names,
    overlapping histograms for the specified positions.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        df_all (pd.DataFrame): The DataFrame containing all data for y-axis scaling.
        column_names (list): List of column names to plot histograms for.
        dataset (str): The dataset name to include in the titles.
        positions (list): List of positions to plot overlapping histograms.
        save_dir (Path): Directory to save the plot.
    """
    n = len(column_names)
    bins = 75
    colors = plt.cm.tab10.colors

    # Set font sizes
    title_fontsize, label_fontsize, tick_fontsize = 16, 14, 12
    title = get_shear_stress_label(df)

    # Create subplots
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)  # Ensure axes is always iterable

    for i, column_name in enumerate(column_names):
        # Calculate bin edges using df_all for consistent settings
        data_min = df_all[column_name].min()
        data_max = df_all[column_name].max()
        bin_width = (data_max - data_min) / bins  # Divide range into `bins` equal parts
        bin_edges = np.arange(data_min, data_max + bin_width, bin_width)

        for j, position in enumerate(positions):
            # Filter data for the current position
            position_data = df[df["position"] == position][column_name].dropna()
            if position_data.empty:
                print(
                    f"Skipping position '{position}' \
                    for column '{column_name}' due to NaN or empty data."
                )
                continue

            # Plot histogram for the current position
            axes[i].hist(
                position_data,
                bins=bin_edges,
                color=colors[j % len(colors)],
                alpha=0.5,
                label=f"Position {position} (N={len(position_data)})",
            )

        axes[i].set_title(title, fontsize=title_fontsize)
        axes[i].set_xlabel(f"{column_name} Intensity", fontsize=label_fontsize)
        axes[i].set_ylabel("Frequency", fontsize=label_fontsize)
        axes[i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[i].grid(True)
        axes[i].legend(fontsize=10)

    plt.show()
    fname = f"{dataset}_intensity_histograms"
    save_plot_to_path(fig, save_dir, fname, transparent=True)
