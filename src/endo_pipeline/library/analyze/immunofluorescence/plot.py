from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from src.endo_pipeline.io import save_plot_to_path


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


# Dataset information with flow rates, flow regimes, and colors
DATASET_INFO = {
    "20250509_20X_IF2": {
        "flow_rate": 0,
        "flow_regime": "24hr No Flow",
        "color": "#1B9E77",
    },  # Green
    "20250509_20X_IF3": {
        "flow_rate": 0,
        "flow_regime": "24hr No Flow",
        "color": "#D95F02",
    },  # Orange
    "20250509_20X_IF12": {
        "flow_rate": 5.82,
        "flow_regime": "24hr Low Flow",
        "color": "#7570B3",
    },  # Purple
    "20250509_20X_IF5": {
        "flow_rate": 5.98,
        "flow_regime": "24hr Low Flow",
        "color": "#E7298A",
    },  # Pink
    "20250509_20X_IF7": {
        "flow_rate": 10.96,
        "flow_regime": "24hr Int. Flow",
        "color": "#66A61E",
    },  # Lime Green
    "20250509_20X_IF1": {
        "flow_rate": 20.8,
        "flow_regime": "24hr High Flow",
        "color": "#A6761D",
    },  # Brown
    "20250509_20X_IF9": {
        "flow_rate": 23.67,
        "flow_regime": "24hr High Flow",
        "color": "#1170AA",
    },  # Blue
}


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
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(6, 6))

    def calc_stats(df: pd.DataFrame, feature: str) -> tuple:
        mean = np.mean(df[feature])
        cov = np.std(df[feature]) / mean
        low, high = bootstrap_confidence_cov(df, feature)
        return mean, cov, low, high

    # Define line styles for positions
    line_styles = ["-", "--", "-.", ":"]  # Add more styles if needed

    for dataset_name in dataset_name_list:
        if dataset_name not in DATASET_INFO:
            print(f"Skipping dataset {dataset_name} as it is not in DATASET_INFO.")
            continue

        info = DATASET_INFO[dataset_name]
        color = info["color"]
        flow_rate = info["flow_rate"]
        flow_regime = info["flow_regime"]

        if per_dataset:
            # Create a new figure for each dataset
            fig = plt.figure(figsize=(6, 6))

        if pool_positions:
            # Pool all positions together for the dataset
            df = df_all[df_all["dataset"] == dataset_name]
            if df.empty:
                print(f"Skipping dataset {dataset_name} due to no data.")
                continue

            mean, cov, low, high = calc_stats(df, feature)
            label = (
                f"{flow_regime}, {flow_rate} dyn/cm², All positions, "
                f"N={len(df)}, Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
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
                if df.empty:
                    print(
                        f"Skipping position {position} for dataset {dataset_name} due to no data."
                    )
                    continue

                mean, cov, low, high = calc_stats(df, feature)
                label = (
                    f"{flow_regime}, {flow_rate} dyn/cm², Pos={position}, "
                    f"N={len(df)}, Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
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
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            plt.tight_layout()
            plt.show()
            fname = f"{feature}_{dataset_name}_poolpos{pool_positions}_density_plot"
            save_plot_to_path(fig, save_dir, fname, transparent=True)
            plt.close(fig)

    if not per_dataset:
        ax.set_xlabel(f"{feature_name}")
        ax.set_ylabel("Density")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        plt.tight_layout()
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
    colors = plt.cm.tab10.colors  # Colormap with 10 distinct colors

    # Set font sizes
    title_fontsize, label_fontsize, tick_fontsize = 16, 14, 12

    # Retrieve dataset info
    info = DATASET_INFO[dataset]
    flow_regime, flow_rate = info["flow_regime"], info["flow_rate"]

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

        axes[i].set_title(f"{flow_regime}, {flow_rate} dyn/cm²\n{dataset}", fontsize=title_fontsize)
        axes[i].set_xlabel(f"{column_name} Intensity", fontsize=label_fontsize)
        axes[i].set_ylabel("Frequency", fontsize=label_fontsize)
        axes[i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[i].grid(True)
        axes[i].legend(fontsize=10)

    plt.show()
    fname = f"{dataset}_intensity_histograms"
    save_plot_to_path(fig, save_dir, fname, transparent=True)


def feature_boxplot_vs_flowrate(
    df_all: pd.DataFrame,
    dataset_name_list: list[str],
    feature: str,
    save_dir: Path,
) -> None:
    """
    Create a boxplot of a feature against flow rate across datasets.

    Parameters
    ----------
    df_all: pd.DataFrame
        The dataframe containing all datasets.
    dataset_name_list: list of strings
        List of dataset names to include.
    feature: str
        The feature to plot.
    save_dir: Path
        Directory to save the plot.
    ylim: int, optional
        Y-axis limit.
    """
    records = []  # Collect relevant data for the boxplot

    for dataset_name in dataset_name_list:
        if dataset_name not in DATASET_INFO:
            print(f"Skipping dataset {dataset_name} - not in DATASET_INFO.")
            continue

        df = df_all[df_all["dataset"] == dataset_name]
        if df.empty:
            print(f"No data for {dataset_name}, skipping.")
            continue

        flow_rate = DATASET_INFO[dataset_name]["flow_rate"]
        flow_regime = DATASET_INFO[dataset_name]["flow_regime"]
        color = DATASET_INFO[dataset_name]["color"]

        for value in df[feature].dropna():
            records.append(
                {
                    "flow_rate": flow_rate,
                    "flow_regime": flow_regime,
                    "dataset": dataset_name,
                    "feature_value": value,
                    "color": color,
                }
            )

    if not records:
        print("No data available to plot.")
        return

    plot_df = pd.DataFrame(records)
    plot_df = plot_df.sort_values("flow_rate")

    plt.figure(figsize=(8, 6))
    ax = sb.boxplot(
        x="flow_rate",
        y="feature_value",
        hue="dataset",
        data=plot_df,
        palette={d: DATASET_INFO[d]["color"] for d in plot_df["dataset"].unique()},
    )

    ax.set_xlabel("Flow Rate (dyn/cm²)", fontsize=12)
    ax.set_ylabel(f"{feature}", fontsize=12)

    # Optional: show number of points per box
    n_obs = plot_df.groupby("flow_rate").size()
    xticks = [f"{fr}\n(N={n_obs[fr]})" for fr in n_obs.index]
    ax.set_xticklabels(xticks)

    plt.title(f"{feature} vs Flow Rate", fontsize=14)
    ax.legend_.remove()
    plt.tight_layout()
    plt.show()

    fname = f"{feature}_vs_flowrate_boxplot"
    save_plot_to_path(ax.figure, save_dir, fname, transparent=True)
    plt.close()


def feature_boxplot_vs_sample_size(
    df_all: pd.DataFrame,
    dataset_name_list: list[str],
    feature: str,
    save_dir: Path,
) -> None:
    """
    Create a boxplot of a feature against sample size (N), pooling all positions,
    and annotate with flow rate. Sorted by sample size.

    Parameters
    ----------
    df_all: pd.DataFrame
        The dataframe containing all datasets.
    dataset_name_list: list[str]
        List of dataset names to include.
    feature: str
        The feature to plot.
    save_dir: Path
        Directory to save the plot.
    """
    records = []
    label_info = {}

    for dataset_name in dataset_name_list:
        if dataset_name not in DATASET_INFO:
            print(f"Skipping dataset {dataset_name} - not in DATASET_INFO.")
            continue

        df = df_all[df_all["dataset"] == dataset_name]
        if df.empty:
            print(f"No data for {dataset_name}, skipping.")
            continue

        info = DATASET_INFO[dataset_name]
        flow_rate = info["flow_rate"]
        color = info["color"]
        sample_size = len(df)

        label = f"N={sample_size}\n({flow_rate:.2f})"
        label_info[dataset_name] = {"label": label, "sample_size": sample_size, "color": color}

        for value in df[feature].dropna():
            records.append({"feature_value": value, "dataset": dataset_name, "label": label})

    if not records:
        print("No data available to plot.")
        return

    plot_df = pd.DataFrame(records)

    # Sort labels by sample size
    label_order = pd.DataFrame(label_info).T.sort_values("sample_size").label.tolist()

    plot_df["label"] = pd.Categorical(plot_df["label"], categories=label_order, ordered=True)

    plt.figure(figsize=(max(10, len(label_order) * 0.6), 6))
    ax = sb.boxplot(
        x="label",
        y="feature_value",
        width=0.15,
        hue="dataset",
        data=plot_df,
        palette={ds: label_info[ds]["color"] for ds in plot_df["dataset"].unique()},
    )

    ax.set_xlabel("Sample Size (N) and Flow Rate (dyn/cm²)", fontsize=12)
    ax.set_ylabel(feature.replace("_", " "), fontsize=12)
    ax.set_title(f"{feature.replace('_', ' ')} vs Sample Size", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels())
    ax.legend_.remove()

    plt.tight_layout()
    plt.show()

    fname = f"{feature}_vs_sample_size_boxplot"
    save_plot_to_path(ax.figure, save_dir, fname, transparent=True)
    plt.close()


def feature_scatter_vs_flowrate(
    df_all: pd.DataFrame,
    dataset_name_list: list[str],
    feature: str,
    save_dir: Path,
    by_flowrate: bool = True,
) -> None:
    """
    Create a scatter plot of the mean of a feature vs. either flow rate or sample size.

    Parameters
    ----------
    df_all: pd.DataFrame
        The dataframe containing all datasets.
    dataset_name_list: list of strings
        List of dataset names to include.
    feature: str
        The feature to plot.
    save_dir: Path
        Directory to save the plot.
    by_flowrate: bool, optional
        If True, plot mean vs. flow rate; if False, plot mean vs. sample size.
    """
    records = []

    for dataset_name in dataset_name_list:
        if dataset_name not in DATASET_INFO:
            print(f"Skipping dataset {dataset_name} - not in DATASET_INFO.")
            continue

        df = df_all[df_all["dataset"] == dataset_name]
        if df.empty:
            print(f"No data for {dataset_name}, skipping.")
            continue

        # Calculate the mean of the feature for the dataset
        mean_value = df[feature].mean()

        # Get flow rate or sample size (N)
        flow_rate = DATASET_INFO[dataset_name]["flow_rate"]
        sample_size = len(df)

        x_value = flow_rate if by_flowrate else sample_size

        records.append({"dataset": dataset_name, "mean_value": mean_value, "x_value": x_value})

    if not records:
        print("No data available to plot.")
        return

    plot_df = pd.DataFrame(records)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    ax = sb.scatterplot(
        x="x_value",
        y="mean_value",
        hue="dataset",
        data=plot_df,
        palette={d: DATASET_INFO[d]["color"] for d in plot_df["dataset"].unique()},
        s=75,
    )

    if by_flowrate:
        ax.set_xlabel("Flow Rate (dyn/cm²)", fontsize=12)
    else:
        ax.set_xlabel("Sample Size (N)", fontsize=12)

    ax.set_ylabel(f"Mean {feature}", fontsize=12)
    ax.set_title(f"Mean {feature} vs {'Flow Rate' if by_flowrate else 'Sample Size'}", fontsize=14)
    ax.legend_.remove()
    plt.tight_layout()
    plt.show()

    fname = f"{feature}_vs_flowrate_sample_size_scatter_plot"
    save_plot_to_path(ax.figure, save_dir, fname, transparent=True)
    plt.close()
