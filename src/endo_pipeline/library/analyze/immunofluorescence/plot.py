from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image, save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    gfp_max_proj,
    max_proj_640,
)
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM, FONTSIZE_SMALL


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


def calc_stats(df: pd.DataFrame, feature: str) -> tuple:
    mean = df[feature].mean()
    cov = df[feature].std() / mean if mean != 0 else np.nan
    low, high = bootstrap_confidence_cov(df, feature)
    return mean, cov, low, high


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
    hide_labels: bool = False,
    stagger_vertical: bool = False,
) -> None:
    """
    Plot feature density distributions for multiple datasets,
    optionally split or pooled by positions.
    """

    # line styles cycle cleanly for multiple positions
    line_styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 5, 1, 5))]

    # consistent color assignment per dataset
    # color map
    all_dataset_names = df_all["dataset"].unique().tolist()
    n = len(all_dataset_names)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        colors = {dataset: cmap(i) for i, dataset in enumerate(all_dataset_names)}
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = {dataset: cmap(i) for i, dataset in enumerate(all_dataset_names)}
    else:
        cmap = plt.get_cmap("hsv")
        colors = {dataset: cmap(i / n) for i, dataset in enumerate(all_dataset_names)}

    with plt.rc_context({"font.size": FONTSIZE_MEDIUM}):

        # -------------------------
        # Case 1: One combined plot
        # -------------------------
        if not per_dataset:
            fig, ax = plt.subplots(figsize=(10, 9))

            for dataset_name in dataset_name_list:
                df_dataset = df_all[df_all["dataset"] == dataset_name]
                ds_positions = positions or df_dataset["position"].unique()
                color = colors[dataset_name]

                avg_n_per_pos = len(df_dataset) / df_dataset["position"].nunique()

                if pool_positions:
                    df = df_dataset
                    mean, cov, low, high = calc_stats(df, feature)
                    shear_label = get_shear_stress_label(df)
                    label = (
                        f"{shear_label}"
                        f"Avg N / Pos={round(avg_n_per_pos)}, Mean={round(mean)}\nCOV={cov:.2f}, "
                        f"CI=[{low:.2f}, {high:.2f}]\n"
                    )

                    sb.kdeplot(df[feature], ax=ax, color=color, label=label, linewidth=3)

                else:
                    # separate lines per position
                    for j, pos in enumerate(ds_positions):
                        df = df_dataset[df_dataset["position"] == pos]
                        if df.empty:
                            continue

                        mean, cov, low, high = calc_stats(df)
                        shear_label = get_shear_stress_label(df)
                        line_style = line_styles[j % len(line_styles)]

                        label = (
                            f"{dataset_name} - Pos {pos} "
                            f"(N={len(df)}, Mean={mean:.2f}, COV={cov:.2f})"
                        )

                        sb.kdeplot(
                            df[feature],
                            ax=ax,
                            color=color,
                            linestyle=line_style,
                            label=label,
                            linewidth=3,
                            alpha=0.9,
                        )

            # formatting
            if not hide_labels:
                ax.set_xlabel(feature_name)
                ax.set_ylabel("Density")
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if xlim is not None:
                ax.set_xlim(0, xlim)
            if ylim is not None:
                ax.set_ylim(0, ylim)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            handles, labels = ax.get_legend_handles_labels()
            # reverse them
            handles = handles[::-1]
            labels = labels[::-1]
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),  # outside to the right
                frameon=False,
                fontsize=FONTSIZE_SMALL,
            )

            plt.tight_layout()

            fname = f"{feature}_poolpos{pool_positions}_all_datasets_density"
            fname += "_nolabels" if hide_labels else ""
            save_plot_to_path(fig, save_dir, fname, transparent=True)
            plt.show()
            plt.close(fig)
            return

        # ---------------------------------
        # Case 2: Separate plot per dataset
        # ---------------------------------
        for dataset_name in dataset_name_list:
            fig, ax = plt.subplots(figsize=(10, 9))

            df_dataset = df_all[df_all["dataset"] == dataset_name]
            ds_positions = positions or df_dataset["position"].unique()
            color = colors[dataset_name]

            if pool_positions:
                df = df_dataset
                mean, cov, low, high = calc_stats(df)
                shear_label = get_shear_stress_label(df)

                label = (
                    f"{shear_label}"
                    f"N={len(df)}, Mean={mean:.2f}\nCOV={cov:.2f}, "
                    f"CI=[{low:.2f}, {high:.2f}]\n"
                )

                sb.kdeplot(df[feature], ax=ax, color=color, label=label, linewidth=3)

            else:
                # individual positions
                for j, pos in enumerate(ds_positions):
                    df = df_dataset[df_dataset["position"] == pos]
                    if df.empty:
                        continue

                    mean, cov, low, high = calc_stats(df)
                    shear_label = get_shear_stress_label(df)
                    line_style = line_styles[j % len(line_styles)]

                    label = f"Pos {pos} " f"(N={len(df)}, Mean={mean:.2f}, COV={cov:.2f})"

                    sb.kdeplot(
                        df[feature],
                        ax=ax,
                        color=color,
                        linestyle=line_style,
                        label=label,
                        linewidth=3,
                    )

            # formatting
            if not hide_labels:
                ax.set_xlabel(feature_name)
                ax.set_ylabel("Density")
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if xlim is not None:
                ax.set_xlim(0, xlim)
            if ylim is not None:
                ax.set_ylim(0, ylim)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            handles, labels = ax.get_legend_handles_labels()
            # reverse them
            handles = handles[::-1]
            labels = labels[::-1]
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),  # outside to the right
                frameon=False,
                fontsize=FONTSIZE_SMALL,
            )

            plt.tight_layout()
            plt.show()
            fname = f"{feature}_{dataset_name}_poolpos{pool_positions}_density"
            fname += "_nolabels" if hide_labels else ""
            save_plot_to_path(fig, save_dir, fname, transparent=True)

            plt.close(fig)


def stacked_feature_density(
    df_all,
    dataset_name_list,
    feature,
    feature_name,
    save_dir: Path,
    y_offset_step: float = 0.0001,
    xlim: float | None = None,
) -> None:
    """
    Plot multiple KDEs staggered vertically so they overlay along x-axis.

    Parameters
    ----------
    df_all : pd.DataFrame
        The dataframe containing all datasets.
    dataset_name_list : list of str
        List of datasets to plot.
    feature : str
        Feature column to plot.
    feature_name : str
        Name for x-axis label.
    save_dir : Path
        Directory to save the plot.
    positions : list, optional
        Positions to plot. If None, use all positions per dataset.
    y_offset_step : float
        Vertical spacing between curves.
    xlim: float or none
        Limit for x axis
    """

    fig, ax = plt.subplots(figsize=(7, 8))

    # color map
    all_dataset_names = df_all["dataset"].unique().tolist()
    n = len(all_dataset_names)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        colors = {dataset: cmap(i) for i, dataset in enumerate(all_dataset_names)}
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = {dataset: cmap(i) for i, dataset in enumerate(all_dataset_names)}
    else:
        cmap = plt.get_cmap("hsv")
        colors = {dataset: cmap(i / n) for i, dataset in enumerate(all_dataset_names)}

    curve_index = 0.0  # for vertical staggering
    for dataset_name in dataset_name_list:
        df_dataset = df_all[df_all["dataset"] == dataset_name]
        color = colors[dataset_name]

        mean, cov, low, high = calc_stats(df_dataset, feature)
        shear_label = get_shear_stress_label(df_dataset)

        avg_n_per_pos = len(df_dataset) / df_dataset["position"].nunique()

        df = df_dataset
        label = (
            f"{shear_label}"
            f"Avg N / Pos={round(avg_n_per_pos)}, Mean={round(mean)},\nCOV={cov:.2f}, "
            f"CI=[{low:.2f}, {high:.2f}]\n"
        )
        sb.kdeplot(df[feature], ax=ax, color=color, linewidth=3, label=label)
        # apply vertical offset
        ax.lines[-1].set_ydata(ax.lines[-1].get_ydata() + curve_index)
        curve_index += y_offset_step

    # formatting
    all_y = np.concatenate([line.get_ydata() for line in ax.lines])
    ax.set_ylim(0, all_y.max() * 1.05)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    y_ticks = np.arange(0, all_y.max() * 1.05, y_offset_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([""] * len(y_ticks))
    ax.set_ylabel(f"Density (interval = {y_offset_step} a.u.)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    # reverse them
    handles = handles[::-1]
    labels = labels[::-1]
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),  # outside to the right
        frameon=False,
        fontsize=FONTSIZE_SMALL,
    )
    fname = f"{feature}_staggered_vertical_density"
    save_plot_to_path(fig, save_dir, fname, transparent=True)
    plt.show()
    plt.close(fig)


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

        axes[i].set_title(title, fontsize=FONTSIZE_LARGE)
        axes[i].set_xlabel(f"{column_name} Intensity", fontsize=FONTSIZE_MEDIUM)
        axes[i].set_ylabel("Frequency", fontsize=FONTSIZE_MEDIUM)
        axes[i].tick_params(axis="both", labelsize=FONTSIZE_SMALL)
        axes[i].grid(True)
        axes[i].legend(fontsize=10)

    plt.show()
    fname = f"{dataset}_intensity_histograms"
    save_plot_to_path(fig, save_dir, fname, transparent=True)


def if_dataset_contact_sheet(df: pd.DataFrame, dataset_list: list[str], output_dir: Path) -> None:
    """
    Generate contact sheets for GFP and SMAD1 images grouped by date.

    Images within each contact sheet are contrast-matched across datasets
    using the 1st and 99th percentile intensity values.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the dataset information.
    dataset_list: list of str
        List of dataset names to include in the contact sheets.
    output_dir: Path
        Directory to save the contact sheets.
    """
    img_manifest = load_image_manifest("image_zarr")

    gfp_img_list, smad1_img_list, data_labels = [], [], []

    for dataset_name in dataset_list:
        df_dataset = df[df["dataset"] == dataset_name]
        dataset_config = load_dataset_config(dataset_name)
        positions = dataset_config.zarr_positions[:6]  # limit to first 6 positions
        data_label = get_shear_stress_label(df_dataset)
        data_labels.append(data_label)

        for position in positions:
            img_location = get_image_location_for_dataset(img_manifest, dataset_config, position)
            img = load_image(img_location, level=1, read=False)

            img_max_640 = max_proj_640(img, frame=0)
            img_max_gfp = gfp_max_proj(img, frame=0)
            # crop image to center. remove 200 pixels from each side
            img_max_640_center = img_max_640[200:-200, 200:-200]
            img_max_gfp_center = img_max_gfp[200:-200, 200:-200]

            smad1_img_list.append(img_max_640_center)
            gfp_img_list.append(img_max_gfp_center)

        # flatten img lists to apply matching contrast stretching
        gfp_flat = np.concatenate([img.flatten() for img in gfp_img_list])
        smad1_flat = np.concatenate([img.flatten() for img in smad1_img_list])
        gfp_vmin, gfp_vmax = np.percentile(gfp_flat, [1, 99])
        smad1_vmin, smad1_vmax = np.percentile(smad1_flat, [1, 99])

    contrasted_gfp_img_list = [
        contrast_stretching(img, "min-max", custom_range=(gfp_vmin, gfp_vmax))
        for img in gfp_img_list
    ]
    contrasted_smad1_img_list = [
        contrast_stretching(img, "min-max", custom_range=(smad1_vmin, smad1_vmax))
        for img in smad1_img_list
    ]

    # create contact sheets
    n_cols = len(positions)
    n_rows = len(dataset_list)
    for img_content, panels in zip(
        ["SMAD1", "CDH5"], [contrasted_smad1_img_list, contrasted_gfp_img_list]
    ):
        fig = make_contact_sheet(
            panels=panels,
            max_rows=n_rows,
            max_cols=n_cols,
            col_titles=positions,
            row_titles=data_labels,
            direction="left-right first",
            gridspec_kwargs={"wspace": 0.03, "hspace": 0.0},
            fig_kwargs={"figsize": (n_cols * 3, n_rows * 3)},
        )
        plt.show(fig)
        save_plot_to_path(fig, output_dir, f"{img_content}_contact_sheet_{dataset_name[:8]}")
