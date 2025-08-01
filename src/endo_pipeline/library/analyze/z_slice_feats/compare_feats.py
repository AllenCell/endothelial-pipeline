import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from src.endo_pipeline.library.analyze.diffae_manifest import get_pc_column_names
from src.endo_pipeline.library.analyze.immunofluorescence.plot import bootstrap_confidence_cov
from src.endo_pipeline.library.visualize import viz_base


def feature_density(
    df_all: pd.DataFrame,
    # dataset_name_list: list[str],
    feature: str,
    # save_dir: str,
    xlim: list[float],
    pool_positions: bool = False,
    title: str | None = None,
) -> tuple[plt.Figure, sb.axisgrid.FacetGrid]:

    fig = plt.figure(figsize=(15, 6))

    def calc_stats(df: pd.DataFrame, feature: str) -> tuple:
        mean = np.mean(df[feature])
        cov = np.std(df[feature]) / mean
        low, high = bootstrap_confidence_cov(df, feature)
        return mean, cov, low, high

    if pool_positions:
        mean, cov, low, high = calc_stats(df_all, feature)
        label = f"N={len(df_all)}, Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
        ax = sb.kdeplot(df_all[feature], label=label, alpha=0.85, linestyle="-")
    else:
        for position, df_position in df_all.groupby("position"):
            mean, cov, low, high = calc_stats(df_position, feature)
            label = (
                f"Pos={position}, "
                f"N={len(df_position)}, Mean={mean:.2f}, COV={cov:.2f}, CI=({low:.2f}, {high:.2f})"
            )
            ax = sb.kdeplot(df_position[feature], label=label, alpha=0.85)

        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Density")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.set_xlim(xlim[0], xlim[1])
        # ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    return fig, ax

    # fname = f"{feature}_poolpos{pool_positions}_all_datasets_density_plot"
    # output_dir = save_dir / fname
    # save_plot(fig, str(output_dir), transparent=True)


# %%
def plot_scatter_by_position_and_frame(
    df: pd.DataFrame,
    target_frame: int,
    bounds: list,
) -> tuple[plt.Figure, np.ndarray]:

    fig, ax = viz_base.init_subplots(figsize=(15, 5))
    pc_column_names = get_pc_column_names(df, [0, 1, 2])

    target_frame = 0

    for position, df_pos in df.groupby("position"):
        df_ = df_pos[df_pos["frame_number"] == target_frame]
        # first plot: PC1 v PC2
        ax[0].scatter(df_[pc_column_names[0]], df_[pc_column_names[1]], s=20)

        # second plot: PC1 v PC3
        ax[1].scatter(df_[pc_column_names[0]], df_[pc_column_names[2]], s=20, label=position)

    ax[0].set_xlim(bounds[0])
    ax[0].set_ylim(bounds[1])
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    ax[1].set_xlim(bounds[0])
    ax[1].set_ylim(bounds[2])
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC3")

    ax[1].legend(loc=(1.05, 0.75))
    fig.suptitle(f"Frame {target_frame}")

    return fig, ax


def plot_distribution_by_position_and_frame(
    df: pd.DataFrame, target_frame: int
) -> tuple[plt.Figure, np.ndarray]:

    fig, ax = viz_base.init_subplots(1, 3, figsize=(15, 5))
    pc_column_names = get_pc_column_names(df, [0, 1, 2])

    target_frame = 0

    for position, df_pos in df.groupby("position"):
        df_ = df_pos[df_pos["frame_number"] == target_frame]

        ax[0].hist(df_[pc_column_names[0]], bins=50, alpha=0.5, label=position)
        ax[1].hist(df_[pc_column_names[1]], bins=50, alpha=0.5, label=position)
        ax[2].hist(df_[pc_column_names[2]], bins=50, alpha=0.5, label=position)

    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("Count")

    ax[1].set_xlabel("PC2")
    ax[1].set_ylabel("Count")

    ax[2].set_xlabel("PC3")
    ax[2].set_ylabel("Count")

    ax[2].legend(loc=(1.05, 0.75))
    fig.suptitle(f"Frame {target_frame}")

    return fig, ax


def plot_distribution_by_frame(
    df_list: list[pd.DataFrame], df_info: list[str], target_frame: int, position: int | None = None
) -> tuple[plt.Figure, np.ndarray]:

    pc_column_names = get_pc_column_names(df_list[0], [0, 1, 2])

    target_frame = 0

    for df, z_slice in zip(df_list, df_info):
        if position is not None:
            df = df[df["position"] == f"P{position}"]

        fig, ax = viz_base.init_subplots(1, 3, figsize=(15, 5))
        df_ = df[df["frame_number"] == target_frame]

        ax[0].hist(df_[pc_column_names[0]], bins=50, alpha=0.5, label=z_slice)
        ax[1].hist(df_[pc_column_names[1]], bins=50, alpha=0.5, label=z_slice)
        ax[2].hist(df_[pc_column_names[2]], bins=50, alpha=0.5, label=z_slice)

        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("Count")

        ax[1].set_xlabel("PC2")
        ax[1].set_ylabel("Count")

        ax[2].set_xlabel("PC3")
        ax[2].set_ylabel("Count")

        ax[2].legend(loc=(1.05, 0.75))
        fig.suptitle(f"Frame {target_frame}")

        if position is not None:
            fig.suptitle(f"Frame {target_frame}, Position {position}")

        plt.show()

    # return fig, ax
