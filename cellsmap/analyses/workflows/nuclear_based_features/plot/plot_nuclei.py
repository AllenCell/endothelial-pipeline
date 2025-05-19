from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import dataset_io


def plot_flow_over_time_per_dataset(dataset_name_list: List[str], fig_savedir: str):
    fig = plt.figure(figsize=(12, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())
    linestyles = ["-", "--", "-.", ":"]

    for i, dataset in enumerate(dataset_name_list):
        duration = dataset_io.get_dataset_duration_in_frames(dataset)
        frames = range(0, duration, 1)
        flows = [dataset_io.get_flow_for_frame(dataset, frame) for frame in frames]
        plt.plot(
            frames,
            flows,
            label=dataset,
            color=colors[i % len(colors)],
            alpha=0.75,
            linewidth=2.5,
            linestyle=linestyles[i % len(linestyles)],
        )

    plt.xlabel("Time (frames)")
    plt.ylabel("Flow (dyn/cm²)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    vb.save_plot(fig, fig_savedir + f"flow_over_time.png", dpi=72)


def plot_number_of_nuclei_per_fov(df: pd.DataFrame, dataset: str, fig_savedir: str):
    fig = plt.figure(figsize=(10, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())
    added_labels = set()

    for frame, dft in df.groupby("frame"):
        x = frame
        for idx, (position, dfp) in enumerate(dft.groupby("position")):
            y = dfp["nuclear_label"].nunique()
            if position not in added_labels and len(added_labels) < 6:
                plt.scatter(
                    x, y, alpha=0.5, color=colors[idx % len(colors)], label=position
                )
                added_labels.add(position)
            else:
                plt.scatter(x, y, alpha=0.75, color=colors[idx % len(colors)])

    plt.ylim(0, 350)
    plt.xlabel("Time (frames)")
    plt.ylabel("Number of detected nuclei")
    plt.title(f"{dataset}")
    plt.legend(title="Position", loc="lower left")
    plt.show()
    vb.save_plot(
        fig,
        filename=fig_savedir + f"number_of_detected_nuclei_per_frame{dataset}",
        dpi=72,
    )


def plot_number_of_nuclei_per_dataset(df_list: List[pd.DataFrame], fig_savedir: str):
    print("Starting plot_number_of_nuclei_per_dataset...")

    fig = plt.figure(figsize=(10, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())
    legend_elements = []

    for i, df in enumerate(df_list):
        dataset = df.dataset.iloc[0]
        print(f"Processing dataset {i+1}/{len(df_list)}: {dataset}")

        for frame, dft in df.groupby("frame"):
            x = frame
            for position, dfp in dft.groupby("position"):
                y = dfp["nuclear_label"].nunique()
                plt.scatter(x, y, alpha=0.4, color=colors[i])

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=dataset,
                markerfacecolor=colors[i],
                markersize=10,
            )
        )

    plt.ylim(0, 350)
    plt.xlabel("Time (frames)")
    plt.ylabel("Number of detected nuclei")
    plt.legend(handles=legend_elements, title="Dataset", loc="lower left")
    plt.show()
    print("Plotting complete. Saving figure...")
    vb.save_plot(
        fig,
        filename=fig_savedir + f"number_of_detected_nuclei_per_frame_all_datasets",
        dpi=72,
    )
