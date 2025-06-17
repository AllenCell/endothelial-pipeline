from typing import List

import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bioio import BioImage
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from skimage.measure import label

from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import dataset_io


def plot_flow_over_time_per_dataset(
    dataset_name_list: List[str], fig_savedir: str
) -> None:
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


def plot_number_of_nuclei_per_fov(
    df: pd.DataFrame, dataset: str, fig_savedir: str
) -> None:
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


def plot_number_of_nuclei_per_dataset(
    df_list: List[pd.DataFrame], fig_savedir: str
) -> None:
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


def visualize_nuclear_seg(df, dataset, frame, position, fig_savedir):
    df_img = df[(df["position"] == position) & (df["frame"] == frame)]

    fov_path = df_img["fov_path"].iloc[0]
    print(fov_path)

    image = BioImage(fov_path)
    stdev_proj = image.get_image_data("YX", C=1)
    brightfield_data = image.get_image_data("YX", C=0)
    segmentation_data = image.get_image_data("YX", C=2)
    labeled_image = label(segmentation_data)

    num_labels = labeled_image.max() + 1

    p1, p99 = np.percentile(brightfield_data, (1, 99))
    brightfield_data_norm = np.clip((brightfield_data - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(stdev_proj, (1, 99))
    stdev_proj_norm = np.clip((stdev_proj - p1) / (p99 - p1), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the std. dev brightfield image
    axes[0].imshow(stdev_proj_norm, cmap="gray")
    axes[0].set_title("Std Dev Projection")
    axes[0].axis("off")

    colors = [(0, 0, 0, 1)] + [
        cc.glasbey_light[i % len(cc.glasbey_light)] for i in range(num_labels)
    ]  # Prepend black for the background
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), cmap.N)

    # Plot the labeled image with the custom colormap
    axes[1].imshow(labeled_image, cmap=cmap, norm=norm)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    colors = [(0, 0, 0, 0)] + [
        cc.glasbey_light[i % len(cc.glasbey_light)] for i in range(num_labels)
    ]  # Prepend transparent for the background
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), cmap.N)

    # Plot the merge of the segmentation and the raw brightfield image slice
    axes[2].imshow(brightfield_data_norm, cmap="gray")
    axes[2].imshow(labeled_image, cmap=cmap, norm=norm, alpha=0.4)
    axes[2].set_title("Merged")
    axes[2].axis("off")

    flow = dataset_io.get_flow_for_frame(dataset, frame)
    plt.suptitle(
        f"Dataset: {dataset}, Frame: {frame}, Position: {position}, Flow: {flow} dyn/cm²"
    )
    plt.tight_layout()
    plt.show()
    vb.save_plot(
        fig,
        filename=fig_savedir
        + f"nuclear_segmentation_overlay_{dataset}_{position}_{frame}",
        dpi=72,
    )
