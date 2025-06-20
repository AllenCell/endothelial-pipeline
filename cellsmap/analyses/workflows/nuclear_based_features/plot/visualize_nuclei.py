import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from matplotlib.colors import BoundaryNorm, ListedColormap
from skimage.measure import label

from cellsmap.analyses.utils.viz import viz_base as vb
from endo_pipeline.configs import dataset_io


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
