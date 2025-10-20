import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.visualize import viz_base


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    save_path: str,
    file_name: str,
) -> None:
    """
    Plot a grid of reconstructed image crops representing a latent walk.

    Parameters
    ----------
    array_of_crops
        An ND numpy array of shape (num_dims, num_steps, h, w)
        containing the reconstructed image crops.
    coordinate_values
        An ND numpy array of shape (num_dims, num_steps)
        containing the coordinate values for each dimension and step.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.
    """
    num_pcs = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]

    fig, ax = plt.subplots(
        nrows=num_pcs + 1,
        ncols=num_steps,
        figsize=(num_steps * 3, (num_pcs * 3) + 2),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.02]},
    )

    for i in range(num_pcs + 1):
        # last "row" is just empty for titles
        if i == num_pcs:
            for j in range(num_steps):
                ax[i, j].axis("off")
                column_title = rf"{j - (num_steps // 2)}$\sigma$"
                ax[i, j].set_title(column_title, fontsize=32)
        else:
            for j in range(num_steps):
                ax[i, j].imshow(array_of_crops[i, j], cmap="gray")
                ax[i, j].set_xticks([])  # Turn off x-axis ticks
                ax[i, j].set_yticks([])  # Turn off y-axis ticks
                # add value label as title
                value_label = f"{np.round(coordinate_values[i][j], 2)}"
                ax[i, j].set_title(value_label, fontsize=20)
            # add PC index as y-axis label on left side only
            ax[i, 0].set_ylabel(f"PC {i+1}", fontsize=36)

    plt.tight_layout()

    save_plot_to_path(fig, save_path, file_name)
