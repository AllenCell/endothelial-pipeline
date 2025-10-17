import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.visualize import viz_base


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    save_path: str,
    file_name: str,
) -> None:
    """
    Plot a grid of reconstructed image crops representing a latent walk.

    Parameters
    ----------
    array_of_crops
        An ND numpy array of shape (num_pcs, num_steps, h, w)
        containing the reconstructed image crops.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.

    Returns
    -------
    :
        A matplotlib Figure object containing the grid of images,
        and a numpy array of the same images.
    :
        The corresponging array of Axes objects.
    """
    num_pcs = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]

    fig, ax = viz_base.init_subplots(num_pcs, num_steps, figsize=(num_steps * 3, num_pcs * 3))

    for i in range(num_pcs):
        for j in range(num_steps):
            ax[i, j].imshow(array_of_crops[i, j], cmap="gray")
            ax[i, j].axis("off")
            # each column is - n(sigma) to + n(sigma), where n = num_steps // 2
            if i == 0:
                column_title = rf"{j - (num_steps // 2)}$\sigma$"
                ax[i, j].set_title(column_title, fontsize=26)
        ax[i, 0].set_ylabel(f"PC {i+1}", fontsize=16)  # doesnt work with axis off, add text?

    plt.tight_layout()

    save_plot_to_path(fig, save_path, file_name)
