import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    list_of_axes: list[int],
    use_pcs: bool,
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
    list_of_axes
        List of dimension indices corresponding to the rows in the grid.
    use_pcs
        Boolean indicating whether walking along principal components (PCs) or raw latent dimensions.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.
    """
    num_dims = len(list_of_axes)
    if array_of_crops.shape[0] != num_dims:
        raise ValueError(
            "The first dimension of array_of_crops must match the length of list_of_axes."
        )
    num_steps = array_of_crops.shape[1]

    fig, ax = plt.subplots(
        nrows=num_dims + 1,
        ncols=num_steps,
        figsize=(num_steps * 3, (num_dims * 3) + 2),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.02]},
    )

    label_prefix = "PC" if use_pcs else "Dim"

    for i in range(num_dims + 1):
        # last "row" is just empty for titles
        if i == num_dims:
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
            # add dim index (1-indexing) as y-axis label on left side only
            ax[i, 0].set_ylabel(f"{label_prefix} {list_of_axes[i]+1}", fontsize=36)

    plt.tight_layout()

    save_plot_to_path(fig, save_path, file_name)
