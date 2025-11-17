import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.io import save_plot_to_path


def _plot_latent_walk_batch_as_grid(
    batch_index: int,
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    save_path: str,
    file_name: str,
    use_pcs: bool = True,
) -> None:
    num_rows = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]
    fig, ax = plt.subplots(
        nrows=num_rows + 1,
        ncols=num_steps,
        figsize=(num_steps * 3, (num_rows * 3) + 2),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.02]},
    )
    for i in range(num_rows + 1):
        # last "row" is just empty for titles
        if i == num_rows:
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
            ylabel = f"PC {batch_index*i+1}" if use_pcs else f"Dim {batch_index*i}"
            ax[i, 0].set_ylabel(ylabel, fontsize=36)

        plt.tight_layout()

        save_plot_to_path(fig, save_path, file_name)


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    save_path: str,
    file_name: str,
    use_pcs: bool = True,
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
    use_pcs
        Whether the latent walk was performed along principal components.
    """

    # plot as series of batches dealing with up to 3 dims at a time
    num_dims = array_of_crops.shape[0]
    num_batches = num_dims // 3 + int(num_dims % 3 > 0)

    for batch in range(num_batches):
        start_idx = batch * 3
        end_idx = min(start_idx + 3, num_dims)
        batch_array_of_crops = array_of_crops[start_idx:end_idx, :, :, :]
        batch_coordinate_values = coordinate_values[start_idx:end_idx]

        batch_suffix = f"_{start_idx+1}_to_{end_idx}" if use_pcs else f"{start_idx}_to_{end_idx-1}"
        batch_file_name = f"{file_name}{batch_suffix}"

        _plot_latent_walk_batch_as_grid(
            batch,
            batch_array_of_crops,
            batch_coordinate_values,
            save_path,
            batch_file_name,
            use_pcs,
        )
