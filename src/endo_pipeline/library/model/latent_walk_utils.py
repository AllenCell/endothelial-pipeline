import cv2
import numpy as np


def write_text(img: np.ndarray, text: str) -> np.ndarray:
    """Write text on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = tuple([img.max()] * 3)
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - 3  # 3 pixels from the right edge
    text_y = text_size[1] + 3  # 3 pixels from the top edge
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img


def write_pc_vals(walk_img: np.ndarray, ranges: list) -> np.ndarray:
    """Write dimension index and value on image."""
    idx = 0
    for i, range_ in enumerate(ranges):
        for val in range_:
            walk_img[idx] = write_text(walk_img[idx], f"{i+1}:{val:.1f}")
            idx += 1
    return walk_img


def get_walk(
    data: np.ndarray, list_of_axes: list[int], sigma: float, n_steps: int
) -> tuple[list, list]:
    """
    Generate a latent walk based on standard deviation
    or min/max of each dimension.

    Parameters
    ----------
    data: np.ndarray
        Numpy array containing the data to be traversed.
    list_of_axes: list[int]
        List of dimensions to traverse.
    sigma: float
        Range of values for the latent walk.
    n_steps: int
        Number of steps in the latent walk.
    """
    walk = []
    ranges = []
    for dim in list_of_axes:
        if sigma is None:
            data_min = data[:, dim].min()
            data_max = data[:, dim].max()
            range_ = np.linspace(data_min, data_max, n_steps)
        else:
            std = data[:, dim].std()
            range_ = np.arange(-sigma, sigma + 0.01) * std
        dim_traversal = np.stack([data.mean(axis=0)] * range_.shape[0])
        dim_traversal[:, dim] = range_
        walk.append(dim_traversal)
        ranges.append(range_)
    walk = np.concatenate(walk).squeeze()
    return walk, ranges
