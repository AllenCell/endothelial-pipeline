import matplotlib.axes as maxes
import matplotlib.patches as patches


def add_scalebar(
    ax: maxes.Axes,
    length_px: float,
    location: str = "lower left",
    bar_thickness: float = 10,
    padding: float = 20,
    color: str = "white",
) -> None:
    """
    Add a scale bar to an image displayed with imshow (no text label).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the scale bar to.
    length_px : float
        Length of the scale bar in pixels.
    location : str, optional
        One of 'upper left', 'upper right', 'lower left', 'lower right'.
    bar_thickness : float, optional
        Thickness of the scale bar in pixels.
    padding : float, optional
        Padding from the edge of the image in pixels.
    color : str, optional
        Color of the scale bar.
    """

    ny, nx = ax.images[0].get_array().shape  # image dimensions

    # Determine bar position
    if location == "upper left":
        x_start = padding
        y_start = padding
    elif location == "upper right":
        x_start = nx - padding - length_px
        y_start = padding
    elif location == "lower left":
        x_start = padding
        y_start = ny - padding - bar_thickness
    elif location == "lower right":
        x_start = nx - padding - length_px
        y_start = ny - padding - bar_thickness
    else:
        raise ValueError(f"Invalid location: {location}")

    # Draw the scale bar
    rect = patches.Rectangle(
        (x_start, y_start),
        length_px,
        bar_thickness,
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(rect)
