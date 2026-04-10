"""Module for plotting phase portraits of 2D systems of ODEs."""

import logging

from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_MARKER_DICT,
    StabilityLabel,
    StabilityLegendHandle,
)

logger = logging.getLogger(__name__)


def make_legend_handles_for_fixed_pts(
    fpt_stabilities: list[str],
    face_color_dict: dict[str, str] = STABILITY_COLOR_DICT,
    marker_dict: dict[str, str] = STABILITY_MARKER_DICT,
    marker_size: int = 10,
    edge_color: str = "black",
) -> list[StabilityLegendHandle]:
    """Make a custom legend for the fixed point types, nullclines and trajectories.

    Purpose of this method is to create a legend that only includes the fixed
    point types that are present in the plot, since the number and type of fixed
    points can vary across parameter space. That is, we want to avoid having
    duplicate labels where we have multiple fixed points of the same type, but
    we also want to avoid having labels for types that are not present.

    Parameters
    ----------
    fpt_stabilities
        List of stability labels for the fixed points.
    face_color_dict
        Dictionary mapping stability labels to face colors.
    marker_dict
        Dictionary mapping stability labels to marker styles.
    marker_size
        Size of the markers for the legend handles.
    edge_color
        Color of the marker edges.

    Returns
    -------
    :
        List of StabilityLegendHandle objects representing the legend handles
        for the fixed point types.

    """
    my_handles = []
    # get legend handles for the fixed point types that are present in given
    # list of fixed point stabilities, in the order given by StabilityLabel enum
    for stability_type in StabilityLabel:
        if stability_type in fpt_stabilities:
            my_handles.append(
                StabilityLegendHandle(
                    stability_label=stability_type,
                    legend_label=stability_type,
                    marker=marker_dict[stability_type],
                    face_color=face_color_dict[stability_type],
                    edge_color=edge_color,
                    marker_size=marker_size,
                )
            )

    return my_handles
