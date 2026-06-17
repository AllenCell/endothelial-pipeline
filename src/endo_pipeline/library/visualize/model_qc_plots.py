"""Visualization functions for Model QC workflow."""

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.plot_defaults import (
    MODEL_QC_GRIDSPEC_KWARGS,
    MODEL_QC_PLOT_DIRECTION,
    MODEL_QC_SUBPLOT_KWARGS,
)

if TYPE_CHECKING:
    import matplotlib.figure


# ========================
# Contact Sheet Functions
# ========================


def create_validation_examples_contact_sheet(
    conditioning_crops: list[np.ndarray],
    diffusion_crops: list[np.ndarray],
    denoised_by_conditioning: list[np.ndarray],
    denoised_scrambled_latent: list[np.ndarray],
    denoised_scrambled_input: list[np.ndarray],
    label_for_conditioning: str,
    *,
    pixel_size: float,
    figure_width: float,
    figure_height: float = 3.4,
    scalebar_um: int = 10,
    scalebar_location: str = "lower right",
    max_rows: int = 3,
    font_size: int = 10,
) -> "matplotlib.figure.Figure":
    """Build the publication-styled DiffAE validation contact sheet.

    One row per validation example, with five columns: the conditioning
    encoder input, the target VE-cadherin, and three denoised predictions
    (true conditioning latent + two negative controls). Used as the panel-A
    contact sheet for both the VE-cadherin-conditioned model figure
    (``supp-fig-diffae-model``) and the brightfield-conditioned schematic
    figure (``supp-fig-diffae-schematic``).

    The conditioning channel differs by model (brightfield vs. VE-cadherin),
    so the column labels are parametrized via ``label_for_conditioning``; the
    diffusion target is always VE-cadherin.

    Parameters
    ----------
    conditioning_crops
        Per-example conditioning input crops (column 1).
    diffusion_crops
        Per-example ground-truth VE-cadherin crops (column 2).
    denoised_by_conditioning
        Per-example denoised outputs from the true conditioning latent (column 3).
    denoised_scrambled_latent
        Per-example denoised outputs from a scrambled latent vector (column 4).
    denoised_scrambled_input
        Per-example denoised outputs from a latent extracted from a
        pixel-scrambled conditioning image (column 5).
    label_for_conditioning
        Label for the conditioning channel (e.g. ``"Brightfield"`` or
        ``"VE-cadherin"``).
    pixel_size
        Physical pixel size for the scale bar.
    figure_width
        Figure width in inches.
    figure_height
        Figure height in inches.
    scalebar_um
        Scale-bar length in microns.
    scalebar_location
        Corner of each panel to place the scale bar in (every image gets a
        bar; only the top-left panel is annotated with the length label).
    max_rows
        Maximum number of example rows.
    font_size
        Font size for column titles and the prediction supertitle.

    Returns
    -------
        The contact sheet figure (caller is responsible for saving).
    """
    titles = [
        f"{label_for_conditioning}\nencoder input",
        "Target\nVE-cadherin",
        f"{label_for_conditioning}\nlatent vector",
        "Scrambled\nlatent vector",
        "Scrambled\ninput image",
    ]

    panels = [
        img
        for img_list in [
            conditioning_crops,
            diffusion_crops,
            denoised_by_conditioning,
            denoised_scrambled_latent,
            denoised_scrambled_input,
        ]
        for img in img_list
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=max_rows,
        max_cols=5,
        col_titles=titles,
        row_titles=None,
        direction=cast(Literal["left-right first", "top-down first"], MODEL_QC_PLOT_DIRECTION),
        font_size=font_size,
        subplot_kwargs=MODEL_QC_SUBPLOT_KWARGS,
        gridspec_kwargs=MODEL_QC_GRIDSPEC_KWARGS,
        fig_kwargs={"figsize": (figure_width, figure_height)},
    )

    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0)
    all_axes = fig.get_axes()
    col_3_pos = all_axes[3].get_position()
    center_x = col_3_pos.x0 + (col_3_pos.width / 2)
    fig.text(
        x=center_x,
        y=0.97,
        s="Predicted VE-cadherin",
        ha="center",
        fontsize=font_size,
    )
    # Scale bar on every image; only the top-left panel carries the text label
    # so the sheet isn't cluttered with the length repeated on every tile.
    for i, ax in enumerate(all_axes):
        if not ax.images:
            continue
        add_scalebar(
            ax,
            pixel_size=pixel_size,
            scale_bar_um=scalebar_um,
            bar_thickness=3,
            padding=5,
            location=scalebar_location,
            include_label=(i == 0),
        )

    return fig
