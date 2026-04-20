"""Generate an intro schematic figure for the paper."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_MEDIUM
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

# Colors matching the schematic design
TEAL = "#008B8B"
BLACK = "#000000"


def create_intro_schematic(
    figure_size: tuple,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create the intro schematic showing the conceptual framework:
    timelapse imaging -> ML featurization -> (r, theta, rho) at t and t+1.

    Parameters
    ----------
    figure_size : tuple
        Size of the figure (width, height) in inches.
    """
    fig, ax = plt.subplots(figsize=figure_size)
    w, h = figure_size
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Square box side length (proportional to height so boxes stay visible)
    box = 0.35 * h

    # ── 1. Timelapse imaging data (left side) ────────────────────────
    ax.text(
        0.00 * w,
        h,
        "Timelapse\nimaging data",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=TEAL,
        ha="left",
        va="top",
    )

    # Frame at t (back square)
    rect_t_x, rect_t_y = 0.05 * w, 0.4 * h
    rect_t = patches.FancyBboxPatch(
        (rect_t_x, rect_t_y),
        box,
        box,
        boxstyle="square,pad=0",
        linewidth=1.0,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(rect_t)
    ax.text(
        rect_t_x + 0.1 * box,
        rect_t_y + 0.75 * box,
        "t",
        fontsize=FONTSIZE_LARGE,
        fontstyle="italic",
        color="black",
    )

    # Frame at t+1 (front square, offset down-right)
    rect_t1_x, rect_t1_y = 0.09 * w, 0.25 * h
    rect_t1 = patches.FancyBboxPatch(
        (rect_t1_x, rect_t1_y),
        box,
        box,
        boxstyle="square,pad=0",
        linewidth=1.0,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(rect_t1)
    ax.text(
        rect_t1_x + 0.1 * box,
        rect_t1_y + 0.1 * box,
        "t+1",
        fontsize=FONTSIZE_LARGE,
        fontstyle="italic",
        color="black",
    )

    # ── 2. Curved arrows: ML-based featurization ─────────────────────
    # Arrow from t frame to (r, theta, rho) at t
    arrow_t = FancyArrowPatch(
        (rect_t_x + box + 0.02 * w, rect_t_y + 0.8 * box),
        (0.40 * w, 0.78 * h),
        connectionstyle="arc3,rad=-0.3",
        arrowstyle="->,head_length=5,head_width=3",
        color=BLACK,
        linewidth=1.5,
    )
    ax.add_patch(arrow_t)

    ax.text(
        0.23 * w,
        0.5 * h,
        "ML-based\nfeaturization",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=TEAL,
        ha="left",
        va="bottom",
    )

    # Arrow from t+1 frame to (r, theta, rho) at t+1
    arrow_t1 = FancyArrowPatch(
        (rect_t1_x + box + 0.02 * w, rect_t1_y + 0.4 * box),
        (0.40 * w, 0.35 * h),
        connectionstyle="arc3,rad=0.2",
        arrowstyle="->,head_length=5,head_width=3",
        color=BLACK,
        linewidth=1.5,
    )
    ax.add_patch(arrow_t1)

    # ── 3. (r, theta, rho) at t (top right) ────────────────────────
    ax.text(
        0.42 * w,
        0.78 * h,
        f"(r, {Unicode.THETA}, {Unicode.RHO}) at t",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=BLACK,
        ha="left",
        va="center",
    )
    # "Classic quantitative biology / interpretability" label
    ax.text(
        0.62 * w,
        0.95 * h,
        "Classic quantitative\nbiology and interpretability",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=TEAL,
        ha="left",
        va="top",
    )

    # ── 4. Downward arrow with question ─────────────────────────────
    arrow_mid_x = 0.49 * w
    ax.annotate(
        "",
        xy=(arrow_mid_x, 0.42 * h),
        xytext=(arrow_mid_x, 0.68 * h),
        arrowprops={
            "arrowstyle": "->,head_length=0.5,head_width=0.25",
            "color": BLACK,
            "lw": 1.8,
        },
    )

    # Question text (right of arrow)
    ax.text(
        0.52 * w,
        0.55 * h,
        "Quantitatively, what are the dynamic relationships\n"
        "between observables that characterize cell state?",
        fontsize=FONTSIZE_MEDIUM,
        color=BLACK,
        ha="left",
        va="center",
    )

    # ── 5. (r, theta, rho) at t+1 (bottom right) ────────────────────
    ax.text(
        0.42 * w,
        0.35 * h,
        f"(r, {Unicode.THETA}, {Unicode.RHO}) at t+1",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=BLACK,
        ha="left",
        va="center",
    )

    # "Conceptual/mathematical framework" label
    ax.text(
        0.6 * w,
        0.22 * h,
        "Conceptual/mathematical\nframework for characterizing\ndynamic cell state",
        fontsize=FONTSIZE_LARGE,
        fontweight="bold",
        color=TEAL,
        ha="left",
        va="center",
    )

    # ── 6. Bottom text: second question ─────────────────────────────
    ax.text(
        0.10 * w,
        0.08 * h,
        "How do these relationships change across\n"
        "different cell states (i.e. with shear stress)?",
        fontsize=FONTSIZE_MEDIUM,
        color=BLACK,
        ha="left",
        va="center",
    )

    return fig, ax
