# %% import libraries and set preliminary variables
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_fpt")
"FPT_correlation_summary_for_figure.svg"

fig_width = 6.1
fig_height = 5.5

# %% Build figure panels and figure
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "FPT_mean_vs_threshold_fp_0_stable.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "FPT_percent_trajectories_vs_threshold_fp_0_stable.svg",
        x_position=3,
        y_position=0,
        x_offset=0.1,
        y_offset=0,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / "FPT_correlation_summary_for_figure.svg",
        x_position=0,
        y_position=3,
        x_offset=0.1,
        y_offset=0,
    ),
]

build_figure_from_panels(
    figure_panels=panels,
    output_path=save_dir / "supp_fig_fpt.svg",
    width=min(fig_width, MAX_FIGURE_WIDTH),
    height=min(fig_height, MAX_FIGURE_HEIGHT),
)
