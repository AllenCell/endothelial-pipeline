import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_4")


dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]


panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "",
        x_position=0,
        y_position=0,
        x_offset=0,  # 0.2,
        y_offset=0,  # 0.08,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "",
        x_position=0,
        y_position=2.5,
        x_offset=0,  # 0,
        y_offset=0,  # 0.2,
    ),
]

build_figure_from_panels(panels, save_dir / "figure_4.svg", width=MAX_FIGURE_WIDTH, height=4.75)
