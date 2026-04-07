# %%
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

plt.style.use("endo_pipeline.figure")

DESCRIPTION = "Test figure building with different panel sizes and and offsets."

# %%
save_dir = get_output_path("figure_test")
figsize = (MAX_FIGURE_WIDTH / 2, 3)
fig, ax = plt.subplots(figsize=figsize)
ax.plot([0, 1], [0, 1])
save_plot_to_path(fig, save_dir, "test_fig", file_format=".svg")

# %%
output_path = save_dir / "test_build_fig.svg"
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "test_fig.svg",
        x_position=0,
        y_position=0,
        x_offset=0.08,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "test_fig.svg",
        x_position=3.25,
        y_position=0,
        x_offset=0.08,
        y_offset=0,
    ),
]
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=3)
# %%
