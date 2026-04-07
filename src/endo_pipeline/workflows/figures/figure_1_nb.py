# %%
import matplotlib.pyplot as plt

from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.intro_schematic import create_intro_schematic
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

DESCRIPTION = "Figure panels for Figure 1"

# %%
plt.style.use("endo_pipeline.figure")

# %%
save_dir = get_output_path("figure_1")
fig, ax = create_intro_schematic(figure_size=(MAX_FIGURE_WIDTH, 2))
save_plot_to_path(fig, save_dir, "intro_schematic", file_format=".svg", dpi=900)

# %%
