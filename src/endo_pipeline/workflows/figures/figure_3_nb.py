# %%
import matplotlib.pyplot as plt

from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.visualize.data_example_figures import create_panel_intermediate_examples
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

plt.style.use("endo_pipeline.figure")
# %%
save_dir = get_output_path("figure_3")

# Example images of intermediate shear stress condition
create_panel_intermediate_examples(
    examples=FIGURE_3_EXAMPLE_IMAGES,
    save_dir=save_dir,
    figure_size=(MAX_FIGURE_WIDTH * 0.75, 2.5),
)

# %%
