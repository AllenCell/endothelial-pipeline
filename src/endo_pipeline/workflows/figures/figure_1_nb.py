# %%
import matplotlib.pyplot as plt

from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.data_example_figures import (
    create_panel_b_biological_system_examples,
    create_panel_c_patch_featurization,
)
from endo_pipeline.library.visualize.intro_schematic import create_intro_schematic
from endo_pipeline.settings.examples import (
    FIGURE_1_PANEL_B_EXAMPLE_IMAGES,
    FIGURE_1_PANEL_C_EXAMPLE_IMAGE,
)
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.workflows.development.visualize_feature_correlations import (
    main as visualize_feature_correlations,
)
from endo_pipeline.workflows.production.visualize_latent_walk import main as visualize_latent_walk

DESCRIPTION = "Figure panels for Figure 1"

# %%
plt.style.use("endo_pipeline.figure")

# %% Panel A: Intro schematic
save_dir = get_output_path("figure_1")
fig, ax = create_intro_schematic(figure_size=(MAX_FIGURE_WIDTH, 2))
save_plot_to_path(fig, save_dir, "intro_schematic", file_format=".svg", dpi=900)

# %% Panel B: Example images from biological system at low and high shear stress
create_panel_b_biological_system_examples(
    examples=FIGURE_1_PANEL_B_EXAMPLE_IMAGES,
    save_dir=save_dir,
)

# %% Panel C: Patch featurization example
create_panel_c_patch_featurization(
    example=FIGURE_1_PANEL_C_EXAMPLE_IMAGE,
    save_dir=save_dir,
)

# %% Panel D: Correlation heatmaps of ai learned and measured features
visualize_feature_correlations()

# %% Panel E: Latent walk visualization
visualize_latent_walk()

# %%
