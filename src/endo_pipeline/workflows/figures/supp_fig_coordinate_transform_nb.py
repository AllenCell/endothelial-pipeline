# %%
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.supp_fig_coordinate_transform import (
    perform_latent_walk_along_top_pcs,
)

# %%
plt.style.use("endo_pipeline.figure")
output_path = get_output_path("supp_fig_coords")

# load model manifest, get run name, and load model
latent_walk_filename = "latent_walk_top_3_pcs"

walk_img_grid = perform_latent_walk_along_top_pcs(output_path, latent_walk_filename)
latent_walk_path = output_path / f"{latent_walk_filename}_scale_bar_10um.svg"
