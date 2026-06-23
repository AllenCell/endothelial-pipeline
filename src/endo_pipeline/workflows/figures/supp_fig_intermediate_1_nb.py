# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.spatial_feature_grid import create_panel_spatial_feature_grid
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_intermediate")

# %%
fig = create_panel_spatial_feature_grid(
    feature_columns=[ColumnName.DiffAEData.PC3_FLIPPED],
    example_images=FIGURE_3_EXAMPLE_IMAGES,
    include_bf_images=True,
    crop_size=256,
    start_x_col=ColumnName.DiffAEData.START_X,
    start_y_col=ColumnName.DiffAEData.START_Y,
    grid_start_xy=(128, 128),
    grid_dimensions=(3, 3),
    save_dir=save_dir,
)
save_plot_to_path(
    fig,
    save_dir,
    "spatial_feature_grid_examples_supp",
    file_format=".svg",
    tight_layout=False,
    pad_inches=0,
)
# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "spatial_feature_grid_examples_supp.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, save_dir / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)
# %%
