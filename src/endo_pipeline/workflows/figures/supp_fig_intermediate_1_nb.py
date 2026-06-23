# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.spatial_feature_grid import create_panel_spatial_feature_grid
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_3")

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
    filename="spatial_feature_grid_examples",
    figure_size=(MAX_FIGURE_WIDTH, 3.3),
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "spatial_feature_grid_examples.svg",
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
