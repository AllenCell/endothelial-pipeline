# %%
from pathlib import Path

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.spatial_feature_grid import create_panel_spatial_feature_grid
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.workflows.figures import assets as figure_assets

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_3")

assets_dir = Path(figure_assets.__path__[0])
schematic_fp = assets_dir / "figure_3a_hypotheses.svg"

# %%
feature_columns = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
]
fig = create_panel_spatial_feature_grid(
    feature_columns=feature_columns,
    example_images=FIGURE_3_EXAMPLE_IMAGES,
    figure_size=(MAX_FIGURE_WIDTH, 4.4),
)
save_plot_to_path(
    fig,
    save_dir,
    "spatial_feature_grid_examples_main",
    file_format=".svg",
    tight_layout=False,
    pad_inches=0,
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=schematic_fp,
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "spatial_feature_grid_examples_main.svg",
        x_position=0,
        y_position=2.6,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, save_dir / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)

# %%
