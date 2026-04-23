# %% import libraries and set preliminary variables
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figure_fpt import generate_first_passage_time_example
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_4")

example_dataset_name = "20250618_20X"

# %% Generate example of a tracked and grid-crop trajectory starting from the same bin
# in feature space traveling to the fixed point
generate_first_passage_time_example(dataset_name=example_dataset_name, out_dir=save_dir)

# %% Build figure panels and figure
panels = [
    FigurePanel(
        letter="A",
        path=save_dir
        / example_dataset_name
        / f"{example_dataset_name}_FPT_fp_0_mean_3d_scatter.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "20250618_20X_FPT_fp_0_stable_mean_correlation_for_figure.svg",
        x_position=0,
        y_position=2,
        x_offset=0,
        y_offset=0.2,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / "20250611_20X_FPT_fp_3_stable_mean_correlation_for_figure.svg",
        x_position=0,
        y_position=4,
        x_offset=0,
        y_offset=0.2,
    ),
    FigurePanel(
        letter="D",
        path=save_dir / "FPT_correlation_r_value_histogram_for_figure.svg",
        x_position=0,
        y_position=6,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    figure_panels=panels,
    output_path=save_dir / "figure_4.svg",
    width=MAX_FIGURE_WIDTH // 2,
    height=8,
)
