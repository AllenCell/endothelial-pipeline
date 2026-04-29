# %% import libraries and set preliminary variables
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figure_fpt import generate_first_passage_time_example
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.examples import FPT_FIG_EXAMPLES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_4")

low_flow_dataset = FPT_FIG_EXAMPLES["low_flow"]
high_flow_dataset = FPT_FIG_EXAMPLES["high_flow"]

# %% Generate example of a tracked and grid-crop trajectory starting from the same bin
# in feature space traveling to the fixed point
generate_first_passage_time_example(
    dataset_name=low_flow_dataset.dataset_name,
    example_fixed_point_index=low_flow_dataset.fixed_point_index,
    example_tracked_crop_index=low_flow_dataset.tracked_crop_index,
    example_grid_crop_index=low_flow_dataset.grid_crop_index,
    out_dir=save_dir,
)

# %% Build figure panels and figure
panels = [
    FigurePanel(
        letter="A",
        path=save_dir
        / f"{low_flow_dataset.dataset_name}_FPT_fp_{low_flow_dataset.fixed_point_index}_mean_3d_scatter.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="B",
        path=save_dir
        / f"{low_flow_dataset.dataset_name}_FPT_fp_{low_flow_dataset.fixed_point_index}_stable_mean_correlation.svg",
        x_position=0,
        y_position=2.2,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="C",
        path=save_dir
        / f"{high_flow_dataset.dataset_name}_FPT_fp_{high_flow_dataset.fixed_point_index}_stable_mean_correlation.svg",
        x_position=0,
        y_position=4.1,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="D",
        path=save_dir / "FPT_correlation_summary_mean_histogram.svg",
        x_position=0,
        y_position=6,
        x_offset=0,
        y_offset=0.1,
    ),
]

build_figure_from_panels(
    figure_panels=panels,
    output_path=save_dir / "figure_4.svg",
    width=2,
    height=MAX_FIGURE_HEIGHT,
)

# %%
