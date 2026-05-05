# %% import libraries and set preliminary variables
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.io.input import load_dataframe
from endo_pipeline.library.analyze.track_integration import get_line_fit_and_filtered_df
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.integration.track_integration_viz import (
    plot_first_passage_time_correlation_summary,
    plot_first_passage_time_parameter_sweep,
)
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import FPT_FIG_EXAMPLES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.workflow_defaults import FIRST_PASSAGE_TIME_MANIFEST_NAME

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_fpt")

high_flow_dataset = FPT_FIG_EXAMPLES["high_flow"]

fig_width = 6.1
fig_height = 5.5

# %% Load the first passage time statistics dataframe to make correlation plots
# from and get the fitted lines
fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_MANIFEST_NAME)
metric_to_plot = "mean"
line_fit_df, _ = get_line_fit_and_filtered_df(
    first_passage_time_manifest=fpt_manifest, metric_to_fit=metric_to_plot
)
fpt_param_sweep_df = load_dataframe(fpt_manifest.locations["first_passage_time_parameter_sweep"])
# %% make the plots for the desired datasets
dataset_name = high_flow_dataset.dataset_name
fp_idx = high_flow_dataset.fixed_point_index

# this check should be done in case the fixed point index is not an integer
# because if it is a float then it will cause an issue when trying to save
# a plot with `save_plot_to_path` (because the decimal point shows up in the filename)
if not isinstance(fp_idx, int):
    raise ValueError(
        f"Expected fixed point index to be an integer, but got {fp_idx} for example {dataset_name}"
    )

df = fpt_param_sweep_df[
    (fpt_param_sweep_df[Column.DATASET] == dataset_name)
    & (fpt_param_sweep_df[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
]
fp_stability = df[Column.VectorField.STABILITY].unique().item()

fp_param_sweep_fpt, fp_param_sweep_num_traj = plot_first_passage_time_parameter_sweep(
    dataset_name=dataset_name,
    fixed_point_index=fp_idx,
    fixed_point_stability=fp_stability,
    first_passage_time_param_sweep_df=df,
    fixed_point_radius_threshold=fpt_manifest.parameters["fixed_point_radius_threshold"],
    out_dir=save_dir,
    metric_to_plot=metric_to_plot,
)

filename_summary = f"FPT_correlation_summary_{metric_to_plot}"
plot_first_passage_time_correlation_summary(line_fit_df, save_dir, filename_summary)

# %% Build figure panels and figure
panels = [
    FigurePanel(
        letter="A",
        path=fp_param_sweep_fpt,
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=fp_param_sweep_num_traj,
        x_position=3,
        y_position=0,
        x_offset=0.1,
        y_offset=0,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / f"{filename_summary}.svg",
        x_position=0,
        y_position=3,
        x_offset=0.1,
        y_offset=0,
    ),
]

build_figure_from_panels(
    figure_panels=panels,
    output_path=save_dir / "supp_fig_fpt.svg",
    width=min(fig_width, MAX_FIGURE_WIDTH),
    height=min(fig_height, MAX_FIGURE_HEIGHT),
)

# %%
