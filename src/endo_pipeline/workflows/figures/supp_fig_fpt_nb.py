# %% import libraries and set preliminary variables
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.track_integration import get_line_fit_and_filtered_df
from endo_pipeline.library.visualize.figure_fpt import generate_first_passage_time_example
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.integration.track_integration_viz import (
    plot_first_passage_time_correlations,
    plot_first_passage_time_parameter_sweep,
)
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_first_passage_time_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import FPT_FIG_EXAMPLES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FIRST_PASSAGE_TIME_MANIFEST_NAME

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_fpt")

low_flow_dataset = FPT_FIG_EXAMPLES["low_flow"]
high_flow_dataset = FPT_FIG_EXAMPLES["high_flow"]

fig_width = 6.1
fig_height = 6.85


# %% Generate example of a tracked and grid-crop trajectory starting from the same bin
# in feature space traveling to the fixed point
trajectory_example_filepath = generate_first_passage_time_example(
    dataset_name=low_flow_dataset.dataset_name,
    example_fixed_point_index=low_flow_dataset.fixed_point_index,
    example_tracked_crop_index=low_flow_dataset.tracked_crop_index,
    example_grid_crop_index=low_flow_dataset.grid_crop_index,
    out_dir=save_dir,
)

# %% Load the first passage time statistics dataframe to make correlation plots
# from and fit lines to the points in the correlation plots
fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_MANIFEST_NAME)
metric_to_plot = "mean"
line_fit_df, fpt_stats_df_no_nan = get_line_fit_and_filtered_df(
    first_passage_time_manifest=fpt_manifest, metric_to_fit=metric_to_plot
)

# %% make correlation plots for low and high flow examples
correlation_plot_filepaths: dict = {}
for example in FPT_FIG_EXAMPLES:
    dataset_name = FPT_FIG_EXAMPLES[example].dataset_name
    fp_idx = FPT_FIG_EXAMPLES[example].fixed_point_index

    # this check should be done in case the fixed point index is not an integer
    # because if it is a float then it will cause an issue when trying to save
    # a plot with `save_plot_to_path` (because the decimal point shows up in the filename)
    if not isinstance(fp_idx, int):
        raise ValueError(
            f"Expected fixed point index to be an integer, but got {fp_idx} for example {example}"
        )

    # extract the line fit results for this dataset and fixed point
    line_fit_result = line_fit_df[
        (line_fit_df[Column.DATASET] == dataset_name)
        & (line_fit_df[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
    ]
    df = fpt_stats_df_no_nan[
        (fpt_stats_df_no_nan[Column.DATASET] == dataset_name)
        & (fpt_stats_df_no_nan[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
    ]
    fp_stability = df[Column.VectorField.STABILITY].unique().item()
    filename = plot_first_passage_time_correlations(
        dataset_name=dataset_name,
        first_passage_time_stats_df=df,
        line_fit_df=line_fit_result,
        fixed_point_id=fp_idx,
        fixed_point_stability=fp_stability,
        out_dir=save_dir,
        metric_to_plot=metric_to_plot,
    )
    correlation_plot_filepaths[example] = filename

# %% Load the first passage time statistics dataframe to make correlation plots
# from and get the fitted lines
fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_MANIFEST_NAME)
metric_to_plot = "mean"
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
    figsize=(2.12, 2.12),
)

# --- Histogram of first passage time correlation ---
dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]
first_passage_summary_df = build_dataframe_for_first_passage_time_dataset_summary(
    dataset_names=dataset_summary_list, first_passage_time_manifest=fpt_manifest
)
fpt_pearson_r_path = plot_cross_dataset_summaries(
    first_passage_summary_df,
    output_dir=save_dir,
    column_names=[Column.VectorField.PEARSON_R],
    axis_mode="dataset",
    figure_size=(4.2, 2.3),
    set_y_lims=True,
    category_order=dataset_summary_list,
)
fpt_slope_path = plot_cross_dataset_summaries(
    first_passage_summary_df,
    output_dir=save_dir,
    column_names=[Column.VectorField.LINEFIT_SLOPE],
    axis_mode="dataset",
    figure_size=(4.2, 2.3),
    set_y_lims=True,
    category_order=dataset_summary_list,
)

# %% Build figure panels and figure
panels = [
    FigurePanel(
        letter="A",
        path=trajectory_example_filepath,
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="B",
        path=correlation_plot_filepaths["low_flow"],
        x_position=2.1,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="C",
        path=correlation_plot_filepaths["high_flow"],
        x_position=4.1,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="D",
        path=fp_param_sweep_fpt,
        x_position=0,
        y_position=2.1,
        x_offset=-0.1,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="E",
        path=fp_param_sweep_num_traj,
        x_position=0,
        y_position=4.4,
        x_offset=-0.1,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="F",
        path=fpt_pearson_r_path,
        x_position=1.9,
        y_position=2.1,
        x_offset=0,
        y_offset=0.15,
    ),
    FigurePanel(
        letter="G",
        path=fpt_slope_path,
        x_position=1.9,
        y_position=4.4,
        x_offset=0,
        y_offset=0.15,
    ),
]

build_figure_from_panels(
    figure_panels=panels,
    output_path=save_dir / "supp_fig_fpt.svg",
    width=min(fig_width, MAX_FIGURE_WIDTH),
    height=min(fig_height, MAX_FIGURE_HEIGHT),
)

# %%
