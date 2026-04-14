# %%
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    get_kernel_density_estimate_from_trajectories,
    get_kramers_moyal_coeffs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
    plot_drift_contours,
    plot_drift_quiver,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMIT_PERCENTILE_CUTOFF,
    BIN_LIMITS_DYNAMICS,
    BIN_LIMITS_THETA_RESCALED,
    BIN_WIDTHS_DYNAMICS,
    HISTOGRAM_THRESHOLD_FOR_MASKING,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
    RESCALE_THETA,
)
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")

DESCRIPTION = "Build panels with visualizations of dynamics in r v. rho and separately in theta."

# %%

logger = logging.getLogger(__name__)

# figure is for grid based crops
crop_pattern = "grid"

# get labels for provided set of feature columns
column_names = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
variable_labels = [get_label_for_column(col).replace("polar ", "") for col in column_names]
columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

# unpack default bin widths and limits for each column, adjusting limits if rescaling theta
global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
if RESCALE_THETA:
    global_bin_limits_dict[Column.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
polar_angle_period = (
    global_bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][1]
    - global_bin_limits_dict[Column.DiffAEData.POLAR_ANGLE][0]
)


# initialize kernels and bin widths for each of the three variables for flow
# field estimation
kernels: list[KramersMoyalKernel] = []
bin_widths: list[float] = []

# Get the corresponding kernels and bin widths for each variable. For the
# polar angle variable, also specify the period for the kernel based on the
# rescaled theta range, to ensure that the periodicity of the polar angle is
# taken into account in the flow field estimation.
for column_name in column_names:
    name = KERNEL_NAMES_DYNAMICS[column_name]
    bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
    bin_width = BIN_WIDTHS_DYNAMICS[column_name]
    kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=None))
    bin_widths.append(bin_width)

# get dataframe manifest for crop-based features
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

# Use provided datasets or default if none provided.
low_shear_stress_repr_example = "20250402_20X"
high_shear_stress_repr_example = "20251001_20X"

# global plotting kwargs
gridspec_kwargs = {"wspace": 0.1, "hspace": 0.1}
fig_kwargs = {"constrained_layout": True}
legend_kwargs = {"fontsize": "xx-small", "title_fontsize": "xx-small", "loc": (1.05, 0.7)}

# %%
base_output_dir = get_output_path(__file__)
# make svg of just the colorbar with set ticks and extended on both sides
fig, ax = plt.subplots(figsize=(0.85, MAX_FIGURE_WIDTH / 4))
# center colormap at zero to visualize sign and magnitude of drift
vmin = DRIFT_CONTOUR_VMIN
vmax = DRIFT_CONTOUR_VMAX
colormap_norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
colorbar_ticks = np.linspace(vmin, vmax, DRIFT_CONTOUR_CBAR_NUM_TICKS)
colorbar_ticks = np.round(colorbar_ticks, DRIFT_CONTOUR_CBAR_ROUND)
cb = fig.colorbar(
    ScalarMappable(norm=colormap_norm, cmap=DRIFT_CONTOUR_COLORMAP),
    cax=ax,
    orientation="vertical",
    ticks=colorbar_ticks,
    extend="both",
)
save_plot_to_path(fig, base_output_dir, "colorbar", file_format=".svg")

# %%
# loop over datasets in collection, compute 2D drift coefficients for each
# pairwise combination of polar coordinates, and plot contours of drift coefficients
panels = []
for dataset_name, panel_letter, y_position in [
    (low_shear_stress_repr_example, "A", 0.0),
    (high_shear_stress_repr_example, "C", 2.05),
]:
    if dataset_name not in feature_dataframe_manifest.locations:
        logger.warning(
            "No location found in dataframe manifest [ %s ] for dataset [ %s ], skipping visualization.",
            feature_dataframe_manifest_name,
            dataset_name,
        )
        continue

    fig_savedir = get_output_path(__file__, dataset_name)
    dataset_config = load_dataset_config(dataset_name)

    # load dataframe and perform additional filtering (remove
    # non-steady-state timepoints based on annotations), computing
    # only the columns needed for flow field estimation and analysis to save memory.
    df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df = df_[columns_to_compute].compute()
    df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    # build kernels for each variable in the pair based on settings,
    # adjusting for periodicity if needed, and get bin edges and
    # centers for each variable in the pair. also get variable labels
    # and axis limits for plotting, adjusting limits if rescaling
    # theta and if not using global limits
    bins, centers = get_bins(
        bin_widths=bin_widths,
        data=df_steady_state[column_names].to_numpy(),
        lower_percentile=BIN_LIMIT_PERCENTILE_CUTOFF,
        upper_percentile=100 - BIN_LIMIT_PERCENTILE_CUTOFF,
    )

    # get 2D trajectories and differences for the pair of variables
    traj_2d, diff_2d = get_traj_and_diff(df_steady_state, column_names=column_names)

    drift, _ = get_kramers_moyal_coeffs(
        traj_2d,
        diff_2d,
        bins=bins,
        dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
        kernel=kernels,
    )

    # get 2D meshgrid of bin centers for plotting
    centers_mesh = np.meshgrid(*centers, indexing="ij")

    # get histogram for masking low-confidence regions of drift
    # estimates, using same kernels as for drift estimation, and set
    # drift to nan in low-confidence regions
    hist_kde = get_kernel_density_estimate_from_trajectories(
        traj_2d,
        bins=bins,
        kernel=kernels,
    )
    low_confidence_mask = hist_kde < HISTOGRAM_THRESHOLD_FOR_MASKING
    drift[low_confidence_mask] = np.nan

    filename_prefix = f"{dataset_name}_{'_'.join(column_names)}"
    # plot drift contours and save
    axes_limits = [(bins[0][0], bins[0][-1]), (bins[1][0], bins[1][-1])]

    filename_prefix = f"{dataset_name}_{'_'.join(column_names)}"
    contour_plot_filename = f"{filename_prefix}_contours"
    contour_plot_figsize = (MAX_FIGURE_WIDTH / 4, MAX_FIGURE_HEIGHT / 4)
    # plot drift contours and save
    fig, _ = plot_drift_contours(
        centers_mesh,
        drift,
        variable_labels=variable_labels,
        figsize=contour_plot_figsize,
        axes_limits=axes_limits,
        axes_aspect="equal",
        include_colorbar=False,
        gridspec_kwargs=gridspec_kwargs,
        fig_kwargs=fig_kwargs,
    )
    save_plot_to_path(
        fig, fig_savedir, contour_plot_filename, file_format=".svg", tight_layout=False
    )

    # plot quiver plot of drift and save
    quiver_plot_filename = f"{filename_prefix}_quiver"
    quiver_plot_figsize = (2.25, 2.0)
    fig, ax = plot_drift_quiver(
        centers_mesh,
        drift,
        variable_labels=variable_labels,
        figsize=quiver_plot_figsize,
        axes_limits=axes_limits,
        include_nullclines=True,
        gridspec_kwargs=gridspec_kwargs,
        fig_kwargs=fig_kwargs,
        legend_kwargs=legend_kwargs,
    )
    ax.set_aspect("equal")
    save_plot_to_path(
        fig,
        fig_savedir,
        quiver_plot_filename,
        file_format=".svg",
        tight_layout=False,
    )

    contour_plots = FigurePanel(
        letter=panel_letter,
        path=fig_savedir / f"{contour_plot_filename}.svg",
        x_position=0,
        y_position=y_position,
        x_offset=0.08,
        y_offset=0,
    )

    colorbar_panel = FigurePanel(
        letter="",
        path=base_output_dir / "colorbar.svg",
        x_position=MAX_FIGURE_WIDTH / 4 - 0.5,
        y_position=y_position,
        x_offset=0.08,
        y_offset=0,
    )

    quiver_plot = FigurePanel(
        letter="",
        path=fig_savedir / f"{quiver_plot_filename}.svg",
        x_position=MAX_FIGURE_WIDTH / 4 + 0.5,
        y_position=y_position,
        x_offset=0.0,
        y_offset=-0.15,
    )
    panels.extend([contour_plots, colorbar_panel, quiver_plot])

# %%
# build figure from panels and save
figure_filename = "figure_2_dynamics"
build_figure_from_panels(
    panels,
    base_output_dir / f"{figure_filename}.svg",
    width=MAX_FIGURE_WIDTH,
    height=MAX_FIGURE_WIDTH,
)
# %%
