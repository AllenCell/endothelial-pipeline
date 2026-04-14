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
columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
column_theta = Column.DiffAEData.POLAR_ANGLE
feature_column_names = [column_theta, *columns_r_rho]
column_labels_r_rho = [get_label_for_column(col).replace("polar ", "") for col in columns_r_rho]
column_label_theta = get_label_for_column(column_theta).replace("polar ", "")
columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *feature_column_names]

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
kernels_r_rho: list[KramersMoyalKernel] = []
bin_widths_r_rho: list[float] = []

# Get the corresponding kernels and bin widths for each variable. For the
# polar angle variable, also specify the period for the kernel based on the
# rescaled theta range, to ensure that the periodicity of the polar angle is
# taken into account in the flow field estimation.
for column_name in columns_r_rho:
    name = KERNEL_NAMES_DYNAMICS[column_name]
    bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
    bin_width = BIN_WIDTHS_DYNAMICS[column_name]
    kernels_r_rho.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=None))
    bin_widths_r_rho.append(bin_width)

# get dataframe manifest for crop-based features
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

# Use provided datasets or default if none provided.
low_shear_stress_repr_example = "20250409_20X"
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
for dataset_name, panel_letters, y_position, quiver_ylabel_pad in [
    (low_shear_stress_repr_example, ("A", "B"), 0.0, 2),
    (high_shear_stress_repr_example, ("C", "D"), 2.05, -3),
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

    # get drift in (r, rho) space
    bins_r_rho, centers_r_rho = get_bins(
        bin_widths=bin_widths_r_rho,
        data=df_steady_state[columns_r_rho].to_numpy(),
        lower_percentile=BIN_LIMIT_PERCENTILE_CUTOFF,
        upper_percentile=100 - BIN_LIMIT_PERCENTILE_CUTOFF,
    )

    # get 2D trajectories and differences for the pair of variables (r, rho)
    traj_r_rho, diff_r_rho = get_traj_and_diff(df_steady_state, column_names=columns_r_rho)

    drift_r_rho = get_kramers_moyal_coeffs(
        traj_r_rho,
        diff_r_rho,
        bins=bins_r_rho,
        dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
        kernel=kernels_r_rho,
    )[0]

    # get 2D meshgrid of bin centers for plotting
    centers_mesh = np.meshgrid(*centers_r_rho, indexing="ij")

    # get histogram for masking low-confidence regions of drift
    # estimates, using same kernels as for drift estimation, and set
    # drift to nan in low-confidence regions
    hist_kde_r_rho = get_kernel_density_estimate_from_trajectories(
        traj_r_rho,
        bins=bins_r_rho,
        kernel=kernels_r_rho,
    )
    low_confidence_mask = hist_kde_r_rho < HISTOGRAM_THRESHOLD_FOR_MASKING
    drift_r_rho[low_confidence_mask] = np.nan
    axes_limits_r_rho = [
        (bins_r_rho[0][0], bins_r_rho[0][-1]),
        (bins_r_rho[1][0], bins_r_rho[1][-1]),
    ]

    # get in 1D for theta
    bins_theta, centers_theta = get_bins(
        bin_widths=(BIN_WIDTHS_DYNAMICS[column_theta],),
        data=df_steady_state[column_theta].to_numpy(),
    )

    # get trajectories and differences for the given variable, adjusting
    # polar angle differences for periodicity if needed
    traj_theta, diff_theta = get_traj_and_diff(
        df_steady_state, column_names=[column_theta], polar_angle_period=polar_angle_period
    )

    kernel_theta = KramersMoyalKernel(
        name=KERNEL_NAMES_DYNAMICS[column_theta],
        bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_theta],
        period=polar_angle_period,
    )

    drift_theta = get_kramers_moyal_coeffs(
        trajectories=traj_theta,
        displacements=diff_theta,
        bins=bins_theta,
        dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
        kernel=kernel_theta,
    )[0]

    # make and save plots
    filename_prefix_r_rho = f"{dataset_name}_{'_'.join(columns_r_rho)}"
    filename_prefix_theta = f"{dataset_name}_{Column.DiffAEData.POLAR_ANGLE}"

    contour_plot_filename = f"{filename_prefix_r_rho}_contours"
    contour_plot_figsize = (MAX_FIGURE_WIDTH / 4, MAX_FIGURE_HEIGHT / 4)

    # plot drift contours and save
    fig, ax = plot_drift_contours(
        centers_mesh,
        drift_r_rho,
        variable_labels=column_labels_r_rho,
        figsize=contour_plot_figsize,
        axes_limits=axes_limits_r_rho,
        axes_aspect="equal",
        include_colorbar=False,
        gridspec_kwargs=gridspec_kwargs,
        fig_kwargs=fig_kwargs,
    )
    for ax_index, ax_ in enumerate(ax):
        # adjust label padding and drop tick labels on shared x axis
        ax_.xaxis.labelpad = 2
        ax_.yaxis.labelpad = 3
        if ax_index == 0:
            ax_.tick_params(labelbottom=False)

    save_plot_to_path(
        fig, fig_savedir, contour_plot_filename, file_format=".svg", tight_layout=False
    )

    # plot quiver plot of drift and save
    quiver_plot_filename = f"{filename_prefix_r_rho}_quiver"
    quiver_plot_figsize = (2.05, 1.65)

    fig, ax = plot_drift_quiver(
        centers_mesh,
        drift_r_rho,
        variable_labels=column_labels_r_rho,
        figsize=quiver_plot_figsize,
        axes_limits=axes_limits_r_rho,
        include_nullclines=True,
        gridspec_kwargs=gridspec_kwargs,
        legend_kwargs=legend_kwargs,
    )
    ax.set_box_aspect(1.0)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = quiver_ylabel_pad
    save_plot_to_path(
        fig,
        fig_savedir,
        quiver_plot_filename,
        file_format=".svg",
        tight_layout=False,
    )

    # plot 1D drift in theta and save
    theta_plot_filename = f"{filename_prefix_theta}_drift"
    theta_plot_figsize = (MAX_FIGURE_WIDTH / 4, MAX_FIGURE_HEIGHT / 4)

    fig, ax = plt.subplots(figsize=theta_plot_figsize, gridspec_kw=gridspec_kwargs)
    ax.plot(centers_theta[-1], drift_theta, "k-", linewidth=2)
    ax.plot(
        centers_theta[-1],
        np.zeros_like(centers_theta[-1]),
        "--",
        color="gray",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xlim(BIN_LIMITS_THETA_RESCALED)
    ax.set_box_aspect(1.0)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = quiver_ylabel_pad
    ax.set_xlabel(column_label_theta)
    ax.set_ylabel(f"d{column_label_theta}/dt")
    save_plot_to_path(fig, fig_savedir, theta_plot_filename, file_format=".svg")

    # build panels for this dataset's visualizations, adjusting positions based
    # on dataset to stack vertically in figure

    contour_plots = FigurePanel(
        letter=panel_letters[0],
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
        x_offset=-0.1,
        y_offset=0.0,
    )

    theta_plot = FigurePanel(
        letter=panel_letters[1],
        path=fig_savedir / f"{theta_plot_filename}.svg",
        x_position=3 * MAX_FIGURE_WIDTH / 4 - 0.45,
        y_position=y_position,
        x_offset=0.4,
        y_offset=-0.2,
    )
    panels.extend([contour_plots, colorbar_panel, quiver_plot, theta_plot])

# build figure from panels and save
figure_filename = "figure_2_dynamics"
build_figure_from_panels(
    panels,
    base_output_dir / f"{figure_filename}.svg",
    width=MAX_FIGURE_WIDTH,
    height=MAX_FIGURE_WIDTH,
)
# %%
