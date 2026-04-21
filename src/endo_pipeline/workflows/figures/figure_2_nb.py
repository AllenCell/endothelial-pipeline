# %%
"""
Main function to create figure panels for Figure 2.
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.cli import NUM_GPUS
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import (
    get_output_path,
    join_sorted_strings,
    load_dataframe,
    load_model,
    save_plot_to_path,
)
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_reshaped_vector_field_and_grid,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
    plot_contour_colorbar,
    plot_drift_1d,
    plot_drift_contours,
    plot_drift_quiver,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import (
    get_dataset_color,
    get_label_for_column,
)
from endo_pipeline.library.visualize.figure_2 import make_crop_example_contact_sheet
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.migration_coherence import plot_optical_flow_histogram
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import METADATA_COLUMNS_TO_KEEP, POLAR_ANGLE_RANGE
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_CBAR_NUM_TICKS,
    DRIFT_CONTOUR_CBAR_ROUND,
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING,
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    STABILITY_COLOR_DICT,
    STABILITY_MARKER_DICT,
    StabilityLabel,
)
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
plt.style.use("endo_pipeline.figure")

logger = logging.getLogger(__name__)

base_output_dir = get_output_path("figure_2")

# figure is for grid based crops
crop_pattern = "grid"

dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]
dataset_summary_list = SUMMARY_PLOT_DATASETS["low_high"]

# %%

columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
columns_r_rho_str = join_sorted_strings(columns_r_rho)
column_theta = Column.DiffAEData.POLAR_ANGLE
optical_flow_feature = Column.OpticalFlow.UNIT_VECTOR_MEAN
feature_column_names = [column_theta, *columns_r_rho]
feature_columns_str = join_sorted_strings(feature_column_names)

# load dataframe manifests for diffae features, fixed points, optical flow
# features, and bootstrapped fixed points for this crop pattern, which will be
# used for all visualizations in this figure
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
fixed_points_r_rho_dataframe_manifest_name = (
    f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{columns_r_rho_str}_{base_name}"
)
fixed_points_r_rho_dataframe_manifest = load_dataframe_manifest(
    fixed_points_r_rho_dataframe_manifest_name
)
fixed_points_theta_dataframe_manifest_name = (
    f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{column_theta}_{base_name}"
)
fixed_points_theta_dataframe_manifest = load_dataframe_manifest(
    fixed_points_theta_dataframe_manifest_name
)
fixed_points_3d_dataframe_manifest_name = (
    f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{feature_columns_str}_{base_name}"
)
fixed_points_3d_dataframe_manifest = load_dataframe_manifest(
    fixed_points_3d_dataframe_manifest_name
)
bootstrap_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_dataframe_manifest_name)

# get labels for provided set of feature columns
columns_for_summary_plots = [*feature_column_names, optical_flow_feature]
column_labels_r_rho = [get_label_for_column(col).replace("polar ", "") for col in columns_r_rho]
column_label_theta = get_label_for_column(column_theta).replace("polar ", "")
dataframe_columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *feature_column_names]


# load and instantiate model for generating synthetic images
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
model = load_model(model_location, instantiate=True)

# global plotting kwargs / parameters
gridspec_kwargs = {"wspace": 0.1, "hspace": 0.1}
xlabel_kwargs = {"labelpad": 2}
ylabel_kwargs = {"labelpad": -2}

r_lims = (0.2, 1.8)
rho_lims = (-1.05, 1.05)
nullcline_r_style = "dashed"
nullcline_rho_style = (0, (1, 1))  # dense dotted
# %%

# make svg of just the colorbar with set ticks and extended on both sides
fig, ax = plot_contour_colorbar(
    figsize=(0.75, MAX_FIGURE_WIDTH / 4),
    vmin=DRIFT_CONTOUR_VMIN,
    vmax=DRIFT_CONTOUR_VMAX,
    num_ticks=DRIFT_CONTOUR_CBAR_NUM_TICKS,
    tick_label_round=DRIFT_CONTOUR_CBAR_ROUND,
    extend="both",
    colormap=DRIFT_CONTOUR_COLORMAP,
    orientation="vertical",
)
save_plot_to_path(fig, base_output_dir, "colorbar", file_format=".svg", transparent=True)

# %%
# loop over datasets in collection, compute 2D drift coefficients for each
# pairwise combination of polar coordinates, and plot contours of drift coefficients
for dataset_name, include_legend in [(dataset_low, True), (dataset_high, False)]:
    if dataset_name not in feature_dataframe_manifest.locations:
        logger.warning(
            "No location found in dataframe manifest [ %s ] for dataset [ %s ], skipping visualization.",
            feature_dataframe_manifest_name,
            dataset_name,
        )
        continue

    fig_savedir = get_output_path("figure_2", dataset_name)
    dataset_config = load_dataset_config(dataset_name)

    # load fixed points dataframes (if available) for both (r, rho) and theta,
    # filter to just stable fixed points, and store in dict for easy access when plotting
    stable_fixed_points_dict: dict[
        tuple[Column.DiffAEData] | Column.DiffAEData, pd.DataFrame | None
    ] = {}
    for column_key, manifest in [
        (columns_r_rho_str, fixed_points_r_rho_dataframe_manifest),
        (column_theta, fixed_points_theta_dataframe_manifest),
        (feature_columns_str, fixed_points_3d_dataframe_manifest),
    ]:
        df_fixed_points = load_dataframe(manifest.locations[dataset_name])
        df_stable_fixed_points = df_fixed_points[
            df_fixed_points[Column.VectorField.STABILITY] == StabilityLabel.STABLE
        ]
        stable_fixed_points_dict[column_key] = df_stable_fixed_points
    # get drift in (r, rho) space
    drift_r_rho_dataframe = load_drift_dataframe_for_dataset(dataset_name, columns=columns_r_rho)
    drift_r_rho, centers_r_rho = get_reshaped_vector_field_and_grid(
        drift_r_rho_dataframe,
        column_names=columns_r_rho,
    )
    centers_mesh = np.meshgrid(*centers_r_rho, indexing="ij")

    # get in 1D for theta
    drift_theta_dataframe = load_drift_dataframe_for_dataset(dataset_name, columns=[column_theta])
    drift_theta, centers_theta = get_reshaped_vector_field_and_grid(
        drift_theta_dataframe, column_names=[column_theta]
    )

    # make and save plots
    filename_prefix_r_rho = f"{dataset_name}_{'_'.join(columns_r_rho)}"
    filename_prefix_theta = f"{dataset_name}_{Column.DiffAEData.POLAR_ANGLE}"
    contour_plot_filename = f"{filename_prefix_r_rho}_contours"
    quiver_plot_filename = f"{filename_prefix_r_rho}_quiver"
    theta_plot_filename = f"{filename_prefix_theta}_drift"

    # plot drift contours and save
    fig, ax = plot_drift_contours(
        centers_mesh,
        drift_r_rho,
        variable_labels=column_labels_r_rho,
        figsize=(1.75, 1.9),
        axes_limits=(r_lims, rho_lims),
        axes_aspect=None,
        axes_titles=(f"d{column_labels_r_rho[0]}/dt", f"d{column_labels_r_rho[1]}/dt"),
        include_colorbar=False,
        include_nullclines=True,
        nullcline_colors=("k", "k"),
        nullcline_styles=(nullcline_r_style, nullcline_rho_style),
        nullcline_opacity=1.0,
        gridspec_kwargs=gridspec_kwargs,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        axes_title_kwargs={
            "fontsize": "small",
            "x": 1.05,
            "y": 0.5,
            "rotation": 0,
            "ha": "left",
            "va": "center",
        },
    )
    for ax_index, ax_ in enumerate(list(ax)):
        # adjust label padding and drop tick labels on shared x axis
        ax_.set_box_aspect(1.0)
        ax_.set_xticks([0.25, 1.0, 1.75])
        ax_.set_yticks([-0.75, 0.0, 0.75])
        if ax_index == 0:
            ax_.tick_params(labelbottom=False)

    shear_stress = math.ceil(max(fc.shear_stress for fc in dataset_config.flow_conditions))
    shear_stress_label = f"{shear_stress} dyn/cm{Unicode.SQUARED}"
    # reserve left margin for the vertical label
    fig.subplots_adjust(left=0.08)
    # add vertical title to the left of the contour plot spanning all rows
    fig.text(
        0.0,
        0.5,
        shear_stress_label,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=FONTSIZE_MEDIUM,
        fontweight="bold",
    )

    save_plot_to_path(
        fig,
        fig_savedir,
        contour_plot_filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        pad_inches=0,
    )

    fig, ax = plot_drift_quiver(
        centers_mesh,
        drift_r_rho,
        quiver_scale=3.5,
        quiver_color="dimgrey",
        quiver_downsample=4,
        vmin=DRIFT_CONTOUR_VMIN,
        vmax=DRIFT_CONTOUR_VMAX,
        variable_labels=column_labels_r_rho,
        figsize=(2.05, 1.65),
        axes_limits=(r_lims, rho_lims),
        include_nullclines=True,
        nullcline_colors=("k", "k"),
        nullcline_styles=(nullcline_r_style, nullcline_rho_style),
        nullcline_opacity=0.9,
        gridspec_kwargs=gridspec_kwargs,
        legend_kwargs={
            "fontsize": "xx-small",
            "title_fontsize": "xx-small",
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": 2,
            "handletextpad": 0.3,
        },
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        plot_legend=include_legend,
    )

    ax.plot(
        stable_fixed_points_dict[columns_r_rho_str][columns_r_rho[0]],
        stable_fixed_points_dict[columns_r_rho_str][columns_r_rho[1]],
        STABILITY_MARKER_DICT[StabilityLabel.STABLE],
        color=STABILITY_COLOR_DICT[StabilityLabel.STABLE],
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=5,
        label="Stable fixed point",
    )
    if include_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=2,
            handletextpad=0.3,
        )

    # make room above axes for the legend
    fig.subplots_adjust(top=0.82)

    # set plot formatting args and save
    ax.set_box_aspect(1.0)
    ax.set_xticks([0.25, 0.75, 1.25, 1.75])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    save_plot_to_path(
        fig,
        fig_savedir,
        quiver_plot_filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
    )

    # plot 1D drift in theta and save
    fig, ax = plot_drift_1d(
        drift=drift_theta,
        centers=centers_theta[-1],
        figsize=(MAX_FIGURE_WIDTH / 4, MAX_FIGURE_HEIGHT / 4),
        axes_limits=(POLAR_ANGLE_RANGE, (-0.4, 0.4)),
        axes_labels=[column_label_theta, f"d{column_label_theta}/dt"],
        gridspec_kwargs=gridspec_kwargs,
        drift_line_kwargs={"color": "k", "linewidth": 2},
        zero_line_kwargs={"linestyle": "--", "color": "gray", "linewidth": 1, "alpha": 0.7},
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
    )
    # add stable fixed point in theta
    ax.plot(
        stable_fixed_points_dict[column_theta][column_theta],
        np.zeros_like(stable_fixed_points_dict[column_theta][column_theta]),
        STABILITY_MARKER_DICT[StabilityLabel.STABLE],
        color=STABILITY_COLOR_DICT[StabilityLabel.STABLE],
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=5,
    )

    # set plot formatting args and save
    ax.set_box_aspect(1.0)
    ax.set_xticks(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
        labels=[
            f"0={Unicode.PI}",
            f"{Unicode.PI}/4",
            f"{Unicode.PI}/2",
            f"3{Unicode.PI}/4",
            f"{Unicode.PI}=0",
        ],
    )
    ax.set_yticks([-0.3, 0.0, 0.3])
    save_plot_to_path(fig, fig_savedir, theta_plot_filename, file_format=".svg")

    # make contact sheet of example crops at stable fixed points for this
    # dataset (panel below the flow field visualizations)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])
    dataframe_steady_state = filter_dataframe_to_steady_state(feature_dataframe, dataset_config)
    crop_contact_sheet_path = make_crop_example_contact_sheet(
        dataset_config=dataset_config,
        stable_fixed_point_dataframe=stable_fixed_points_dict[feature_columns_str],
        crop_features_dataframe=dataframe_steady_state,
        feature_column_names=feature_column_names,
        model=model,
        n_crop_examples=2,
        fig_savedir=fig_savedir,
        fig_filename=f"{dataset_name}_crop_examples",
        file_format=".svg",
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": (MAX_FIGURE_WIDTH / 2 - 0.2, 2), "layout": "constrained"},
        random_seed=7,
        num_gpus=NUM_GPUS,
    )

# %%
# --- Cross-dataset summary plots ---
plot_cross_dataset_summaries(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    fixed_points_bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
    output_dir=base_output_dir,
    column_names=columns_for_summary_plots,
    x_axis_mode="shear_stress_categorical",
    figure_size=(MAX_FIGURE_WIDTH - 2.1, 2),
    stable_only=True,
)

# %%
fig, ax = plt.subplots(figsize=(2, 2), layout="constrained")
for dataset_name in [dataset_low, dataset_high]:
    # get settings
    dataset_config = load_dataset_config(dataset_name)
    shear_stress = math.ceil(max(fc.shear_stress for fc in dataset_config.flow_conditions))

    # load and filter data
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df_ = df[dataframe_columns_to_compute].compute()
    df_steady_state = filter_dataframe_to_steady_state(df_, dataset_config)

    df_of = add_optical_flow_features(
        df_steady_state,
        datasets=[dataset_name],
    )
    df_flow_no_nan = df_of.dropna(subset=[optical_flow_feature])

    fig = plot_optical_flow_histogram(
        df=df_of,
        optical_flow_feature=optical_flow_feature,
        feature_label="Migration Coherence",
        feature_lim=(0, 1),
        ss_label=f"{shear_stress} dyn/cm{Unicode.SQUARED}",
        color=get_dataset_color(dataset_name),
        df_fp=None,
        binwidth=0.02,
        figure=(fig, ax),
        legend_loc=None,
    )
save_plot_to_path(
    fig,
    base_output_dir,
    "migration_coherence_distribution_high_low_flow_comparison",
    pad_inches=0,
    tight_layout=False,
    file_format=".svg",
)

# %%
# --- Assemble all panels into final figure ---
# Helper: construct per-dataset plot paths
columns_r_rho_str = "_".join(columns_r_rho)
fig_savedir_low = get_output_path("figure_2", dataset_low)
fig_savedir_high = get_output_path("figure_2", dataset_high)

panels = [
    # --- Low flow dataset (row 1) ---
    FigurePanel(
        letter="A",
        path=fig_savedir_low / f"{dataset_low}_{columns_r_rho_str}_contours.svg",
        x_position=0,
        y_position=0.0,
        x_offset=0.15,
        y_offset=-0.1,
    ),
    FigurePanel(
        letter="",
        path=base_output_dir / "colorbar.svg",
        x_position=MAX_FIGURE_WIDTH / 4 - 0.1,
        y_position=0.0,
        x_offset=0.08,
        y_offset=0.00,
    ),
    FigurePanel(
        letter="",
        path=fig_savedir_low / f"{dataset_low}_{columns_r_rho_str}_quiver.svg",
        x_position=MAX_FIGURE_WIDTH / 4 + 0.9,
        y_position=0.0,
        x_offset=-0.1,
        y_offset=0.0,
    ),
    FigurePanel(
        letter="B",
        path=fig_savedir_low / f"{dataset_low}_{Column.DiffAEData.POLAR_ANGLE}_drift.svg",
        x_position=3 * MAX_FIGURE_WIDTH / 4 - 0.35,
        y_position=0.0,
        x_offset=0.4,
        y_offset=-0.2,
    ),
    # --- High flow dataset (row 2) ---
    FigurePanel(
        letter="C",
        path=fig_savedir_high / f"{dataset_high}_{columns_r_rho_str}_contours.svg",
        x_position=0,
        y_position=1.85,
        x_offset=0.15,
        y_offset=-0.05,
    ),
    FigurePanel(
        letter="",
        path=base_output_dir / "colorbar.svg",
        x_position=MAX_FIGURE_WIDTH / 4 - 0.1,
        y_position=1.85,
        x_offset=0.08,
        y_offset=0,
    ),
    FigurePanel(
        letter="",
        path=fig_savedir_high / f"{dataset_high}_{columns_r_rho_str}_quiver.svg",
        x_position=MAX_FIGURE_WIDTH / 4 + 0.9,
        y_position=1.85,
        x_offset=-0.1,
        y_offset=-0.1,
    ),
    FigurePanel(
        letter="D",
        path=fig_savedir_high / f"{dataset_high}_{Column.DiffAEData.POLAR_ANGLE}_drift.svg",
        x_position=3 * MAX_FIGURE_WIDTH / 4 - 0.35,
        y_position=1.85,
        x_offset=0.4,
        y_offset=-0.2,
    ),
    # --- Contact sheets (row 3) ---
    FigurePanel(
        letter="E",
        path=fig_savedir_low / f"{dataset_low}_crop_examples.svg",
        x_position=0.0,
        y_position=3.8,
        x_offset=0.08,
        y_offset=0.12,
    ),
    FigurePanel(
        letter="F",
        path=fig_savedir_high / f"{dataset_high}_crop_examples.svg",
        x_position=MAX_FIGURE_WIDTH / 2,
        y_position=3.8,
        x_offset=0.08,
        y_offset=0.12,
    ),
    # --- Bottom row ---
    FigurePanel(
        letter="G",
        path=base_output_dir / "migration_coherence_distribution_high_low_flow_comparison.svg",
        x_position=0,
        y_position=6.0,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="H",
        path=base_output_dir
        / "polar_theta_polar_r_rho_ema01_optical_flow_mean_unit_vector_dt1_fp_vs_shear_stress.svg",
        x_position=2.1,
        y_position=6.0,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, base_output_dir / "figure_2.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)
# %%
