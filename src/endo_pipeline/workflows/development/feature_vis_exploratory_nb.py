# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.cli.logs import setup_logging, silence_external_loggers
from endo_pipeline.configs import load_dataset_collection_config, load_dataset_config
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    get_smallest_angle_difference,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
    plot_1d_diffusion,
    plot_1d_drift,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# set up logging
VERBOSE = True
DEBUG = False

if VERBOSE:
    logging_level = logging.DEBUG if DEBUG else logging.INFO
    setup_logging(level=logging_level)

silence_external_loggers()

# notebook constant
SPLIT_THETA_DATASETS = [
    "20250402_20X",
    "20250409_20X",
    "20250428_20X",
    "20250604_20X",
    "20250618_20X",
    "20250716_20X",
] + load_dataset_collection_config("perturbation").datasets

DATASET_COLLECTION_NAME = "timelapse"

KERNEL_BANDWIDTH = 0.175  # bandwidth for kernel density estimation in KM calculation

INCLUDE_CELL_PILING = False
INCLUDE_NOT_STEADY_STATE = False

POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]

BIN_WIDTH = 0.075

TICK_STEP_NUM = [15, 5]

BIN_LIMITS_THETA = (-np.pi, np.pi)
BIN_LIMITS_RADIUS = (0, 2.75)

# %%
# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# only need first two PCs
pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)
# %%
# loop over datasets in collection
# plot summary plots
# compute drift and diffusion coefficients in polar coordinates
for dataset_name in load_dataset_collection_config(DATASET_COLLECTION_NAME).datasets:
    fig_savedir_summary = get_output_path(__file__, "summary_plots", dataset_name)
    fig_savedir_km = get_output_path(__file__, "kramers_moyal", dataset_name)
    dataset_config = load_dataset_config(dataset_name)

    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca=pca,
        include_cell_piling=INCLUDE_CELL_PILING,
        include_not_steady_state=INCLUDE_NOT_STEADY_STATE,
    )

    df_by_flow, shear_stress_list = split_dataset_by_flow(
        df,
        dataset_config,
    )

    for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
        dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
        fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

        bin_limits = [BIN_LIMITS_THETA, BIN_LIMITS_RADIUS]

        is_low_shear_regime = shear_stress < 11.0 and shear_stress > 0.0

        if dataset_name in SPLIT_THETA_DATASETS or is_low_shear_regime:
            df_[ColumnName.POLAR_ANGLE] = df_[ColumnName.POLAR_ANGLE].apply(
                lambda x: x + 2 * np.pi if x < 0 else x
            )
            bin_limits = [(0, 2 * np.pi), BIN_LIMITS_RADIUS]

        bins, centers = get_bins(
            bin_widths=(BIN_WIDTH, BIN_WIDTH),
            bin_limits=bin_limits,
        )

        for i, column_name in enumerate(POLAR_COLUMN_NAMES):
            # plot per position mean
            fig, ax = plt.subplots()

            for pos, df_pos in df_.groupby(ColumnName.POSITION):
                df_pos_ = df_pos.sort_values(by=ColumnName.TIMEPOINT)
                mean_over_crops = df_pos_.groupby(ColumnName.TIMEPOINT)[column_name].mean()
                timepoints = df_pos_[ColumnName.TIMEPOINT].unique()
                ax.plot(timepoints, mean_over_crops, label=pos)

            ax.legend()
            ax.set_xlabel("frame number")
            ax.set_ylabel(column_name)
            ax.set_title(fig_title)
            save_plot_to_path(
                fig, fig_savedir_summary, f"{dataset_name_flow}_{column_name}_average"
            )

            # plot histogram heatmap over time
            num_bins = len(bins[i]) - 1
            frame_min = df_[ColumnName.TIMEPOINT].min()
            frame_max = df_[ColumnName.TIMEPOINT].max()
            num_frames = frame_max - frame_min + 1
            hist_array = np.zeros((num_bins, num_frames))

            for t, df_frame in df_.groupby(ColumnName.TIMEPOINT):
                timepoint_idx = int(t - frame_min)
                values = df_frame[column_name].values
                hist = np.histogram(values, bins=bins[i], density=True)[0]
                hist_array[:, timepoint_idx] = hist

            fig, ax = plt.subplots()
            ax.imshow(
                hist_array, aspect="auto", cmap="inferno", interpolation="nearest", origin="lower"
            )

            ax.set_ylabel(column_name)
            ax.set_xlabel("frame_number")
            ax.set_xticks(np.arange(frame_min, frame_max + 1, step=100))
            ax.set_xticklabels(np.arange(frame_min, frame_max + 1, step=100))
            ax.set_yticks(np.arange(0, num_bins + 1, step=TICK_STEP_NUM[i]))
            ax.set_yticklabels(np.round(bins[i], 2)[:: TICK_STEP_NUM[i]])
            ax.set_title(fig_title)
            save_plot_to_path(
                fig, fig_savedir_summary, f"{dataset_name_flow}_{column_name}_histogram_heatmap"
            )

        # compute Kramers-Moyal coefficients
        for i, column_name in enumerate(POLAR_COLUMN_NAMES):
            traj_list = []
            d_traj_list = []
            for crop_index, df_crop in df_.groupby(ColumnName.CROP_INDEX):
                df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)

                # add column giving difference in timepoint between consecutive dataframe rows
                # convert NaN to 0 -- occurs at end of trajectory
                df_crop_[f"{ColumnName.TIMEPOINT}_diff"] = (
                    df_crop_[ColumnName.TIMEPOINT].diff().shift(-1).fillna(0)
                )
                # compute one-step differences in polar angle and radius
                if column_name == ColumnName.POLAR_ANGLE:
                    angle_diffs = get_smallest_angle_difference(
                        df_crop_[column_name].values[1:],
                        df_crop_[column_name].values[:-1],
                        units="rad",
                    )
                    df_crop_[f"{column_name}_diff"] = np.concatenate(
                        (
                            angle_diffs,
                            np.array([np.nan]),
                        )  # no valid difference at end of trajectory
                    )
                else:
                    df_crop_[f"{column_name}_diff"] = df_crop_[column_name].diff().shift(-1)

                # trajectory values to keep -- only keep steps where time difference is 1
                # and also the last point in the trajectory (which has time difference 0)
                traj_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] <= 1

                # for the gradient, only keep steps where time difference is exactly 1
                # i.e., no valid difference at the end of the trajectory (only forward differences)
                gradient_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] == 1

                traj_list.append(df_crop_[traj_mask][column_name].values)
                d_traj_list.append(df_crop_[gradient_mask][f"{column_name}_diff"].values)

            drift, diffusion = get_kramers_moyal(
                traj_list,
                d_traj_list,
                bins=[bins[i]],
                dt=5,
                kernel_params={"kernel": "gaussian", "bandwidth": KERNEL_BANDWIDTH},
            )

            variable_name = "\\theta" if column_name == ColumnName.POLAR_ANGLE else "r"
            fig, ax = plot_1d_drift(
                centers[i],
                drift,
                variable_name,
                data_for_density=df_[column_name].values,
                density_kernel_bandwidth=KERNEL_BANDWIDTH,
            )

            # find zero crossing -- look at sign changes
            drift_signed = np.sign(drift)
            sign_changes = np.where(np.diff(drift_signed))[0]
            for idx in sign_changes:
                # use the point closer to zero drift (before or after zero-crossing)
                point_1 = centers[i][idx]
                point_2 = centers[i][idx + 1]
                idx_ = idx if abs(drift[idx]) < abs(drift[idx + 1]) else idx + 1

                fpt_candidate = centers[i][idx_]
                ax.plot(fpt_candidate, drift[idx_], "bo", markersize=5)
                ax.vlines(
                    centers[i][idx_],
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    colors="b",
                    linestyles="dashed",
                    alpha=0.5,
                    label=f"${variable_name}^* =$ {np.round(fpt_candidate,2)}",
                )

            ax.legend()
            ax.set_title(fig_title)
            save_plot_to_path(fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_drift")

            fig, ax = plot_1d_diffusion(
                centers[i],
                diffusion,
                variable_name,
                data_for_density=df_[column_name].values,
                density_kernel_bandwidth=KERNEL_BANDWIDTH,
            )
            ax.set_ylim((0.0, 1.1 * ax.get_ylim()[1]))
            ax.set_title(dataset_name)
            save_plot_to_path(fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_diffusion")
# %%
