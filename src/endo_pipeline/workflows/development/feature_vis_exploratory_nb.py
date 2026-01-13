# %%
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    get_smallest_angle_difference,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
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

# %%
# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# only need first two PCs
pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=2)
# %%

include_cell_piling = False
include_not_steady_state = False

# load PC-projected dataframe for an example dataset
dataset_name = "20250618_20X"  # replicate 1 "no flow"
df = get_dataframe_for_dynamics_workflows(
    dataset_name,
    dataframe_manifest,
    pca=pca,
    include_cell_piling=include_cell_piling,
    include_not_steady_state=include_not_steady_state,
)

# df[ColumnName.POLAR_ANGLE] = df[ColumnName.POLAR_ANGLE].apply(
#     lambda x: x + 2 * np.pi if x < 0 else x
# )

# %%
for column_name in [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]:
    fig, ax = plt.subplots()

    for pos, df_pos in df.groupby(ColumnName.POSITION):
        df_pos_ = df_pos.sort_values(by=ColumnName.TIMEPOINT)
        mean_over_crops = df_pos_.groupby(ColumnName.TIMEPOINT)[column_name].mean()
        timepoints = df_pos_[ColumnName.TIMEPOINT].unique()
        ax.plot(timepoints, mean_over_crops, label=pos)

    ax.legend()
    ax.set_xlabel("frame number")
    ax.set_ylabel(column_name)
    ax.set_title(dataset_name)
    plt.show()

# %%
bin_width = 0.05
bin_limits = [(-np.pi, np.pi), (0, 2.75)]
# bin_limits = [(0, 2 * np.pi), (0, 2.75)]
bins, centers = get_bins(
    bin_widths=(bin_width, bin_width),
    bin_limits=bin_limits,
)
column_names = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]
tick_step_num = [15, 5]
num_frames = df[ColumnName.TIMEPOINT].max() + 1

# %%
for i in range(2):
    fig, ax = plt.subplots()
    column_values = df[column_names[i]].values
    ax.hist(column_values, bins[i], density=True)

    ax.set_xlabel(column_names[i])
    ax.set_ylabel("density")
    ax.set_title(dataset_name)
    plt.show()

# %%
for i in range(2):
    column_name = column_names[i]

    num_bins = len(bins[i]) - 1
    hist_array = np.zeros((num_bins, num_frames))

    for t, df_frame in df.groupby(ColumnName.TIMEPOINT):
        values = df_frame[column_name].values
        hist = np.histogram(values, bins=bins[i], density=True)[0]
        hist_array[:, int(t)] = hist

    fig, ax = plt.subplots()
    ax.imshow(hist_array, aspect="auto", cmap="inferno", interpolation="nearest", origin="lower")

    ax.set_ylabel(column_name)
    ax.set_xlabel("frame_number")
    ax.set_xticks(np.arange(0, num_frames, step=100))
    ax.set_xticklabels(np.arange(0, num_frames, step=100))
    ax.set_yticks(np.arange(0, num_bins + 1, step=tick_step_num[i]))
    ax.set_yticklabels(np.round(bins[i], 2)[:: tick_step_num[i]])
    ax.set_title(dataset_name)
    plt.show()

# %%
theta_traj_list = []
d_theta_list = []
r_traj_list = []
d_r_list = []
for crop_index, df_crop in df.groupby(ColumnName.CROP_INDEX):
    df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)
    # compute one-step angle differences

    # add column giving difference in timepoint between consecutive dataframe rows
    # convert NaN to 0 -- occurs at end of trajectory
    df_crop_[f"{ColumnName.TIMEPOINT}_diff"] = (
        df_crop_[ColumnName.TIMEPOINT].diff().shift(-1).fillna(0)
    )
    # compute one-step differences in polar angle and radius
    angle_diffs = get_smallest_angle_difference(
        df_crop_[ColumnName.POLAR_ANGLE].values[1:],
        df_crop_[ColumnName.POLAR_ANGLE].values[:-1],
        units="rad",
    )
    df_crop_[f"{ColumnName.POLAR_ANGLE}_diff"] = np.concatenate(
        (angle_diffs, np.array([np.nan]))  # no valid difference at end of trajectory
    )
    # df_crop_[ColumnName.POLAR_ANGLE].diff().shift(-1)
    df_crop_[f"{ColumnName.POLAR_RADIUS}_diff"] = df_crop_[ColumnName.POLAR_RADIUS].diff().shift(-1)

    # trajectory values to keep -- only keep steps where time difference is 1
    # and also the last point in the trajectory (which has time difference 0)
    traj_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] <= 1

    # for the gradient, only keep steps where time difference is exactly 1
    # i.e., no valid difference at the end of the trajectory (only forward differences)
    gradient_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] == 1

    theta_traj_list.append(df_crop_[traj_mask][ColumnName.POLAR_ANGLE].values)
    d_theta_list.append(df_crop_[gradient_mask][f"{ColumnName.POLAR_ANGLE}_diff"].values)
    r_traj_list.append(df_crop_[traj_mask][ColumnName.POLAR_RADIUS].values)
    d_r_list.append(df_crop_[gradient_mask][f"{ColumnName.POLAR_RADIUS}_diff"].values)

# %%
drift_theta, diffusion_theta = get_kramers_moyal(
    theta_traj_list,
    d_theta_list,
    bins=[bins[0]],
    dt=5,
    kernel_params={"kernel": "gaussian", "bandwidth": 0.2},
)

drift_r, diffusion_r = get_kramers_moyal(
    r_traj_list,
    d_r_list,
    bins=[bins[1]],
    dt=5,
    kernel_params={"kernel": "gaussian", "bandwidth": 0.2},
)
# %%
fig, ax = plt.subplots()
ax.plot(centers[0], drift_theta, "k-")
ax.plot(centers[0], np.zeros_like(centers[0]), "r--", alpha=0.5)
ax.set_ylim((-0.1, 0.1))
ax.set_xlabel("polar angle $\\theta$ (rad)")
ax.set_ylabel("drift in $\\theta$ (rad/min)")

# find zero crossing
where_zero = np.argmin(np.abs(drift_theta))
ax.plot(centers[0][where_zero], drift_theta[where_zero], "bo")
ax.vlines(
    centers[0][where_zero],
    ax.get_ylim()[0],
    ax.get_ylim()[1],
    colors="b",
    linestyles="dashed",
    alpha=0.5,
    label=f"$\\theta^* =$ {np.round(centers[0][where_zero],2)} rad",
)
ax.legend()

fig, ax = plt.subplots()
ax.plot(centers[0], diffusion_theta, "k-")
ax.set_ylim((0.0, 1.1 * ax.get_ylim()[1]))
ax.set_xlabel("polar angle $\\theta$ (rad)")
ax.set_ylabel("MSD in $\\theta$ (rad^2/min)")

# %%
fig, ax = plt.subplots()
ax.plot(centers[1], drift_r, "k-")
ax.plot(centers[1], np.zeros_like(centers[1]), "r--", alpha=0.5)
ax.set_xlabel("polar radius $r$")
ax.set_ylabel("drift in $r$ (1/min)")

# find zero crossing
where_zero = np.argmin(np.abs(drift_r))
ax.plot(centers[1][where_zero], drift_r[where_zero], "bo")
ax.vlines(
    centers[1][where_zero],
    ax.get_ylim()[0],
    ax.get_ylim()[1],
    colors="b",
    linestyles="dashed",
    alpha=0.5,
    label=f"$r^* =$ {np.round(centers[1][where_zero],2)} rad",
)
ax.legend()

fig, ax = plt.subplots()
ax.plot(centers[1], diffusion_r, "k-")
ax.set_ylim((0.0, 1.1 * ax.get_ylim()[1]))
ax.set_xlabel("polar radius $r$ (rad)")
ax.set_ylabel("MSD in $r$ (1/min)")
# %%
