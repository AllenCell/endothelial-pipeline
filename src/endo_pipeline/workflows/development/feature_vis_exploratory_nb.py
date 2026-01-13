# %%
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    get_smallest_angle_difference,
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
num_bins = [int(np.ceil((bin_limits[i][1] - bin_limits[i][0]) / bin_width)) for i in range(2)]
bins = [np.linspace(bin_limits[i][0], bin_limits[i][1], num_bins[i] + 1) for i in range(2)]
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

    hist_array = np.zeros((num_bins[i], num_frames))

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
    ax.set_yticks(np.arange(0, num_bins[i] + 1, step=tick_step_num[i]))
    ax.set_yticklabels(np.round(bins[i], 2)[:: tick_step_num[i]])
    ax.set_title(dataset_name)
    plt.show()

# %%
theta_traj_list = []
d_theta_list = []
for crop_index, df_crop in df.groupby(ColumnName.CROP_INDEX):
    df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)
    # compute one-step angle differences

    # add column giving difference in timepoint between consecutive dataframe rows
    df_crop_["timepoint_diff"] = df_crop_[ColumnName.TIMEPOINT].diff().shift(-1)

    angle_diffs = get_smallest_angle_difference(
        df_crop_[ColumnName.POLAR_ANGLE].values[:-1],
        df_crop_[ColumnName.POLAR_ANGLE].values[1:],
        units="rad",
    )
    theta_traj_list.append(df_crop_[ColumnName.POLAR_ANGLE].values)
    d_theta_list.append(angle_diffs)


# %%
fig, ax = plt.subplots()
for traj, d_traj in zip(theta_traj_list, d_theta_list, strict=True):
    arg_sort = np.argsort(traj[:-1])
    ax.plot(traj[:-1][arg_sort], d_traj[arg_sort], "k.", alpha=0.5)
# %%
