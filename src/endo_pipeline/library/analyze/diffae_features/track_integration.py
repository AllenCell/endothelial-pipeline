import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from cellsmap.util.manifest_io import get_diffae_manifest
from src.endo_pipeline.configs import get_model_manifest, load_dataset_config, load_model_config
from src.endo_pipeline.configs.dataset_io import (
    get_live_segmentation_features_manifest,
    get_reference_datasets,
    ipython_cli_flexecute,
)
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import get_output_path, load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import plot_one_slice_quiver
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import (
    get_valid_slice_indexes,
    plot_quiver_slices_from_diffae_table,
)

logger = logging.getLogger(__name__)


# def get_and_process_diffae_data(dataset_name: str) -> pd.DataFrame:
#     # read in the grid crop-based diffae features
#     diffae_grid_crops = get_diffae_manifest(dataset_name)
#     diffae_grid_crops = diffae_preproc.add_crop_index(diffae_grid_crops)
#     diffae_grid_crops = diffae_preproc.add_description_column(
#         diffae_grid_crops, dataset_name, simple=True
#     )  # add description column (e.g., 48hr_High)

#     # in Erin's code in workflows/flow_field3d/preprocessing.py
#     # adding the dataset name to the crop index was required to
#     # make the crop index unique if multiple datasets were used
#     diffae_grid_crops["crop_index"] = (
#         diffae_grid_crops["dataset"] + "_" + diffae_grid_crops["crop_index"].astype(str)
#     )

#     return diffae_grid_crops


def merge_diffae_feats_liveseg_feats_tables(
    diffae_tracking_df: pd.DataFrame,
    live_seg_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the DiffAE tracking data with the live segmentation features data.

    Parameters:
        diffae_tracking_df (pd.DataFrame): DataFrame containing DiffAE tracking data.
        live_seg_feats_df (pd.DataFrame): DataFrame containing live segmentation features data.

    Returns:
        pd.DataFrame: Merged DataFrame with DiffAE and live segmentation features.
    """
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    diffae_tracking_df["is_unique"] = diffae_tracking_df.groupby(
        ["dataset", "position", "frame_number", "track_id"]
    )["frame_number"].transform(lambda t: t.nunique() == t.size)
    diffae_tracking_df = diffae_tracking_df[diffae_tracking_df["is_unique"]]

    # give the crop_index column the same value as the track_ids
    diffae_tracking_df["crop_index"] = diffae_tracking_df["track_id"]
    diffae_tracking_df = diffae_preproc.add_description_column(
        diffae_tracking_df, dataset_name, simple=True
    )  # add description column (e.g., 48hr_High)
    diffae_tracking_df["track_id"] = diffae_tracking_df["track_id"].astype(int)
    diffae_tracking_df.rename(columns={"position": "position_as_str"}, inplace=True)

    logging.debug("processing the live segmentation features data...")
    live_seg_feats_df["position_as_str"] = live_seg_feats_df["position"].transform(
        lambda x: "P" + str(x)
    )
    live_seg_feats_df["track_id"] = live_seg_feats_df["track_id"].astype(int)

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merged_feats_df = pd.merge(
        left=live_seg_feats_df,
        right=diffae_tracking_df,
        how="left",
        left_on=["dataset_name", "position_as_str", "image_index", "track_id"],
        right_on=["dataset", "position_as_str", "frame_number", "track_id"],
        validate="one_to_one",
    )

    return merged_feats_df


def get_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> pd.DataFrame:

    logging.debug(f"Loading dataset config file for dataset: {dataset_name}...")
    dataset_config = load_dataset_config(dataset_name)

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_fms_id = dataset_config.diffae_tracking_integration_fmsid
    if diffae_fms_id is None:
        logging.warning(
            f"No DiffAE track integration FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    diffae_tracking_df = load_dataframe_from_fms(diffae_fms_id)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_fmsid = dataset_config.live_merged_seg_features_manifest_fmsid
    if live_seg_fmsid is None:
        logging.warning(
            f"No live segmentation features FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    live_seg_feats_df = load_dataframe_from_fms(live_seg_fmsid)  # this takes a minute

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    return merged_feats_df


def get_traj_and_flowfield(
    df: pd.DataFrame,
    bounds: Pipeline,
) -> tuple[np.ndarray, dict]:

    # load default config, get kernel params
    dynamics_config = load_dynamics_config("default")
    kernel_params = dynamics_config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
    dt = dynamics_config["dt"]

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = [0.0, 2880.0]

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array([-0.1, -0.7, -0.1])

    num_bins = [50, 50, 50]
    bins, centers = rh.get_bins(num_bins, bin_limits=bounds)

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = [f"pc{pc+1}" for pc in range(3)]

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = rh.get_traj_and_diff(df, cols)

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = rh.get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = ddff.compute_extrapolated_vector_field(
        drift_km, centers, interpolator="nearest"
    )

    # solve IVP, get back trajectory
    print("Trying to solve ODE...")
    traj = ddff.solve_ddff_ode(flow_field_dict, init, time_span)
    print("ODE solved.")

    return traj, flow_field_dict


# def compute_ff_test() -> :

#     # load dataframe and get top 3 PCs
#     df = preprocessing.get_manifest_for_dynamics_workflows(model_manifest, pca)
#     pc_column_names = get_pc_column_names(df, pc_axes=[0, 1, 2])

#     # get list of per-crop trajectories, the corresponding
#     # displacement vectors, and time differences
#     traj_list, d_traj_list = regression_helper.get_traj_and_diff(df, pc_column_names)
#     # get drift and diffusion estimates
#     # (Kramers-Moyal coefficients)
#     drift_km, diff_km = regression_helper.get_kramers_moyal(
#         traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
#     )

#     # compute interpolated flow field - drift
#     flow_field_dict = compute_extrapolated_vector_field(drift_km, centers, interpolator="nearest")
#     # save flow field dictionary as npy
#     np.save(
#         output_savedir / f"flow_field_dict_{model_manifest.dataset_name}.npy",
#         flow_field_dict,  # type: ignore
#         allow_pickle=True,
#     )

#     # compute interpolated diffusion field
#     # (diagonal diffusion tensor represented as 3D vector field)
#     diffusion_field_dict = compute_extrapolated_vector_field(
#         diff_km, centers, interpolator="nearest"
#     )
#     # save diffusion field dictionary as npy
#     np.save(
#         output_savedir / f"diffusion_field_dict_{model_manifest.dataset_name}.npy",
#         diffusion_field_dict,  # type: ignore
#         allow_pickle=True,
#     )

#     ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
#     # with initial conditions given by init
#     # solve IVP, get back trajectory
#     traj = solve_ddff_ode(flow_field_dict, init, time_span)

#     return traj


dataset_name = "20241120_20X"

## NOTE CODE FOR DEV ONLY
live_seg_feats_df = pd.read_csv(
    r"C:\Users\serge.parent\Documents\projects\cellsmap\results\2025-07-16\make_seg_feats_manifest\segmentation_features_manifests\20241120_20X_live_segmentation_features.tsv",
    sep="\t",
)
diffae_tracking_df = pd.read_parquet(
    r"C:\Users\serge.parent\Documents\projects\cellsmap\results\models\diffae_04_10\20241120_20X\predict_20241120_20X_diffae_04_10_tracked_crop_features.parquet"
)

merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)
## NOTE END OF DEV CODE


# load the tables
merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name)

# filter the merged table
merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

# remove any rows that were not evaluated by the model and thus have no mlflow_id
merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

# fit the PCA (uses the reference datasets)
pca = fit_pca()

# read in the grid crop-based diffae features
model_name = diffae_tracking_df["model_name"].unique()[0]
model_config = load_model_config(model_name)
model_manifest = get_model_manifest(dataset_name, model_config)
diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

# add the PC columns to the track-based DiffAE table
# (the grid-based DiffAE table already has them, but
# but I believe that the columns are named "feat_0",
# "feat_1", etc. when they should be named "pc1",
# "pc2", etc.)
merged_feats_df = project_manifest_to_pcs(merged_feats_df, pca)
# df_all_positions.dropna(axis="index", how="any", subset="is_unique", inplace=True)

# use the full set of datasets to be analyzed for the bounds
datasets_for_bounds = [
    "20241120_20X",
    "20250409_20X",
    "20241217_20X",
    "20250428_20X",
    "20250319_20X",
    "20250326_20X",
]

# NOTE IT TAKES WAY LONGER TO SOLVE THE ODE IF DATASET_NAME_LIST IS 20241120_20X
# AND THEREFORE THE BOUNDS ARE DIFFERENT THAN IF THE BOUNDS ARE SET TO THE DEFAULT
# LIST OF DATASETS ABOVE (BOUNDS TOO RESTRICTIVE?)

model_manifest_list = [
    get_model_manifest(dataset_name, model_config) for dataset_name in datasets_for_bounds
]
bounds = ddff.set_3d_bounds_from_data(model_manifest_list, pca)

# This takes about 1 minute to run with
# datasets_for_bounds = ["20241120_20X", "20250409_20X",
# "20241217_20X", "20250428_20X", "20250319_20X", "20250326_20X"]
logger.debug("getting trajectory and flow field for grid-based crops...")
traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
    df=diffae_grid_crops,
    bounds=bounds,
)

logger.debug("saving the trajectory data from the grid-based crops...")
out_subdir_traj = get_output_path(Path(__file__).stem, dataset_name)
np.save(out_subdir_traj / f"{dataset_name}_traj_grids.npy", traj_grids)

# This takes about 5 minutes to run with
# datasets_for_bounds = ["20241120_20X", "20250409_20X",
# "20241217_20X", "20250428_20X", "20250319_20X", "20250326_20X"]
logger.debug("getting trajectory and flow field for tracks-based crops...")
traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(
    df=merged_feats_df,
    bounds=bounds,
)

logger.debug("saving the trajectory data from the track-based crops...")
out_subdir_traj = get_output_path(Path(__file__).stem, dataset_name)
np.save(out_subdir_traj / f"{dataset_name}_traj_tracks.npy", traj_tracks)


fig, axs = plot_quiver_slices_from_diffae_table(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)

fig, axs = plot_quiver_slices_from_diffae_table(
    merged_feats_df, traj_tracks, flow_field_dict_tracks
)


# NOTE VERY ROUGH WORK IN PROGRESS BELOW

# def get_quiver_args():
yvalids_grids, zvalids_grids = get_valid_slice_indexes(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)

slice_indexes_ = (zvalids_grids, yvalids_grids)

# get flow field
v1, v2, v3 = flow_field_dict_grids["vectors"]

# get grid and grid spacing
xgrid, ygrid, zgrid = flow_field_dict_grids["grid"]

velocities, grid, slice_indexes = (v1, v2), (xgrid, ygrid), slice_indexes_[0]


my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

x1_grid = grid[0][slice_indexes].reshape(my_shape)
x2_grid = grid[1][slice_indexes].reshape(my_shape)
dx1 = velocities[0][slice_indexes].reshape(my_shape)
dx2 = velocities[1][slice_indexes].reshape(my_shape)

color, scale = "dimgrey", 1
plt.quiver(
    x1_grid.squeeze().T,
    x2_grid.squeeze().T,
    dx1.squeeze().T,
    dx2.squeeze().T,
    color=color,
    scale=scale,
)

test = np.dot

v1_grids, v2_grids, v3_grids = flow_field_dict_grids["vectors"]
g1_grids, g2_grids, g3_grids = flow_field_dict_grids["grid"]
v1_tracks, v2_tracks, v3_tracks = flow_field_dict_tracks["vectors"]
g1_tracks, g2_tracks, g3_tracks = flow_field_dict_tracks["grid"]

test1 = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
test2 = np.asarray(list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks))))


vx, vy = [0, 1], [1, 0]
ux, uy = [0, 1], [1, 0]

v, u = np.array([0, 1]), np.array([1, 0])

us = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1]])
vs = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, 2]])

np.vectorize(np.dot)(us, vs)
np.dot(us, vs)

(vs * us).sum(-1)


def get_vector_vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle_rad


# [get_vector_vector_angle(vs[i], us[i], as_degrees=True) for i in range(len(vs))]


test_ang = np.array([get_vector_vector_angle(test1[i], test2[i]) for i in range(len(test1))])


## below is PROBABLY the same as test_ang above, and
## definitely faster; I think it's just rounding errors
test_dot = np.einsum("ij,ij->i", test1, test2)
test_norm1 = np.linalg.norm(test1, axis=1)
test_norm2 = np.linalg.norm(test2, axis=1)
test_ang2 = np.arccos(test_dot / (test_norm1 * test_norm2))

test_ang[:10]
test_ang2[:10]
test_ang[:10] == test_ang2[:10]


test_ang_arr = test_ang.reshape((50, 50, 50))
angles = test_ang_arr[slice_indexes].reshape(my_shape)

fig, ax = plt.subplots(figsize=(6, 6))
hist2D = ax.imshow(
    angles.squeeze(), cmap="seismic_r", vmin=-np.pi, vmax=np.pi, origin="lower", label="angle (rad)"
)
plt.colorbar(hist2D)


fig, axs = plot_quiver_slices_from_diffae_table(
    merged_feats_df, traj_tracks, flow_field_dict_tracks
)
fig.suptitle("DiffAE tracked crops")

fig, axs = plot_quiver_slices_from_diffae_table(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)
fig.suptitle("DiffAE grid crops")


fig, axs = plot_quiver_slices_from_diffae_table(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)
fig.suptitle("deviation hist2D")


from matplotlib.lines import Line2D

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_one_slice_quiver(
    velocities=(v1_grids, v2_grids),
    grid=(g1_grids, g2_grids),
    slice_indexes=slice_indexes_[0],
    ax=ax,
    color="blue",
)
plot_one_slice_quiver(
    velocities=(v1_tracks, v2_tracks),
    grid=(g1_tracks, g2_tracks),
    slice_indexes=slice_indexes_[0],
    ax=ax,
    color="red",
)
custom_lines = [
    Line2D([0], [0], color="red", lw=2, label="seg-based DiffAE features"),
    Line2D([0], [0], color="blue", lw=2, label="grid-based DiffAE features"),
]
ax.legend(custom_lines, [str(x.get_label()) for x in custom_lines], loc="upper right")


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fig.suptitle("Angular deviation hist2D")
hist2D = ax.imshow(
    # hist2D = axs[0].imshow(
    angles.squeeze(),
    cmap="seismic_r",
    vmin=-np.pi,
    vmax=np.pi,
    origin="lower",
    label="angle (rad)",
)
plt.colorbar(hist2D)

for nm, df in merged_feats_df.groupby(["dataset_name", "position", "track_id"]):
    break

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_one_slice_quiver(
    velocities=(v1_grids, v2_grids),
    grid=(g1_grids, g2_grids),
    slice_indexes=slice_indexes_[0],
    ax=ax,
    color="blue",
)
sns.scatterplot(
    data=df, x="pc1", y="pc2", hue="frame_number", marker=".", palette="Spectral", ax=ax, s=100
)
ax.plot(
    df["pc1"],
    df["pc2"],
    color="grey",
    linewidth=0.5,
    linestyle="-",
    # label="track trajectory"
)

df["dpc1"] = df["pc1"].diff()
df["dpc2"] = df["pc2"].diff()
