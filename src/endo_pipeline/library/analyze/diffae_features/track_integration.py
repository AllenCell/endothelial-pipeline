import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from dask import dataframe as dd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.endo_pipeline.configs import (
    get_model_manifest,
    load_dataset_collection_config,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
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
from src.endo_pipeline.library.analyze.optical_flow_calculator import (
    one_direction_vector_field_example,
)
from src.endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import plot_one_slice_quiver
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import (
    get_valid_slice_indexes,
    grid_vs_track_vec_angle_hist2d,
    grid_vs_track_vec_dot_prod_hist2d,
    plot_grid_vs_tracks_flow_field,
    plot_quiver_slices_from_diffae_table,
)
from src.endo_pipeline.library.visualize.seg_features.general_standard_plots import hist_2D_of_feats

logger = logging.getLogger(__name__)


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
    dataset_name = sequence_to_scalar(diffae_tracking_df["dataset"])
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    diffae_tracking_df["is_unique"] = diffae_tracking_df.groupby(
        ["dataset", "position", "frame_number", "track_id"]
    )["frame_number"].transform(lambda t: t.nunique() == t.size)
    diffae_tracking_df = diffae_tracking_df[diffae_tracking_df["is_unique"]]

    # give the crop_index column the same value as the track_ids
    # diffae_tracking_df["crop_index"] = diffae_tracking_df["track_id"]
    diffae_tracking_df["crop_index"] = (
        diffae_tracking_df.groupby(["position", "track_id"], as_index=False).ngroup().astype(int)
    )
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
    load_precomputed_trajectories: Path | None,
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

    if load_precomputed_trajectories is not None:
        logger.debug("Loading precomputed trajectories...")
        traj = np.load(load_precomputed_trajectories)
    else:
        # solve IVP, get back trajectory
        logger.debug("Trying to solve ODE...")
        traj = ddff.solve_ddff_ode(flow_field_dict, init, time_span)
        logger.debug("ODE solved.")

    return traj, flow_field_dict


def get_vector_vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle_rad


def get_vector_vector_angle_fast(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    ## This is PROBABLY the same as test_ang above, and
    ## definitely faster; I think any differences are
    ## just floatingpoint rounding errors
    dot_prod = np.einsum("ij,ij->i", v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    angle_rad = np.arccos(dot_prod / (norm1 * norm2))
    return angle_rad


def get_approx_vec_from_grid(
    pc1_pc2_points: np.ndarray,
    # pc1: float, pc2: float,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: np.ndarray,
    # ) -> tuple[float, float]:
    # ) ->  tuple[np.ndarray, np.ndarray]:
) -> np.ndarray:

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes])))
    # pc1_pc2_point = np.array([pc1, pc2])
    # pc1_pc2_points = pc1_pc2_points.reshape((1, point_grids_pc1pc2.shape[-1], len(pc1_pc2_points)))
    # point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=-1)
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    # min_idx = np.where(dist_grid == dist_grid.min())
    # v1_grids_approx = v1_grids[slice_indexes].reshape((50,50))[min_idx].item()
    # v2_grids_approx = v2_grids[slice_indexes].reshape((50,50))[min_idx].item()

    min_idx = np.argmin(dists, axis=0)
    v1_grids_approx = v1_grids[slice_indexes][min_idx]
    v2_grids_approx = v2_grids[slice_indexes][min_idx]

    # plot the distance grid
    # plt.imshow(dist_grid.T, origin="lower")

    # return v1_grids_approx, v2_grids_approx
    return np.array(tuple(zip(v1_grids_approx.tolist(), v2_grids_approx.tolist())))


def get_approx_point_from_grid(
    pc1_pc2_points: np.ndarray,
    # pc1: float, pc2: float,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: np.ndarray,
    # ) -> tuple[float, float]:
    # ) ->  tuple[np.ndarray, np.ndarray]:
) -> np.ndarray:
    # pc1_pc2_point = np.array([pc1, pc2])
    # point_grids_pc1pc2 = np.asarray(list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes])))
    # dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_point, axis=1)
    # dist_grid = dists.reshape((50,50))
    # # get the index of the closest point
    # min_idx = np.where(dist_grid == dist_grid.min())
    # # v1_grids_approx = v1_grids[slice_indexes].reshape((50,50))[min_idx].item()
    # # v2_grids_approx = v2_grids[slice_indexes].reshape((50,50))[min_idx].item()
    # g1_grids_approx = g1_grids[slice_indexes].reshape((50,50))[min_idx].item()
    # g2_grids_approx = g2_grids[slice_indexes].reshape((50,50))[min_idx].item()

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes])))
    # pc1_pc2_point = np.array([pc1, pc2])
    # pc1_pc2_points = pc1_pc2_points.reshape((1, point_grids_pc1pc2.shape[-1], len(pc1_pc2_points)))
    # point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=-1)
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    # min_idx = np.where(dist_grid == dist_grid.min())
    # v1_grids_approx = v1_grids[slice_indexes].reshape((50,50))[min_idx].item()
    # v2_grids_approx = v2_grids[slice_indexes].reshape((50,50))[min_idx].item()

    min_idx = np.argmin(dists, axis=0)
    g1_grids_approx = g1_grids[slice_indexes][min_idx]
    g2_grids_approx = g2_grids[slice_indexes][min_idx]

    # return g1_grids_approx, g2_grids_approx
    return np.array(tuple(zip(g1_grids_approx.tolist(), g2_grids_approx.tolist())))


def plot_pc_integrated_track(
    dataset_name: str,
    position_name: str,
    track_id: int,
    df: pd.DataFrame,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    slice_indexes: np.ndarray,
    out_subdir: Path,
) -> None:

    out_subdir_integrated_tracks = out_subdir / "integrated_tracks"
    out_subdir_integrated_tracks.mkdir(parents=True, exist_ok=True)

    out_subdir_integrated_tracks_hued = out_subdir / "integrated_tracks_hued"
    out_subdir_integrated_tracks_hued.mkdir(parents=True, exist_ok=True)

    # break
    # find nearest PC1 and PC2 and vector in the grid-based
    # PC-space for each PC1, PC2 combination in the tracks
    # NOTE this could probably be vectorized similar to taking
    # the minimum distance when solving a centroid-to-centroid
    # tracking problem
    # df[["approx_bin_pc1","approx_bin_pc2"]] = np.asarray(df.apply(lambda df_row: get_approx_grid_bin(df_row[["pc1", "pc2"]].values), axis=1).values.tolist())
    # df[["approx_vec_pc1", "approx_vec_pc2"]] = np.asarray(df.apply(lambda df_row: get_approx_grid_vec(df_row[["pc1", "pc2"]].values), axis=1).values.tolist())

    # df["approx_angle_deviation_rad"] = df.apply(
    #     lambda df_row: get_vector_vector_angle(
    #         df_row[["approx_vec_pc1", "approx_vec_pc2"]].values,
    #         df_row[["dpc1", "dpc2"]].values,
    #     ), axis=1)  # type: ignore
    # df["track_angular_deviation_deg"] = df["track_angular_deviation_rad"].transform(np.rad2deg)

    # break

    # plot a single track integrated into the flow field
    # shown as dots connected by arrows to give an idea
    # of the direction of motion of the cell through the
    # flow field
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,  # type: ignore
        # ds=1,
        # scale=60,
        ax=ax,
        color="blue",
    )
    ax.quiver(
        df["pc1"].iloc[:-1],
        df["pc2"].iloc[:-1],
        df["dpc1"].iloc[1:],
        df["dpc2"].iloc[1:],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.004,
    )
    sns.scatterplot(
        data=df.query("time_hours == time_hours.min()"),
        x="pc1",
        y="pc2",
        marker="o",
        color="red",
        alpha=0.7,
        lw=0,
        ax=ax,
        s=50,
        legend=False,
    )
    sns.scatterplot(
        data=df.query("time_hours == time_hours.max()"),
        x="pc1",
        y="pc2",
        marker="x",
        color="red",
        alpha=0.7,
        lw=2,
        ax=ax,
        s=50,
        legend=False,
    )
    # sns.scatterplot(
    #     data=df,
    #     x="pc1",
    #     y="pc2",
    #     marker="o",
    #     color="red",
    #     alpha=0.7,
    #     lw=0,
    #     ax=ax,
    #     s=50,
    #     legend=False,
    # )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    fig.savefig(
        out_subdir_integrated_tracks
        / f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    cmap = sns.color_palette("dark:red", as_cmap=True)
    angle_deg_to_color = lambda a: cmap(np.abs(a) / 180.0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,  # type: ignore
        # ds=1,
        # scale=60,
        ax=ax,
        color="blue",
    )
    ax.quiver(
        df["pc1"].iloc[:-1],
        df["pc2"].iloc[:-1],
        df["dpc1"].iloc[1:],
        df["dpc2"].iloc[1:],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color=angle_deg_to_color(df["track_angular_deviation_deg"].iloc[1:]),
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    fig.savefig(
        out_subdir_integrated_tracks_hued
        / f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field_hued.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ## NOTE CODE FOR DEV ONLY
# dataset_name = "20241120_20X"

# live_seg_feats_df = pd.read_csv(
#     r"C:\Users\serge.parent\Documents\projects\cellsmap\results\2025-07-16\make_seg_feats_manifest\segmentation_features_manifests\20241120_20X_live_segmentation_features.tsv",
#     sep="\t",
# )
# diffae_tracking_df = pd.read_parquet(
#     r"C:\Users\serge.parent\Documents\projects\cellsmap\results\models\diffae_04_10\20241120_20X\predict_20241120_20X_diffae_04_10_tracked_crop_features.parquet"
# )

# merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)
# ## NOTE END OF DEV CODE


## NOTE THIS NEEDS TO BE BROKEN UP INTO SMALLER AND MORE
## INFORMATIVE FUNCTIONS
def process_dataset(dataset_name: str, make_integrated_plots: bool = True) -> None:
    logger.info(f"Processing dataset: {dataset_name}")

    out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

    # out_subdir_traj_precomputed = (
    #     Path(
    #         r"C:\Users\serge.parent\Documents\projects\cellsmap\results\2025-07-22\track_integration"
    #     )
    #     / dataset_name
    # )

    # load the tables
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name)

    # filter the merged table
    merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

    # remove any rows that were not evaluated by the model and thus have no mlflow_id
    merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

    # keep only the columns that are needed for the analysis to reduce memory usage
    cols_to_keep = [
        "dataset_name",
        "position",
        "position_as_str",
        "track_id",
        "label",
        "crop_index",
        "mlflow_id",
        "model_name",
        "image_index",
        "frame_number",
        "time_hours",
        "time_minutes",
        "track_duration",
    ] + [col for col in merged_feats_df.columns if "feat" in col]

    merged_feats_df = merged_feats_df[cols_to_keep]

    # fit the PCA (uses the reference datasets)
    pca = fit_pca()

    # read in the grid crop-based diffae features
    model_name = merged_feats_df["model_name"].unique()[0]
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
        # "20241016_20X",
    ]
    # datasets_for_bounds = load_dataset_collection_config("live_20X_objective_3i_microscope").datasets
    # datasets_for_bounds = load_dataset_collection_config("pca_reference").datasets

    # NOTE IT TAKES WAY LONGER TO SOLVE THE ODE IF DATASET_NAME_LIST IS 20241120_20X
    # AND THEREFORE THE BOUNDS ARE DIFFERENT THAN IF THE BOUNDS ARE SET TO THE DEFAULT
    # LIST OF DATASETS ABOVE (BOUNDS TOO RESTRICTIVE?)

    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in datasets_for_bounds
    ]
    bounds_grids = ddff.set_3d_bounds_from_data(model_manifest_list, pca)
    bounds_tracks = []
    for pc_idx in range(len(bounds_grids)):
        pc_num = pc_idx + 1
        bounds_tracks.append(
            np.array(
                [
                    min(merged_feats_df[f"pc{pc_num}"].min(), bounds_grids[pc_idx].min()),
                    max(merged_feats_df[f"pc{pc_num}"].max(), bounds_grids[pc_idx].max()),
                ]
            )
        )

    # This takes about 1 minute to run with
    precomputed_trajectories_path = out_subdir / f"{dataset_name}_traj_grids.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for grid-based crops...")
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        df=diffae_grid_crops,
        bounds=bounds_grids,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if precomputed_trajectories_path is None:
        logger.debug("saving the trajectory data from the grid-based crops...")
        np.save(precomputed_trajectories_path, traj_grids)

    # This takes about 5 minutes to run with
    precomputed_trajectories_path = (
        precomputed_trajectories_path / f"{dataset_name}_traj_tracks.npy"
    )
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for tracks-based crops...")
    traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(
        df=merged_feats_df,
        bounds=bounds_tracks,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if precomputed_trajectories_path is None:
        logger.debug("saving the trajectory data from the track-based crops...")
        np.save(precomputed_trajectories_path, traj_tracks)

    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig.suptitle("DiffAE grid crops")

    fig, axs = plot_quiver_slices_from_diffae_table(
        merged_feats_df, traj_tracks, flow_field_dict_tracks
    )
    fig.suptitle("DiffAE tracked crops")

    # def get_quiver_args():
    yvalids_grids, zvalids_grids = get_valid_slice_indexes(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )

    slice_indexes_ = (zvalids_grids, yvalids_grids)

    # # get flow field
    # v1, v2, v3 = flow_field_dict_grids["vectors"]

    # # get grid and grid spacing
    # xgrid, ygrid, zgrid = flow_field_dict_grids["grid"]

    # velocities, grid = (v1, v2), (xgrid, ygrid)

    slice_indexes = slice_indexes_[0]
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    # x1_grid = grid[0][slice_indexes].reshape(my_shape)
    # x2_grid = grid[1][slice_indexes].reshape(my_shape)
    # dx1 = velocities[0][slice_indexes].reshape(my_shape)
    # dx2 = velocities[1][slice_indexes].reshape(my_shape)

    # color, scale = "dimgrey", 1
    # plt.quiver(
    #     x1_grid.squeeze().T,
    #     x2_grid.squeeze().T,
    #     dx1.squeeze().T,
    #     dx2.squeeze().T,
    #     color=color,
    #     scale=scale,
    # )

    v1_grids, v2_grids, v3_grids = flow_field_dict_grids["vectors"]
    g1_grids, g2_grids, g3_grids = flow_field_dict_grids["grid"]
    v1_tracks, v2_tracks, v3_tracks = flow_field_dict_tracks["vectors"]
    g1_tracks, g2_tracks, g3_tracks = flow_field_dict_tracks["grid"]

    vecs_grids = np.asarray(list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids))))
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks)))
    )

    # test_ang = np.array(
    #     [get_vector_vector_angle(vecs_grids[i], vecs_tracks[i]) for i in range(len(vecs_grids))]
    # )
    # test_ang_arr = test_ang.reshape((50, 50, 50))
    # angles = test_ang_arr[slice_indexes].reshape(my_shape)
    test_ang = get_vector_vector_angle_fast(vecs_grids, vecs_tracks)
    test_ang_arr = test_ang.reshape((50, 50, 50))
    angles = test_ang_arr[slice_indexes].reshape(my_shape)

    test_dot_prod = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    test_dot_prod_arr = test_dot_prod.reshape((50, 50, 50))
    dot_prod = test_dot_prod_arr[slice_indexes].reshape(my_shape)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # hist2D = ax.imshow(
    #     angles.squeeze(), cmap="seismic_r", vmin=-np.pi, vmax=np.pi, origin="lower", label="angle (rad)"
    # )
    # plt.colorbar(hist2D)

    # fig, axs = plot_quiver_slices_from_diffae_table(
    #     merged_feats_df, traj_tracks, flow_field_dict_tracks
    # )
    # fig.suptitle("DiffAE tracked crops")

    # fig, axs = plot_quiver_slices_from_diffae_table(
    #     diffae_grid_crops, traj_grids, flow_field_dict_grids
    # )
    # fig.suptitle("DiffAE grid crops")

    # fig, axs = plot_quiver_slices_from_diffae_table(
    #     diffae_grid_crops, traj_grids, flow_field_dict_grids
    # )
    # fig.suptitle("deviation hist2D")

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the full resolution:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_full_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,  # type: ignore
        ds=1,
        scale=60,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # save_plot_to_path(
    #     fig, out_subdir_traj,
    #     f"{dataset_name}_quiver_slice_comparison_full",
    # )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the standard/default resolution:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_partial_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,  # type: ignore
    )
    ax.scatter(
        traj_grids[-1, 0],
        traj_grids[-1, 1],
        s=250,
        color="cyan",
        marker="*",
        lw=1,
        edgecolor="darkblue",
        zorder=10,
    )
    ax.scatter(
        traj_tracks[-1, 0],
        traj_tracks[-1, 1],
        s=250,
        color="yellow",
        marker="*",
        lw=1,
        edgecolor="darkred",
        zorder=10,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # save_plot_to_path(
    #     fig, out_subdir_traj,
    #     f"{dataset_name}_quiver_slice_comparison_full",
    # )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the angular deviation between the grid and cell-centric crop-based
    # flow field vectors:
    out_path = out_subdir / f"{dataset_name}_vecvec_angles.png"
    grid_vs_track_vec_angle_hist2d(angles, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Plot the dot product between the grid and cell-centric crop-based
    out_path = out_subdir / f"{dataset_name}_vecvec_dot_products.png"
    grid_vs_track_vec_dot_prod_hist2d(dot_prod, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Plot flow fields overlaid on the PC1 vs PC2
    # histograms to get an idea of where the flow
    # fields have the most data to work with
    out_path = out_subdir / f"{dataset_name}_grid_crops_pc1_pc2_hist2d.png"
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=diffae_grid_crops,
        x="pc1",
        y="pc2",
        bins=50,
        # cbar=True,
        # cbar_kws={"label": "Counts"},
        cmap="Blues",
        # cmap="Greys",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
        # color="tab:blue",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out_path = out_subdir / f"{dataset_name}_tracked_crops_pc1_pc2_hist2d.png"
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=merged_feats_df,
        x="pc1",
        y="pc2",
        bins=50,
        # cbar=True,
        # cbar_kws={"label": "Counts"},
        cmap="Reds",
        # cmap="Greys",
        # stat="density",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_tracks, v2_tracks),
        grid=(g1_tracks, g2_tracks),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
        # color="tab:red",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Compare the angles between grid crop PC vectors
    # and the PC vectors of a single track
    merged_feats_df["dpc1"] = merged_feats_df.groupby("crop_index")["pc1"].diff()
    merged_feats_df["dpc2"] = merged_feats_df.groupby("crop_index")["pc2"].diff()
    merged_feats_df["dt"] = merged_feats_df.groupby("crop_index")["time_minutes"].diff()

    # g1_grids.min(), g1_grids.max()  # this is pc1
    # g2_grids.min(), g2_grids.max()  # this is pc2

    get_approx_grid_bin = lambda pc1_pc2_arr: get_approx_point_from_grid(
        # pc1_pc2_arr[0], pc1_pc2_arr[1],
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,  # type: ignore
    )
    # get_approx_grid_bin_from_df = lambda df: df.apply(lambda df_row: get_approx_grid_bin(df_row[["pc1", "pc2"]].values), axis=1)
    get_approx_grid_bin_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_bin(df.to_numpy()), index=df.index
    )

    get_approx_grid_vec = lambda pc1_pc2_arr: get_approx_vec_from_grid(
        # pc1_pc2_arr[0], pc1_pc2_arr[1],
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,  # type: ignore
    )
    # get_approx_grid_vec_from_df = lambda df: df.apply(lambda df_row: get_approx_grid_vec(df_row[["pc1", "pc2"]].values), axis=1)
    get_approx_grid_vec_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_vec(df.to_numpy()), index=df.index
    )

    # this takes FOREVER (actually about 7.5 minutes), could
    # definitely speed it up with some clever vectorization
    # test = []
    # for pc1_pc2 in tqdm(merged_feats_df[["pc1", "pc2"]].values.tolist()):
    #     test.append(get_approx_grid_bin(pc1_pc2))
    # test = np.asarray(test)
    groups = merged_feats_df.groupby("crop_index")
    # df = groups.get_group(name=7135)
    # for nm, df in groups:
    #     break
    # merged_feats_df[["approx_bin_pc1","approx_bin_pc2"]] = np.asarray(merged_feats_df.groupby("crop_index", as_index=False).apply(lambda df: get_approx_grid_bin(df[["pc1", "pc2"]].to_numpy())).values.tolist())
    # merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = np.asarray(merged_feats_df.groupby("crop_index", as_index=False).apply(lambda df: get_approx_grid_vec(df[["pc1", "pc2"]].to_numpy())).values.tolist())

    merged_feats_df[["approx_bin_pc1", "approx_bin_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_bin_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )
    merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_vec_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )

    # the fast version of the angle calculation seems to in disagreement with the slow
    # version, and sometimes even returns np.nan values unexpectedly, so I am using the
    # slow version
    merged_feats_df["track_angle_deviation_rad"] = get_vector_vector_angle_fast(
        merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]].values,
        merged_feats_df[["dpc1", "dpc2"]].values,
    )

    # merged_feats_df["approx_angle_deviation_rad"] = merged_feats_df.apply(
    #     lambda df_row: get_vector_vector_angle(
    #         df_row[["approx_vec_pc1", "approx_vec_pc2"]].values,
    #         df_row[["dpc1", "dpc2"]].values,
    #     ), axis=1)  # type: ignore

    merged_feats_df["track_angular_deviation_deg"] = merged_feats_df[
        "track_angle_deviation_rad"
    ].transform(np.rad2deg)

    merged_feats_df["pc1_pc2_vec_mag"] = np.linalg.norm(
        merged_feats_df[["dpc1", "dpc2"]].values, axis=1
    )

    test = (
        merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"])[
            ["track_angular_deviation_deg", "pc1_pc2_vec_mag"]
        ]
        .agg("mean")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=test, x="track_angular_deviation_deg", binwidth=1, ax=ax)
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Counts")
    fig.savefig(
        out_subdir / f"{dataset_name}_angular_deviation_histogram.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=test, x="track_angular_deviation_deg", y="pc1_pc2_vec_mag", binwidth=(1, None), ax=ax)  # type: ignore
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    # ax.set_ylim(0, 0.2)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Track PC1-PC2\nvector magnitude")
    fig.savefig(
        out_subdir / f"{dataset_name}_angular_deviation_vs_mag_histogram.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if make_integrated_plots:

        groups = merged_feats_df.query("track_duration > 120").groupby(
            ["dataset_name", "position_as_str", "crop_index"]
        )  # type: ignore

        for nm, df in tqdm(groups, desc=dataset_name):
            ds_nm, pos, tid = nm  # type: ignore
            plot_pc_integrated_track(
                dataset_name=str(ds_nm),
                position_name=str(pos),
                track_id=tid,
                df=df,
                v1_grids=v1_grids,
                v2_grids=v2_grids,
                g1_grids=g1_grids,
                g2_grids=g2_grids,
                slice_indexes=slice_indexes,  # type: ignore
                out_subdir=out_subdir,
            )


dataset_name_list = load_dataset_collection_config("pca_reference").datasets

for dataset_name in dataset_name_list:
    process_dataset(dataset_name, make_integrated_plots=True)

# create a test flow field and test set of vectors
# to check that the angular deviation calculation
# works as expected
test_flow_field = one_direction_vector_field_example()
# test_flow_field_vectors = np.column_stack((
#     test_flow_field[0][0].ravel(),
#     test_flow_field[0][1].ravel()
# ))
# test_flow_field_points = np.column_stack((
#     test_flow_field[1][0].ravel(),
#     test_flow_field[1][1].ravel()
# ))

test_vectors = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ]
)

test_points = np.array(
    [
        [-8.0, -4.0],
        [-6.0, -3.0],
        [-4.0, -2.0],
        [-2.0, -1.0],
        [2.0, 1.0],
        [4.0, 2.0],
        [6.0, 3.0],
        [8.0, 4.0],
    ]
)

slice_indexes = np.where(np.ones_like(test_flow_field[0][1]))
test_flow_field_points = get_approx_point_from_grid(
    test_points,
    test_flow_field[1][0],
    test_flow_field[1][1],
    test_flow_field[0][0],
    test_flow_field[0][1],
    slice_indexes,  # type: ignore
)

test_flow_field_vectors = get_approx_vec_from_grid(
    test_vectors,
    test_flow_field[1][0],
    test_flow_field[1][1],
    test_flow_field[0][0],
    test_flow_field[0][1],
    slice_indexes,  # type: ignore
)

test_angular_deviation = get_vector_vector_angle_fast(test_flow_field_vectors, test_vectors)
test_angular_deviation_deg = np.rad2deg(test_angular_deviation)


cmap = sns.color_palette("dark:red", as_cmap=True)
angle_deg_to_color = lambda a: cmap(np.abs(a) / 180.0)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.quiver(
    test_flow_field[1][0],
    test_flow_field[1][1],
    test_flow_field[0][0],
    test_flow_field[0][1],
    scale_units="xy",
    angles="xy",
    scale=1,
    units="width",
    width=0.005,
    alpha=1,
    color="lightgrey",
)
ax.quiver(
    test_flow_field_points[:, 0],
    test_flow_field_points[:, 1],
    test_flow_field_vectors[:, 0],
    test_flow_field_vectors[:, 1],
    scale_units="xy",
    angles="xy",
    scale=1,
    units="width",
    width=0.005,
    alpha=1,
    color="grey",
)
ax.quiver(
    test_points[:, 0],
    test_points[:, 1],
    test_vectors[:, 0],
    test_vectors[:, 1],
    scale_units="xy",
    angles="xy",
    scale=1,
    units="width",
    width=0.005,
    alpha=1,
    color=angle_deg_to_color(test_angular_deviation_deg),
)
ax.set_xlim(-9, 9)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.set_title("Angular deviation from\nflow field test")
fig.savefig(
    get_output_path(Path(__file__).stem) / "get_angular_deviation_deg_test.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# NOTE VERY ROUGH WORK IN PROGRESS BELOW

# from scipy.stats import bootstrap

# rng = np.random.default_rng()
# res = bootstrap(data, np.std, confidence_level=0.9, rng=rng)
