from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from cellsmap.analyses.track_data_plots import (
    calculate_derived_data_dynamics_independent,
    filter_seg_feature_table,
    merge_segprops_and_track_data,
)

# from cellsmap.analyses.utils.viz import flow_field_viz as ffv
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.analyses.utils.viz.flow_field_viz import (
    get_slice_indexes,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)
from cellsmap.util.dataset_io import (
    get_measurement_data_raws,
    get_reference_datasets,
    get_tracking_data_raws,
    load_config,
)
from cellsmap.util.manifest_io import (
    get_diffae_manifest,
    get_feature_cols,
    get_track_diffae_manifest,
)
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util.manifest_preprocessing.manifest_pca import fit_pca
from cellsmap.util.set_output import get_output_path


def get_traj_and_flowfield(
    df: pd.DataFrame,
    num_pcs: int = 3,
) -> tuple[np.ndarray, dict]:
    assert num_pcs == 3, "Only 3D flow fields are supported"

    pc_cols = [f"pc{i+1}" for i in range(num_pcs)]
    kernel_params = {"bandwidth": 0.09, "kernel": "gaussian"}
    # get state space bounds and grid resolution for estimating flow field
    excluded_fraction = 0.00
    bounds = ddff.set_3d_bounds_from_data(
        diffae_grid_crops.pc1,
        diffae_grid_crops.pc2,
        diffae_grid_crops.pc3,
        excluded_fraction=excluded_fraction,
    )
    # time stepping for the flow field
    dt = 5
    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    t_span = [0.0, 2880.0]
    grid_spacing = 0.05
    Nbins = [
        int((bounds[i][1] - bounds[i][0]) / grid_spacing) + 1 for i in range(num_pcs)
    ]
    bins, centers = rh.get_bins(Nbins, bin_limits=bounds)

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = rh.get_traj_and_diff(df, pc_cols)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = rh.get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = ddff.compute_extrapolated_vector_field(
        drift_km, centers, interpolator="nearest"
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by the mean of the data at T=0

    # get initial conditions for the ODE solver from data
    inits_mean = df.groupby("frame_number").mean(numeric_only=True)[pc_cols].values[0]

    # solve IVP, get back trajectory
    traj = ddff.solve_ddff_ode(flow_field_dict, inits_mean, t_span)

    return traj, flow_field_dict


def get_valid_slice_indexes(
    df: pd.DataFrame,
    traj: np.ndarray,
    flow_field_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    # get grid and grid spacing
    xgrid, ygrid, zgrid = flow_field_dict["grid"]

    # if not specified, use mean of data at last time point
    # if data are not provided, use pc2 = pc3 = 0
    # plot 2D slices at PC2 and PC3 values given by
    # the last point of the trajectory
    pc_vals = (traj[-1, 2], traj[-1, 1])

    if pc_vals is None:
        if df is None:
            pc3_val = 0
            pc2_val = 0
        else:
            # get mean at all time points over crops
            mean_over_crops = df.groupby("frame_number").mean(numeric_only=True)
            # get last time point
            mean_over_crops = mean_over_crops.iloc[-1]
            pc3_val = mean_over_crops["pc3"].mean()
            pc2_val = mean_over_crops["pc2"].mean()
    # if specified, unpack
    else:
        pc3_val = pc_vals[0]
        pc2_val = pc_vals[1]

    # get z-slice closest to PC3 = pc3_val
    zvalids = get_slice_indexes(zgrid, pc3_val)
    # get y-slice closest to PC2 = pc2_val
    yvalids = get_slice_indexes(ygrid, pc2_val)
    return yvalids, zvalids


def get_grid_bounds(flow_field_dict: dict) -> list:
    # get bounds of the grid
    xmin, xmax = (
        flow_field_dict["grid"][0][0, 0, 0],
        flow_field_dict["grid"][0][-1, 0, 0],
    )
    ymin, ymax = (
        flow_field_dict["grid"][1][0, 0, 0],
        flow_field_dict["grid"][1][0, -1, 0],
    )
    zmin, zmax = (
        flow_field_dict["grid"][2][0, 0, 0],
        flow_field_dict["grid"][2][0, 0, -1],
    )
    bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    return bounds


def get_and_process_diffae_data(dataset_name: str) -> pd.DataFrame:
    # read in the grid crop-based diffae features
    diffae_grid_crops = get_diffae_manifest(dataset_name)
    diffae_grid_crops = diffae_preproc.add_crop_index(diffae_grid_crops)
    diffae_grid_crops = diffae_preproc.add_description_column(
        diffae_grid_crops, dataset_name, simple=True
    )  # add description column (e.g., 48hr_High)

    # in Erin's code in workflows/flow_field3d/preprocessing.py
    # adding the dataset name to the crop index was required to
    # make the crop index unique if multiple datasets were used
    diffae_grid_crops["crop_index"] = (
        diffae_grid_crops["dataset"] + "_" + diffae_grid_crops["crop_index"].astype(str)
    )

    return diffae_grid_crops


def add_pc_cols(df: pd.DataFrame, pca_object: Pipeline) -> pd.DataFrame:
    # get column names for features 1-8
    feat_cols = get_feature_cols(df)

    # get the PCs
    x_proj = pca_object.transform(df[feat_cols].values)

    # add PCs to dataframe
    pc_cols: list = []
    for pc in range(num_pcs):
        pc_col_name = f"pc{pc+1}"
        pc_cols.append(pc_col_name)
        df[pc_col_name] = x_proj[:, pc]
    return df


def plot_quiver_slices_from_diffae_table(
    diffae_df: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    plot_trajectory: bool = True,
    plot_fixed_points: bool = True,
) -> tuple[plt.Figure, np.ndarray]:

    # get valid y and z slice indices
    yvalids_grids, zvalids_grids = get_valid_slice_indexes(
        diffae_df, traj_grids, flow_field_dict_grids
    )

    # get limits of grid from the grid crops flow fields
    bounds = get_grid_bounds(flow_field_dict_grids)

    # plot the flow field
    fig, axs = plot_quiver_slices(flow_field_dict_grids, (zvalids_grids, yvalids_grids))
    [ax.set_zorder(0) for ax in axs]
    axs = set_slice_plot_bounds_and_labels(axs, bounds)

    # plot the trajectories
    for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
        if plot_trajectory:
            ax.plot(traj_grids[:, 0], traj_grids[:, j + 1], lw=2, color="navy")
        if plot_fixed_points:
            ax.scatter(
                traj_grids[-1, 0], traj_grids[-1, j + 1], s=50, color="black", zorder=9
            )

    return fig, axs


def plot_measured_feat_overlay_on_flowfield(
    measured_feat_df: pd.DataFrame,
    meas_feat_col: str,
    pc_cols_for_xaxis: tuple[str, ...],
    pc_cols_for_yaxis: tuple[str, ...],
    axs: np.ndarray,
    track_ids: str | List[int] = "mean",
    zorder: int = 0,
) -> tuple[plt.Figure, np.ndarray]:

    pc_cols = [pc for pc in set((*pc_cols_for_xaxis, *pc_cols_for_yaxis))]

    assert len(pc_cols_for_xaxis) == len(
        pc_cols_for_yaxis
    ), "x and y axis must have the same number of PCs"
    assert len(pc_cols_for_xaxis) == len(axs), "PCs must be provided for each ax in axs"
    assert all(
        col in measured_feat_df.columns for col in pc_cols
    ), f"One or more PCs in {pc_cols} not found in measured feature dataframe columns. Check spelling and case?"

    # plot the measured features
    for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
        if track_ids == "mean":
            measured_feat_df = (
                measured_feat_df.groupby("frame_number")
                .mean(numeric_only=True)[pc_cols + [meas_feat_col]]
                .reset_index()
            )
        elif track_ids == "all":
            pass
        elif isinstance(track_ids, list):
            measured_feat_df = measured_feat_df.query("track_id in @track_ids")
        else:
            raise ValueError(
                f"track_ids must be 'mean', 'all', str, or list. Got {track_ids} (type: {type(track_ids)}) instead."
            )

        # sns.lineplot(
        #     data=measured_feat_df,
        #     x=pc_cols_for_xaxis[j],
        #     y=pc_cols_for_yaxis[j],
        #     color='lightgrey',
        #     lw=1, ax=ax, zorder=max(0, zorder-1)
        #     )
        ax.plot(
            measured_feat_df[pc_cols_for_xaxis[j]],
            measured_feat_df[pc_cols_for_yaxis[j]],
            lw=1,
            color="lightgrey",
            zorder=max(0, zorder - 1),
        )
        sns.scatterplot(
            data=measured_feat_df,
            x=pc_cols_for_xaxis[j],
            y=pc_cols_for_yaxis[j],
            hue=meas_feat_col,
            palette="flare",
            linewidth=0,
            marker=".",
            s=50,
            ax=ax,
            zorder=zorder,
        )

    return fig, axs


out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
out_dir.mkdir(parents=True, exist_ok=True)
data_config = load_config("data")
dataset_name_list = get_reference_datasets()

for dataset_name in dataset_name_list:
    # create a subdirectory to save the plots to
    out_subdir = out_dir / dataset_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    # read in the grid crop-based diffae features
    diffae_grid_crops = get_and_process_diffae_data(dataset_name)

    # read in the segmentation-based diffae features
    diffae_tracking = get_track_diffae_manifest(dataset_name)
    diffae_tracking["is_unique"] = diffae_tracking.groupby(
        ["dataset", "position", "frame_number", "track_id"]
    )["frame_number"].transform(lambda t: t.nunique() == t.size)
    diffae_tracking = diffae_tracking[diffae_tracking["is_unique"]]

    # give the crop_index column the same value as the track_ids
    diffae_tracking["crop_index"] = diffae_tracking["track_id"]
    diffae_tracking = diffae_preproc.add_description_column(
        diffae_tracking, dataset_name, simple=True
    )  # add description column (e.g., 48hr_High)
    diffae_tracking["track_id"] = diffae_tracking["track_id"].astype(int)

    # load the tracking data of the measured features and merge them
    seg_props_df = get_measurement_data_raws(
        [dataset_name], kind="segmentation_properties", as_dask=False
    )
    tracking_df = get_tracking_data_raws([dataset_name], position=None, as_dask=False)
    big_table = merge_segprops_and_track_data(seg_props_df, tracking_df)
    del seg_props_df, tracking_df  # remove unnecessary dataframes to save memory
    diffae_tracking.rename(columns={"position": "position_as_str"}, inplace=True)
    big_table["position_as_str"] = big_table["position"].transform(
        lambda x: "P" + str(x)
    )
    big_table["track_id"] = big_table["track_id"].astype(int)

    big_table = pd.merge(
        left=big_table,
        right=diffae_tracking,
        how="left",
        left_on=["dataset_name", "position_as_str", "image_index", "track_id"],
        right_on=["dataset", "position_as_str", "frame_number", "track_id"],
        validate="one_to_one",
    )
    big_table.dropna(axis=0, how="any", subset="is_unique", inplace=True)
    big_table = calculate_derived_data_dynamics_independent(big_table)
    big_table = filter_seg_feature_table(
        big_table,
        out_dir=None,
        min_num_points_per_track=120,
    )

    # fit the PCA (uses the reference datasets)
    num_pcs = 3
    pc_cols = [f"pc{i+1}" for i in range(num_pcs)]
    pca = fit_pca(num_pcs=num_pcs)  # (only working with top 3 PCs)

    # get PCs from the tracking and grid crop dataframes
    diffae_grid_crops = add_pc_cols(diffae_grid_crops, pca)
    big_table = add_pc_cols(big_table, pca)

    # get the trajectories and flow fields for the grip crops
    # and the track-centered crops
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        diffae_grid_crops, num_pcs
    )
    traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(big_table, num_pcs)

    # plot the flow field and the trajectories
    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
        ax.plot(traj_tracks[:, 0], traj_tracks[:, j + 1], lw=2, color="crimson")
        ax.scatter(
            traj_tracks[-1, 0],
            traj_tracks[-1, j + 1],
            s=50,
            color="black",
            marker="*",
            zorder=10,
        )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_trajectory_grids_vs_tracks.png", dpi=300)

    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_overlay_on_flowfield(
        measured_feat_df=big_table,
        meas_feat_col="time_hours",
        pc_cols_for_xaxis=("pc1", "pc1"),
        pc_cols_for_yaxis=("pc2", "pc3"),
        track_ids="mean",
        axs=axs,
        zorder=1,
    )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_timeAvgTracks_timeHue.png", dpi=300)

    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_overlay_on_flowfield(
        measured_feat_df=big_table,
        meas_feat_col="alignment_deg_rel_to_flow",
        pc_cols_for_xaxis=("pc1", "pc1"),
        pc_cols_for_yaxis=("pc2", "pc3"),
        track_ids="mean",
        axs=axs,
        zorder=1,
    )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_timeAvgTracks_alignmentHue.png", dpi=300)

    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_overlay_on_flowfield(
        measured_feat_df=big_table,
        meas_feat_col="eccentricity",
        pc_cols_for_xaxis=("pc1", "pc1"),
        pc_cols_for_yaxis=("pc2", "pc3"),
        track_ids="mean",
        axs=axs,
        zorder=1,
    )
    plt.tight_layout()
    fig.savefig(
        out_subdir / f"{dataset_name}_timeAvgTracks_eccentricityHue.png", dpi=300
    )

    # plot a single track examples
    # get the first track id
    track_ids = big_table["track_id"].unique().tolist()[:1]

    # make the plots
    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_overlay_on_flowfield(
        measured_feat_df=big_table,
        meas_feat_col="alignment_deg_rel_to_flow",
        pc_cols_for_xaxis=("pc1", "pc1"),
        pc_cols_for_yaxis=("pc2", "pc3"),
        track_ids=track_ids,
        axs=axs,
        zorder=1,
    )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_singleTrack_alignmentHue.png", dpi=300)

    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_overlay_on_flowfield(
        measured_feat_df=big_table,
        meas_feat_col="time_hours",
        pc_cols_for_xaxis=("pc1", "pc1"),
        pc_cols_for_yaxis=("pc2", "pc3"),
        track_ids=track_ids,
        axs=axs,
        zorder=1,
    )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_singleTrack_timeHue.png", dpi=300)

    # break
