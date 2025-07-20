from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from cellsmap.util.manifest_io import get_diffae_manifest, get_track_diffae_manifest
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs.dataset_io import (
    get_live_segmentation_features_manifest,
    get_reference_datasets,
    ipython_cli_flexecute,
)
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_features.track_integration import (
    get_diffae_feats_liveseg_feats_merged_table,
)
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)


def get_traj_and_flowfield(
    df: pd.DataFrame,
    bounds: Pipeline,
) -> tuple[np.ndarray, dict]:

    # load default config, get kernel params
    config = load_dynamics_config("default")
    kernel_params = config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
    dt = config["dt"]

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


def get_valid_slice_indexes(
    df: pd.DataFrame,
    traj: np.ndarray,
    flow_field_dict: dict,
) -> tuple[
    tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
]:
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
    fig, axs = plot_quiver_slices(
        flow_field_dict_grids, (zvalids_grids, yvalids_grids), color="#08b4bc"
    )
    [ax.set_zorder(0) for ax in axs]
    axs = set_slice_plot_bounds_and_labels(axs, bounds)

    # plot the trajectories
    for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
        if plot_trajectory:
            ax.plot(traj_grids[:, 0], traj_grids[:, j + 1], lw=2, color="navy", zorder=1)
        if plot_fixed_points:
            ax.scatter(traj_grids[-1, 0], traj_grids[-1, j + 1], s=50, color="black", zorder=2)

    return fig, axs


def plot_measured_feat_pcs(
    measured_feat_df: pd.DataFrame,
    meas_feat_col: str,
    pc_cols_for_xaxis: list[str],
    pc_cols_for_yaxis: list[str],
    fig: plt.Figure | None = None,
    axs: np.ndarray | None = None,
    track_id: Literal["mean"] | int | None = "mean",
    hue_norm: tuple[float, float] | None = None,
    zorder: int = 0,
    alpha: float = 1.0,
) -> tuple[plt.Figure, np.ndarray]:

    pc_cols = [pc for pc in set((*pc_cols_for_xaxis, *pc_cols_for_yaxis))]

    assert len(pc_cols_for_xaxis) == len(
        pc_cols_for_yaxis
    ), "x and y axis must have the same number of PCs"
    if axs is not None:
        assert len(pc_cols_for_xaxis) == axs.size, "PCs must be provided for each ax in axs"
    assert all(
        col in measured_feat_df.columns for col in pc_cols
    ), f"One or more PCs in {pc_cols} not found in measured feature dataframe columns. Check spelling and case?"

    if axs is None:
        fig, axs = plt.subplots(figsize=(14, 5), ncols=2)

    # plot the measured features
    for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
        if track_id == "mean":
            measured_feat_df = (
                measured_feat_df.groupby("frame_number")
                .mean(numeric_only=True)[pc_cols + [meas_feat_col]]
                .reset_index()
            )
        elif isinstance(track_id, int):
            measured_feat_df = measured_feat_df.query("track_id == @track_id")
        elif track_id is None:
            pass  # do not subset or aggregate the data in any way
        else:
            raise ValueError(
                f"track_ids must be 'mean', an integer, or None. Got {track_id} (type: {type(track_id)}) instead."
            )

        if track_id is not None:
            ax.plot(
                measured_feat_df[pc_cols_for_xaxis[j]],
                measured_feat_df[pc_cols_for_yaxis[j]],
                lw=1,
                color="lightgrey",
                alpha=alpha,
                zorder=max(0, zorder),
            )
        sns.scatterplot(
            data=measured_feat_df,
            x=pc_cols_for_xaxis[j],
            y=pc_cols_for_yaxis[j],
            hue=meas_feat_col,
            hue_norm=hue_norm,
            palette="flare",
            linewidth=0,
            marker=".",
            s=50,
            alpha=alpha,
            ax=ax,
            zorder=zorder + 1,
        )

    return fig, axs  # type: ignore[return-value]


def plot_measured_feat_overlay_on_flowfield(
    out_dir: Path,
    dataset_name: str,
    diffae_grid_crops: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    diffae_measured_feat_df: pd.DataFrame,
    meas_feat_col_name_for_color_coding: str,
    track_id_to_plot: Literal["mean"] | int | None = "mean",
    hue_norm: tuple[float, float] | None = None,
    alpha: float = 0.7,
    show_plot: bool = False,
) -> None:
    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    fig, axs = plot_measured_feat_pcs(
        measured_feat_df=diffae_measured_feat_df,
        meas_feat_col=meas_feat_col_name_for_color_coding,
        pc_cols_for_xaxis=["pc1", "pc1"],
        pc_cols_for_yaxis=["pc2", "pc3"],
        track_id=track_id_to_plot,
        fig=fig,
        axs=axs,
        hue_norm=hue_norm,
        zorder=5,
        alpha=alpha,
    )
    plt.tight_layout()
    if track_id_to_plot == "mean":
        data_subset = "_timeAvgTracks"
    elif isinstance(track_id_to_plot, int):
        data_subset = f"_tid{track_id_to_plot}"
    elif track_id_to_plot is None:
        data_subset = ""
    else:
        raise ValueError(
            f"track_ids must be 'mean', an integer, or None. Got {track_id_to_plot} (type: {type(track_id_to_plot)}) instead."
        )
    fig.savefig(
        out_dir / f"{dataset_name}{data_subset}_{meas_feat_col_name_for_color_coding}Hue.png",
        dpi=300,
    )
    if not show_plot:
        plt.close(fig)


def plot_new_traj_overlay_on_grid_traj_and_flowfield(
    out_dir: Path,
    dataset_name: str,
    diffae_grid_crops: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    traj_tracks: np.ndarray,
) -> None:
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
    fig.savefig(out_dir / f"{dataset_name}_trajectory_grids_vs_tracks.png", dpi=300)
    plt.close(fig)


def make_all_plots(
    out_dir: Path,
    dataset_name: str,
    diffae_grid_crops: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    df_all_positions: pd.DataFrame,
    traj_tracks: np.ndarray,
) -> None:

    # create a subdirectory to save the plots to
    out_subdir = out_dir / dataset_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    # create subdirectory to save individual track overlay
    # examples to
    out_subdir_indiv = out_subdir / "individual_track_overlays"
    out_subdir_indiv.mkdir(parents=True, exist_ok=True)

    # plot just the flow field
    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops, traj_grids, flow_field_dict_grids
    )
    plt.tight_layout()
    fig.savefig(out_subdir / f"{dataset_name}_flow_field.png", dpi=300)
    plt.close(fig)

    # plot the flow field and the trajectories
    plot_new_traj_overlay_on_grid_traj_and_flowfield(
        out_subdir,
        dataset_name,
        diffae_grid_crops,
        traj_grids,
        flow_field_dict_grids,
        traj_tracks,
    )

    measured_feats_to_plot = ["time_hours", "alignment_deg_rel_to_flow", "eccentricity"]
    for measured_feature in measured_feats_to_plot:
        plot_measured_feat_overlay_on_flowfield(
            out_subdir,
            dataset_name,
            diffae_grid_crops,
            traj_grids,
            flow_field_dict_grids,
            diffae_measured_feat_df=df_all_positions,
            meas_feat_col_name_for_color_coding=measured_feature,
            track_id_to_plot="mean",
            alpha=0.8,
            show_plot=False,
        )

    # plot single track examples
    for pos, df_one_position in df_all_positions.groupby("position_as_str"):
        out_subdir_indiv_pos = out_subdir_indiv / str(pos)
        out_subdir_indiv_pos.mkdir(parents=True, exist_ok=True)

        track_ids = sorted(df_one_position["track_id"].unique().tolist())
        # only overlay every 10th track id if there are a lot
        # of tracks to save time + space
        track_ids = track_ids[::10] if len(track_ids[::10]) > 10 else track_ids
        for tid in tqdm(track_ids, total=len(track_ids), desc=f"Plotting tracks at {pos}"):
            # make the plots
            plot_measured_feat_overlay_on_flowfield(
                out_subdir_indiv_pos,
                dataset_name,
                diffae_grid_crops,
                traj_grids,
                flow_field_dict_grids,
                diffae_measured_feat_df=df_one_position,
                meas_feat_col_name_for_color_coding="alignment_deg_rel_to_flow",
                track_id_to_plot=tid,
                hue_norm=(0, 90),
                alpha=0.8,
                show_plot=False,
            )


def main() -> None:
    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name_list = get_reference_datasets()

    for dataset_name in dataset_name_list:
        # create subdirectory to save track-based trajectories to
        out_subdir_traj = out_dir / "trajectories_track_based"
        out_subdir_traj.mkdir(parents=True, exist_ok=True)

        df_all_positions = get_diffae_feats_liveseg_feats_merged_table(dataset_name)
        if df_all_positions is None:
            print(f"Dataset {dataset_name} is missing one or more data tables. Skipping...")
            continue

        print("cleaning up merged table...")
        df_all_positions = df_all_positions.query("valid_points >= 120")
        df_all_positions.dropna(axis="index", how="any", subset="is_unique", inplace=True)

        # fit the PCA (uses the reference datasets)
        pca = fit_pca()

        # read in the grid crop-based diffae features
        diffae_grid_crops = get_manifest_for_dynamics_workflows(dataset_name, pca)

        # add the PC columns to the track-based DiffAE table
        # (the grid-based DiffAE table already has them, but
        # but I believe that the columns are named "feat_0",
        # "feat_1", etc. when they should be named "pc1",
        # "pc2", etc.)
        df_all_positions = project_manifest_to_pcs(df_all_positions, pca)

        # use the full set of datasets to be analyzed for the bounds
        bounds = ddff.set_3d_bounds_from_data(dataset_name_list, pca)

        print("getting trajectory and flow field for grid-based crops...")
        traj_grids, flow_field_dict_grids = get_traj_and_flowfield(diffae_grid_crops, bounds)

        print("getting trajectory and flow field for tracks-based crops...")
        traj_tracks, _ = get_traj_and_flowfield(df_all_positions, bounds)
        # save the trajectory data from the track-based crops
        np.save(out_subdir_traj / f"{dataset_name}_traj_tracks.npy", traj_tracks)

        # save plots of the track-based crop trajectories and PCs overlaid
        # on the flow field and trajectories from the grid-based crops
        make_all_plots(
            out_dir,
            dataset_name,
            diffae_grid_crops,
            traj_grids,
            flow_field_dict_grids,
            df_all_positions,
            traj_tracks,
        )


if __name__ == "__main__":
    ipython_cli_flexecute(main)
