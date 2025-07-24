from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from src.endo_pipeline.library.analyze.diffae_features.track_integration import (
    get_coarse_grained_trajectory_heatmap_data,
)
from src.endo_pipeline.library.analyze.numerics.binning import get_bins
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)


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


def plot_quiver_slices_from_diffae_table(
    diffae_df: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    plot_trajectory: bool = True,
    plot_fixed_points: bool = True,
) -> tuple[Figure, np.ndarray]:

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
            ax.plot(traj_grids[:, 0], traj_grids[:, j + 1], lw=2, color="navy", zorder=1)
        if plot_fixed_points:
            ax.scatter(traj_grids[-1, 0], traj_grids[-1, j + 1], s=50, color="black", zorder=2)

    return fig, axs


def plot_measured_feat_pcs(
    measured_feat_df: pd.DataFrame,
    meas_feat_col: str,
    pc_cols_for_xaxis: list[str],
    pc_cols_for_yaxis: list[str],
    fig: Figure | None = None,
    axs: np.ndarray | None = None,
    track_id: Literal["mean"] | int | None = "mean",
    hue_norm: tuple[float, float] | None = None,
    zorder: int = 0,
    alpha: float = 1.0,
) -> tuple[Figure, np.ndarray]:

    pc_cols = [pc for pc in set((*pc_cols_for_xaxis, *pc_cols_for_yaxis))]

    assert len(pc_cols_for_xaxis) == len(
        pc_cols_for_yaxis
    ), "x and y axis must have the same number of PCs"
    if axs is not None:
        assert len(pc_cols_for_xaxis) == axs.size, "PCs must be provided for each ax in axs"
    assert all(col in measured_feat_df.columns for col in pc_cols), (
        f"One or more PCs in {pc_cols} not found in measured feature dataframe columns."
        "Check spelling and case?"
    )

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
                (
                    "track_ids must be 'mean', an integer, or None."
                    f"Got {track_id} (type: {type(track_id)}) instead."
                )
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
            (
                "track_ids must be 'mean', an integer, or None."
                f"Got {track_id_to_plot} (type: {type(track_id_to_plot)}) instead."
            )
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


def overlay_trajectory_heatmap_on_flowfield(
    out_dir: Path,
    dataset_name: str,
    diffae_grid_crops: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    df_all_positions: pd.DataFrame,
    num_bins: list[int] = [150, 150, 150],
):
    """
    Overlay a coarse-grained trajectory heatmap on the flow field.

    Parameters
    ----------
    out_dir
        Directory to save the plot to.
    dataset_name
        Name of the dataset to use for the plot.
    diffae_grid_crops
        DataFrame containing the diffae grid crops.
    traj_grids
        Numpy array containing the trajectory grids.
    flow_field_dict_grids
        Dictionary containing the flow field data for the grids.
    df_all_positions
        DataFrame containing all positions and tracks.
    num_bins
        Number of bins to use for the heatmap in each dimension.
    """
    # plot flow field
    fig, axs = plot_quiver_slices_from_diffae_table(
        diffae_grid_crops,
        traj_grids,
        flow_field_dict_grids,
        plot_trajectory=False,
    )

    bounds = get_grid_bounds(flow_field_dict_grids)
    bins, _ = get_bins(num_bins, bin_limits=bounds)

    project_axis = [2, 1]  # this is axis for projecting binned data for each plot
    plot_dim = [1, 2]  # this is the PC dimension plotted on the y-axis against PC1

    bin_data, bin_counts = get_coarse_grained_trajectory_heatmap_data(
        df_all_positions=df_all_positions,
        bounds=bounds,
        num_bins=num_bins,
    )

    for j, ax in enumerate(axs):
        plot_data = np.divide(
            bin_data, bin_counts, out=np.zeros_like(bin_data), where=bin_counts != 0
        )
        plot_data = np.nanmean(plot_data, axis=project_axis[j]).T
        plot_data = plot_data / np.nanmax(plot_data)
        ax.imshow(
            plot_data,
            extent=(
                bins[0][0],
                bins[0][-1],
                bins[plot_dim[j]][0],
                bins[plot_dim[j]][-1],
            ),
            cmap="viridis",
            aspect="auto",
            origin="lower",
            zorder=-1,
            vmin=0,
        )
    cbar = plt.colorbar(ax.images[-1], ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Normalized trajectory time", rotation=270, labelpad=15)
    # plt.tight_layout()
    fig.savefig(out_dir / f"{dataset_name}_trajectory_heatmap.png", dpi=300)
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
        for tid in tqdm(
            track_ids, total=len(track_ids), desc=f"Plotting tracks at {pos}", leave=False
        ):
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

    # plot trajectory heatmap
    out_subdir_heatmap = out_subdir / "trajectory_heatmap"
    out_subdir_heatmap.mkdir(parents=True, exist_ok=True)

    overlay_trajectory_heatmap_on_flowfield(
        out_dir=out_subdir_heatmap,
        dataset_name=dataset_name,
        diffae_grid_crops=diffae_grid_crops,
        traj_grids=traj_grids,
        flow_field_dict_grids=flow_field_dict_grids,
        df_all_positions=df_all_positions,
    )
