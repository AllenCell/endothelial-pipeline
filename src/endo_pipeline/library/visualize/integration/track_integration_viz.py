from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.integration.track_integration import (
    get_coarse_grained_trajectory_heatmap_data,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_one_slice_quiver,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)
from endo_pipeline.settings import TIMEPOINT_COLUMN_NAME


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
            mean_over_crops = df.groupby(TIMEPOINT_COLUMN_NAME).mean(numeric_only=True)
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
                measured_feat_df.groupby(ColumnName.TIMEPOINT)
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
    save_plot_to_path(
        figure=fig,
        output_path=out_dir,
        figure_name=f"{dataset_name}{data_subset}_{meas_feat_col_name_for_color_coding}Hue",
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
    save_plot_to_path(
        figure=fig, output_path=out_dir, figure_name=f"{dataset_name}_trajectory_grids_vs_tracks"
    )
    plt.close(fig)


def overlay_trajectory_heatmap_on_flowfield(
    out_dir: Path,
    dataset_name: str,
    diffae_grid_crops: pd.DataFrame,
    traj_grids: np.ndarray,
    flow_field_dict_grids: dict,
    df_all_positions: pd.DataFrame,
    num_bins: list[int] = [150, 150, 150],
) -> None:
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
    save_plot_to_path(
        figure=fig, output_path=out_dir, figure_name=f"{dataset_name}_trajectory_heatmap"
    )
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
    save_plot_to_path(figure=fig, output_path=out_subdir, figure_name=f"{dataset_name}_flow_field")
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


def plot_grid_vs_tracks_flow_field(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    g1_tracks: np.ndarray,
    g2_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    ds: int = 3,
    scale: int = 30,
) -> tuple[Figure, plt.Axes]:
    """
    This function is basically a wrapper around the
    `plot_one_slice_quiver` function that plots the
    flow field.
    Plots the vectors "v" starting from the points "g" of a flow field with
    vector scaling "scale" and grid spacing "ds" for both grid-based and
    track-based crops.
    each "v" is a meshgrid of one of the components of the vectors, and each
    "g" is a meshgrid of the corresponding points in the flow field from
    which the vectors start.
    The meshgrids are 3D arrays as they are flow fields derived from the
    first 3 PCs of the PCA of the DiffAE features.
    slice_indexes determines which slices along these 3D arrays to get the
    vectors and grid points for plotting.
    The arrow spacing and scale of the vectors can be adjusted with the "ds"
    and "scale" parameters, respectively.

    Parameters
    ----------
    v1_grids : np.ndarray
        The first component of the vectors of the flow field for the grid-based crops.
    v2_grids : np.ndarray
        The second component of the vectors of the flow field for the grid-based crops.
    g1_grids : np.ndarray
        The first component of the points for the grid-based crops.
    g2_grids : np.ndarray
        The second component of the points for the grid-based crops.
    v1_tracks : np.ndarray
        The first component of the vectors of the flow field for the track-based crops.
    v2_tracks : np.ndarray
        The second component of the vectors of the flow field for the track-based crops.
    g1_tracks : np.ndarray
        The first component of the points for the track-based crops.
    g2_tracks : np.ndarray
        The second component of the points for the track-based crops.
    slice_indexes : tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...]
        The slice indexes to pull the (v1, v2) and (g1, g2) values from
        for the flow field.
    ds : int, optional
        The grid spacing for the flow field vectors (default is 3).
    scale : int, optional
        The scaling factor for the flow field vectors (default is 30).

    Returns
    -------
    fig : Figure
        The figure containing the flow field plot.
    ax : plt.Axes
        The axes of the flow field plot.

    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ds=ds,
        scale=scale,
        ax=ax,
        color="blue",
    )
    plot_one_slice_quiver(
        velocities=(v1_tracks, v2_tracks),
        grid=(g1_tracks, g2_tracks),
        slice_indexes=slice_indexes,
        ds=ds,
        scale=scale,
        ax=ax,
        color="red",
    )
    custom_lines = [
        Line2D([0], [0], color="red", lw=2, label="seg-based DiffAE features"),
        Line2D([0], [0], color="blue", lw=2, label="grid-based DiffAE features"),
    ]
    ax.legend(custom_lines, [str(x.get_label()) for x in custom_lines], loc="upper right")
    return fig, ax


def grid_vs_track_vec_angle_hist2d(
    angles: np.ndarray,
    out_dir: Path,
    filename: str,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Plot a 2D histogram of the angular deviation between
    the grid-based and track-based DiffAE features.
    """

    fig, ax_hist = plt.subplots(figsize=(6, 6))
    ax_hist.set_title("Grid vs. cell-centric crop angular deviation", pad=20)
    hist2D = ax_hist.imshow(
        np.rad2deg(angles.squeeze()).T,
        cmap="RdBu_r",
        vmin=0,
        vmax=180,
        extent=extent,
        origin="lower",
        label="angle (rad)",
    )
    divider = make_axes_locatable(ax_hist)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(hist2D, cax=ax_cb)
    ax_cb.set_yticks(np.arange(0, 181, 30))  # set ticks for angle in degrees
    ax_hist.set_xlabel("PC1")
    ax_hist.set_ylabel("PC2")
    ax_cb.set_ylabel("Angle (degrees)", rotation=270, verticalalignment="bottom")
    plt.tight_layout()
    save_plot_to_path(figure=fig, output_path=out_dir, figure_name=filename)


def grid_vs_track_vec_dot_prod_hist2d(
    dot_prod: np.ndarray,
    out_dir: Path,
    filename: str,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Plot a 2D histogram of the dot product between
    the grid-based and track-based DiffAE features.
    """
    vmin = -1 * abs(dot_prod).max()
    vmax = 1 * abs(dot_prod).max()
    cmap_norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax_hist = plt.subplots(figsize=(6, 6))
    ax_hist.set_title("Grid vs. cell-centric vector dot products", pad=20)
    hist2D = ax_hist.imshow(
        dot_prod.squeeze().T,
        cmap="RdBu",
        norm=cmap_norm,
        extent=extent,
        origin="lower",
        label="dot product",
    )
    divider = make_axes_locatable(ax_hist)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(hist2D, cax=ax_cb)
    ax_cb.set_yscale("linear")
    ax_hist.set_xlabel("PC1")
    ax_hist.set_ylabel("PC2")
    ax_cb.set_ylabel("Dot product", rotation=270, verticalalignment="bottom")
    plt.tight_layout()
    save_plot_to_path(figure=fig, output_path=out_dir, figure_name=filename)


def plot_pc_integrated_track_as_arrows(
    dataset_name: str,
    position_name: str,
    track_id: int,
    df: pd.DataFrame,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
    out_subdir: Path,
    hue_min: float,
    hue_max: float,
    hue_center: float = 0.0,
    cmap_name: str = "RdBu_r",
    hued_feat_name: str = "dot_product_grid_vs_cell",
    track_alpha: float = 0.7,
) -> None:

    out_subdir_integrated_tracks = out_subdir / "integrated_tracks"
    out_subdir_integrated_tracks.mkdir(parents=True, exist_ok=True)

    out_subdir_integrated_tracks_hued = out_subdir / "integrated_tracks_hued"
    out_subdir_integrated_tracks_hued.mkdir(parents=True, exist_ok=True)

    # plot a single track integrated into the flow field
    # shown as dots connected by arrows to give an idea
    # of the direction of motion of the cell through the
    # flow field
    # NOTE: the Figure(figsize), fig.subplots(...) pattern below is
    # used to avoid memory leak issues in interactive sessions
    fig = Figure(figsize=(4, 4))
    ax = fig.subplots(nrows=1, ncols=1)
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
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
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir_integrated_tracks,
        figure_name=f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field",
    )
    fig.clf()
    plt.close(fig)

    cmap_norm = TwoSlopeNorm(vmin=hue_min, vcenter=hue_center, vmax=hue_max)
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    feat_to_color = lambda a: cmap(cmap_norm(a))

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots(nrows=1, ncols=1)
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="grey",
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
        alpha=track_alpha,
        color=feat_to_color(df[hued_feat_name].iloc[1:]),
    )
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ColorbarBase(
        ax_cb,
        cmap=cmap,
        norm=cmap_norm,
        ticks=np.linspace(hue_min, hue_max, 7).tolist(),
        orientation="vertical",
    )
    cbar.set_label(hued_feat_name, rotation=270, labelpad=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_name} {position_name} track {track_id}\nintegrated flow field")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir_integrated_tracks_hued,
        figure_name=f"{dataset_name}_{position_name}_track_{track_id}_integrated_flow_field_hued",
    )
    fig.clf()
    plt.close(fig)

    return


def plot_and_save_track_flow_field_deviations(
    mean_track_deviation_dfs: pd.DataFrame, out_subdir: Path, dataset_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=mean_track_deviation_dfs, x="track_angular_deviation_deg", binwidth=1, ax=ax)
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Counts")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir,
        figure_name=f"{dataset_name}_angular_deviation_histogram",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=mean_track_deviation_dfs, x="track_angular_deviation_deg", y="pc1_pc2_vec_mag", binwidth=(1, None), ax=ax)  # type: ignore[arg-type]
    ax.axvline(90, ls="--", lw=1, c="k", label="90 deg")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.minorticks_on()
    ax.set_xlabel("Angular deviation (deg)")
    ax.set_ylabel("Track PC1-PC2\nvector magnitude")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir,
        figure_name=f"{dataset_name}_angular_deviation_vs_mag_histogram",
    )
    plt.close(fig)


def overlay_flow_fields_on_histograms(
    dataset_name: str,
    out_subdir: Path,
    diffae_grid_crops: pd.DataFrame,
    merged_feats_df: pd.DataFrame,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    g1_tracks: np.ndarray,
    g2_tracks: np.ndarray,
    slice_indexes: tuple,
) -> None:
    # Plot flow fields overlaid on the PC1 vs PC2
    # histograms to get an idea of where the flow
    # fields have the most data to work with

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=diffae_grid_crops,
        x="pc1",
        y="pc2",
        bins=50,
        cmap="Blues",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_grids, v2_grids),
        grid=(g1_grids, g2_grids),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir,
        figure_name=f"{dataset_name}_grid_crops_pc1_pc2_hist2d",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(
        data=merged_feats_df,
        x="pc1",
        y="pc2",
        bins=50,
        cmap="Reds",
        ax=ax,
    )
    plot_one_slice_quiver(
        velocities=(v1_tracks, v2_tracks),
        grid=(g1_tracks, g2_tracks),
        slice_indexes=slice_indexes,
        ax=ax,
        color="black",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save_plot_to_path(
        figure=fig,
        output_path=out_subdir,
        figure_name=f"{dataset_name}_tracked_crops_pc1_pc2_hist2d",
    )
    plt.close(fig)


def plot_and_save_track_flow_field_dot_product_histogram(
    features_dataframe: pd.DataFrame,
    feature_column_name: str,
    out_dir: Path,
    filename: str,
    plot_title: str | None = None,
) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(data=features_dataframe, x=feature_column_name, ax=ax)
    ax.axvline(0, color="red", linestyle="--", label="perpendicular")
    ax.minorticks_on()
    plt.xticks(rotation=30, ha="right")
    ax.set_xlabel("PC1-PC2 vector dot product\n(grid-based vs. cell-centric)")
    if plot_title is not None:
        ax.set_title(plot_title)
    save_plot_to_path(
        figure=fig,
        output_path=out_dir,
        figure_name=filename,
    )
    return fig, ax
