import logging
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

from endo_pipeline.configs.dataset_config import DatasetConfig
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_normalized_time,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_flow_field_slices,
    plot_one_slice_quiver,
)
from endo_pipeline.settings import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_3d import QUIVER_COLORMAP

logger = logging.getLogger(__name__)


def set_global_pc_lims(axs: Sequence[plt.Axes], lim: int = 3) -> None:
    """Set global PC limits for all axes in axs based on lim.

    Parameters
    ----------
    axs:
        Sequence of matplotlib Axes to set limits for.
    lim:
        Limit value for both x and y axes. Axes will be set to [-lim, lim].

    Notes
    -----
    - lim corresponds to the number of standard deviations along each PC axis in the manuscript.
    - using global PC limits allows for direct comparison of positioning but may result in lots
        of empty space in some plots.
    """
    for ax in axs:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)


def get_coarse_grained_trajectory_heatmap_data(
    df_all_positions: pd.DataFrame,
    bounds: np.ndarray | list,
    num_bins: list[int] = [150, 150, 150],
    pc_cols: list[str | Column.DiffAEData] | None = None,
    feature_to_use: str = Column.SegData.NORMALIZED_TIME_PER_TRACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get a coarse-grained trajectory heatmap data from the DataFrame.

    Parameters
    ----------
    df_all_positions
        DataFrame containing tracks for one microscope position.
    bounds
        Bounds for the heatmap in each dimension.
        Should be a list of tuples or a 2D numpy array with shape (ndim, 2),
        where ndim is the number of dimensions.
    num_bins
        Number of bins for each dimension in the heatmap.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the heatmap data and the bin counts.
    """
    if feature_to_use not in df_all_positions.columns:
        raise ValueError(f"Feature '{feature_to_use}' not found in DataFrame columns.")

    if pc_cols is None:
        pc_cols = list(DYNAMICS_COLUMN_NAMES)

    bin_data = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)
    ndim = len(pc_cols)
    bins_list = [np.linspace(bounds[i][0], bounds[i][1], num_bins[i]) for i in range(ndim)]

    for _, df_one_position in df_all_positions.groupby(Column.POSITION):
        for _, df_track in df_one_position.groupby(Column.TRACK_ID):
            trajectory = df_track[pc_cols].values
            feature_values = df_track[feature_to_use].values
            bin_indices = np.zeros((trajectory.shape[0], ndim), dtype=int)
            for dim in range(ndim):
                # get the bin index in which each timepoint lies
                bin_indices_one_dim = np.digitize(trajectory[:, dim], bins_list[dim]) - 1
                # clip the bin indices to be within the valid range
                bin_indices_one_dim = np.clip(bin_indices_one_dim, 0, num_bins[dim] - 1)
                bin_indices[:, dim] = bin_indices_one_dim
            # increment the bin data and count
            for i in range(trajectory.shape[0]):
                bin_data[tuple(bin_indices[i])] += feature_values[i]
                bin_counts[tuple(bin_indices[i])] += 1

    return bin_data, bin_counts


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
            mean_over_crops = df.groupby(Column.TIMEPOINT).mean(numeric_only=True)
            # get last time point
            mean_over_crops = mean_over_crops.iloc[-1]
            pc3_val = mean_over_crops["pc_3"].mean()
            pc2_val = mean_over_crops["pc_2"].mean()
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


def plot_quiver_slices_from_flow_field_dict(
    dataset_name: str,
    flow_field_dict_grids: dict,
    feature_vals: tuple[float, float],
    flow_field_colormap: str = QUIVER_COLORMAP,
    column_names: list[str | Column.DiffAEData] | None = None,
) -> tuple[Figure, np.ndarray]:

    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    # get limits of grid from the grid crops flow fields
    bounds = get_grid_bounds(flow_field_dict_grids)

    # baseline visualization: plot flow field slices
    fig, axs = plot_flow_field_slices(
        flow_field_dict=flow_field_dict_grids,
        dataset_name=dataset_name,
        plot_bounds=bounds,
        fig_savedir=None,
        feature_vals=feature_vals,
        colormap_name=flow_field_colormap,
        log_norm_colormap=True,
        column_names=column_names,
    )
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_zorder(0)

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
    color_map: str = "flare",
    indicate_track_start: bool = True,
    indicate_track_end: bool = True,
    legend: Literal["auto", "brief", "full", False] = "auto",
    zorder: int = 0,
    alpha: float = 1.0,
) -> tuple[Figure, np.ndarray]:

    pc_cols = list({*pc_cols_for_xaxis, *pc_cols_for_yaxis})

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
                measured_feat_df.groupby(Column.TIMEPOINT)
                .mean(numeric_only=True)[pc_cols + [meas_feat_col]]
                .reset_index()
            )
        elif isinstance(track_id, int):
            measured_feat_df = measured_feat_df[measured_feat_df[Column.TRACK_ID] == track_id]
        elif track_id is None:
            pass  # do not subset or aggregate the data in any way
        else:
            raise ValueError(
                "track_ids must be 'mean', an integer, or None. "
                f"Got {track_id} (type: {type(track_id)}) instead."
            )

        if track_id is not None:
            ax.plot(
                measured_feat_df[pc_cols_for_xaxis[j]],
                measured_feat_df[pc_cols_for_yaxis[j]],
                lw=1,
                color="black",
                markersize=10,
                alpha=alpha,
                zorder=max(0, zorder),
            )
        sns.scatterplot(
            data=measured_feat_df,
            x=pc_cols_for_xaxis[j],
            y=pc_cols_for_yaxis[j],
            hue=meas_feat_col,
            hue_norm=hue_norm,
            palette=color_map,
            linewidth=0,
            marker=".",
            s=50,
            alpha=alpha,
            ax=ax,
            zorder=zorder + 1,
            legend=legend,
        )
        if indicate_track_start:
            first_timepoint_record = measured_feat_df.loc[
                measured_feat_df[Column.TIMEPOINT].idxmin()
            ]
            ax.scatter(
                first_timepoint_record[pc_cols_for_xaxis[j]],
                first_timepoint_record[pc_cols_for_yaxis[j]],
                s=100,
                edgecolor=(0, 0, 0, alpha),
                facecolor=(0, 0, 0, 0),
                marker="d",
                lw=1,
                zorder=zorder + 2,
            )
        if indicate_track_end:
            last_timepoint_record = measured_feat_df.loc[
                measured_feat_df[Column.TIMEPOINT].idxmax()
            ]
            ax.scatter(
                last_timepoint_record[pc_cols_for_xaxis[j]],
                last_timepoint_record[pc_cols_for_yaxis[j]],
                s=100,
                edgecolor=(0, 0, 0, alpha),
                facecolor=(0, 0, 0, 0),
                lw=2,
                marker="*",
                zorder=zorder + 3,
            )

    return fig, axs  # type: ignore[return-value]


def overlay_feature_on_flowfield(
    flowfield_fig_and_axs: tuple[plt.Figure, np.ndarray],
    cellcentric_df: pd.DataFrame,
    column_names: list[str],
    column_name_for_color_coding: str,
    indicate_track_start: bool = True,
    indicate_track_end: bool = True,
    track_id_to_plot: Literal["mean"] | int | None = "mean",
    hue_norm: tuple[float, float] | None = None,
    legend: Literal["auto", "brief", "full", False] = "auto",
    alpha: float = 0.7,
    use_global_pc_lims: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axs = flowfield_fig_and_axs
    fig, axs = plot_measured_feat_pcs(
        measured_feat_df=cellcentric_df,
        meas_feat_col=column_name_for_color_coding,
        pc_cols_for_xaxis=[column_names[0], column_names[0]],
        pc_cols_for_yaxis=[column_names[1], column_names[2]],
        track_id=track_id_to_plot,
        indicate_track_start=indicate_track_start,
        indicate_track_end=indicate_track_end,
        fig=fig,
        axs=axs,
        hue_norm=hue_norm,
        legend=legend,
        zorder=5,
        alpha=alpha,
    )
    plt.tight_layout()

    # change the data aspect so that X and Y have the same scaling (e.g. distances along PC1 and PC2
    # axes will be the same and directly comparable).
    for ax in axs:
        ax.set_aspect("equal")

    # changing the data aspect above can result in plots with different rectangular
    # shapes, so here we allow setting global PC limits to make all plots the same size.
    # This will have the side effect of making all positioning and vector shapes comparable
    # but may lead to varying (sometimes large) amounts of empty space in each plot.
    # In the manuscript the PCs range from -3 to 3 standard deviations, so we are using lim=3.
    if use_global_pc_lims:
        set_global_pc_lims(axs, lim=3)  # type:ignore[arg-type]

    return fig, axs


def save_feature_flowfield_overlay(
    out_dir: Path,
    flow_field_figure: plt.Figure,
    dataset_name: str,
    column_name_for_color_coding: str,
    track_id_to_plot: Literal["mean"] | int | None = "mean",
    show_plot: bool = False,
    figure_format: Literal[".png", ".svg", ".pdf"] = ".png",
) -> None:
    if track_id_to_plot == "mean":
        data_subset = "_timeAvgTracks"
    elif isinstance(track_id_to_plot, int):
        data_subset = f"_tid{track_id_to_plot}"
    elif track_id_to_plot is None:
        data_subset = ""
    else:
        raise ValueError(
            "track_ids must be 'mean', an integer, or None. "
            f"Got {track_id_to_plot} (type: {type(track_id_to_plot)}) instead."
        )

    save_plot_to_path(
        figure=flow_field_figure,
        output_path=out_dir,
        figure_name=f"{dataset_name}{data_subset}_{column_name_for_color_coding}Hue",
        file_format=figure_format,
    )
    if not show_plot:
        plt.close(flow_field_figure)


# def plot_new_traj_overlay_on_grid_traj_and_flowfield(
#     out_dir: Path,
#     dataset_name: str,
#     # fixed_points_df: pd.DataFrame | None,
#     flow_field_dict_grids: dict,
#     traj_tracks: np.ndarray,
#     figure_format: Literal[".png", ".svg", ".pdf"] = ".png",
#     use_global_pc_lims: bool = False,
# ) -> None:
#     fig, axs = plot_quiver_slices_from_flow_field_dict(dataset_name, flow_field_dict_grids)
#     for j, ax in enumerate(axs):  # PC1 vs PC2, PC1 vs PC3
#         ax.plot(traj_tracks[:, 0], traj_tracks[:, j + 1], lw=2, color="crimson")
#         ax.scatter(
#             traj_tracks[-1, 0],
#             traj_tracks[-1, j + 1],
#             s=50,
#             color="black",
#             marker="*",
#             zorder=10,
#         )
#     if use_global_pc_lims:
#         [ax.set_xlim(-3, 3) for ax in axs]
#         [ax.set_ylim(-3, 3) for ax in axs]

#     save_plot_to_path(
#         figure=fig,
#         output_path=out_dir,
#         figure_name=f"{dataset_name}_trajectory_grids_vs_tracks",
#         file_format=figure_format,
#     )
#     plt.close(fig)


def overlay_trajectory_heatmap_on_flowfield(
    out_dir: Path,
    dataset_name: str,
    flow_field_dict_grids: dict,
    feature_vals: tuple[float, float],
    dynamics_columns: list[str],
    cellcentric_df: pd.DataFrame,
    bin_widths: tuple[float, float, float],
) -> None:
    """
    Overlay a coarse-grained trajectory heatmap on the flow field.

    Parameters
    ----------
    out_dir
        Directory to save the plot to.
    dataset_name
        Name of the dataset to use for the plot.
    flow_field_dict_grids
        Dictionary containing the flow field data for the grids.
    cellcentric_df
        DataFrame containing all positions and tracks.
    num_bins
        Number of bins to use for the heatmap in each dimension.
    """
    # plot flow field
    fig, axs = plot_quiver_slices_from_flow_field_dict(
        dataset_name=dataset_name,
        flow_field_dict_grids=flow_field_dict_grids,
        feature_vals=feature_vals,
        column_names=dynamics_columns,
    )
    bounds = get_grid_bounds(flow_field_dict_grids)
    bins, _ = get_bins(bin_widths, bin_limits=bounds)

    project_axis = [2, 1]  # this is axis for projecting binned data for each plot
    plot_dim = [1, 2]  # this is the PC dimension plotted on the y-axis against PC1

    cellcentric_df = add_normalized_time(cellcentric_df)
    bin_data, bin_counts = get_coarse_grained_trajectory_heatmap_data(
        df_all_positions=cellcentric_df,
        bounds=bounds,
        num_bins=[len(b) - 1 for b in bins],
        pc_cols=list(DYNAMICS_COLUMN_NAMES),
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
            cmap="binary",
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


PlotMeasFeatAndFlowFieldOverlayArgs = namedtuple(
    "PlotMeasFeatAndFlowFieldOverlayArgs",
    [
        "out_subdir_single_position",
        "dataset_name",
        "flow_field_dict_grids",
        "flow_field_slices",
        "fixed_points_at_slices",
        "df_one_position",
        "measured_feature",
        "dynamics_columns",
        "track_id",
        "hue_norm",
        "legend",
        "figure_format",
        "use_global_pc_lims",
    ],
)


def multiproc_plot_measured_feat_overlay_on_flowfield(
    args: PlotMeasFeatAndFlowFieldOverlayArgs,
) -> None:
    (
        out_subdir_indiv_pos,
        dataset_name,
        flow_field_dict_grids,
        flow_field_slices,
        fixed_points_at_slices,
        df_one_position,
        measured_feature,
        dynamics_columns,
        track_id,
        hue_norm,
        legend,
        figure_format,
        use_global_pc_lims,
    ) = args

    fig, axs = plot_quiver_slices_from_flow_field_dict(
        dataset_name=dataset_name,
        flow_field_dict_grids=flow_field_dict_grids,
        feature_vals=flow_field_slices,
        column_names=dynamics_columns,
    )
    for j, ax in enumerate(axs):
        ax.scatter(*fixed_points_at_slices[j], c="k", s=50)
    fig, axs = overlay_feature_on_flowfield(
        flowfield_fig_and_axs=(fig, axs),
        cellcentric_df=df_one_position,
        column_names=dynamics_columns,
        column_name_for_color_coding=measured_feature,
        indicate_track_start=False,
        indicate_track_end=True,
        track_id_to_plot=track_id,
        hue_norm=hue_norm,
        legend=legend,
        alpha=0.8,
        use_global_pc_lims=use_global_pc_lims,
    )
    save_feature_flowfield_overlay(
        out_dir=out_subdir_indiv_pos,
        flow_field_figure=fig,
        dataset_name=dataset_name,
        column_name_for_color_coding=measured_feature,
        track_id_to_plot=track_id,
        show_plot=False,
        figure_format=figure_format,
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
        downsample_factor=ds,
        scale=scale,
        ax=ax,
        color="blue",
    )
    plot_one_slice_quiver(
        velocities=(v1_tracks, v2_tracks),
        grid=(g1_tracks, g2_tracks),
        slice_indexes=slice_indexes,
        downsample_factor=ds,
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
        df["pc_1"].iloc[:-1],
        df["pc_2"].iloc[:-1],
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
        x="pc_1",
        y="pc_2",
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
        x="pc_1",
        y="pc_2",
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
        df["pc_1"].iloc[:-1],
        df["pc_2"].iloc[:-1],
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
        x="pc_1",
        y="pc_2",
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
        x="pc_1",
        y="pc_2",
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


def plot_trajectory_measured_vs_simulation_over_flow_field_helper(args: dict) -> None:
    """Helper function to call plot_trajectory_measured_vs_simulation_over_flow_field with
    a dictionary of arguments.
    """
    plot_trajectory_measured_vs_simulation_over_flow_field(**args)


def plot_trajectory_measured_vs_simulation_over_flow_field(
    crop_index: int,
    traj_df: pd.DataFrame,
    fixed_point_id: int,
    fixed_point_row: pd.Series,
    flow_field_dict_grid: dict,
    out_dir: Path,
) -> None:
    """Plot measured vs simulated trajectories over a flow field slice taken at a fixed point.

    Parameters
    ----------
    crop_index
        The crop index corresponding to the trajectory being plotted.
    traj_df
        DataFrame containing the measured and simulated trajectory for the crop.
    fixed_point_id
        ID of the fixed point corresponding to the flow field slice being plotted.
    fixed_point_row
        Series containing the fixed point coordinates for the slice.
    flow_field_dict_grid
        Dictionary representing the flow field over the grid to overlay on the plot.
    out_dir
        Directory to save the resulting plot.
    """
    dataset_name = traj_df[Column.DATASET].dropna().unique().item()
    column_names = list(map(str, DYNAMICS_COLUMN_NAMES))

    flow_field_slices = (
        fixed_point_row[column_names[2]],
        fixed_point_row[column_names[1]],
    )  # feature 3, feature 2
    fixed_points_at_slices = (
        fixed_point_row[column_names].drop(index=[column_names[2]]),
        fixed_point_row[column_names].drop(index=[column_names[1]]),
    )

    unwrapped_angle_diff = (
        traj_df[f"{Column.DiffAEData.POLAR_ANGLE}_simulated_unwrapped"].diff().replace(np.nan, True)
    )
    wrapped_angle_diff = (
        traj_df[f"{Column.DiffAEData.POLAR_ANGLE}_simulated"].diff().replace(np.nan, True)
    )
    wrap_discontinuity = cast(pd.Series, ~(unwrapped_angle_diff == wrapped_angle_diff))
    angle_segments_to_plot_indices = cast(
        list[pd.Series],
        np.split(
            wrap_discontinuity,
            wrap_discontinuity.reset_index().index[wrap_discontinuity].tolist(),
        ),
    )

    # plot the underlying flow field slices
    fig, axs = plot_quiver_slices_from_flow_field_dict(
        dataset_name=dataset_name,
        flow_field_dict_grids=flow_field_dict_grid,
        feature_vals=flow_field_slices,
        column_names=column_names,
    )

    # for each axis corresponding to a flow field slice
    for j, ax in enumerate(axs):
        # add the fixed points to the plot
        ax.scatter(*fixed_points_at_slices[j], c="k", s=50)

        cols_measured: list[str] = fixed_points_at_slices[j].index.tolist()
        cols_simulated: list[str] = [f"{col}_simulated" for col in cols_measured]

        # plot measured trajectory points
        sns.scatterplot(
            data=traj_df,
            x=cols_measured[0],
            y=cols_measured[1],
            hue=Column.TIMEPOINT,
            palette="flare",
            marker="D",
            edgecolor="black",
            alpha=0.7,
            s=10,
            ax=ax,
        )
        # plot simulated trajectory segments (as segments to account for periodic data that wraps)
        for segment_indices in angle_segments_to_plot_indices:
            data_segment = traj_df.loc[segment_indices.index]
            ax.plot(
                data_segment[cols_simulated[0]],
                data_segment[cols_simulated[1]],
                ls="--",
                lw=1,
                alpha=0.7,
                c="black",
                zorder=10,
            )
        sns.scatterplot(
            data=traj_df,
            x=cols_simulated[0],
            y=cols_simulated[1],
            hue=Column.TIMEPOINT,
            palette="flare",
            edgecolor=None,
            marker="o",
            alpha=0.7,
            s=10,
            zorder=9,
            ax=ax,
        )
        ax.legend(ncols=2)
    save_plot_to_path(
        figure=fig,
        output_path=out_dir,
        figure_name=f"{dataset_name}_fp{fixed_point_id}_crop{crop_index}_traj_meas_vs_sim.png",
        show_and_close=False,
    )
    plt.close(fig)


def plot_time_of_first_passage_histogram(
    fixed_point_id: int,
    dataset_config: DatasetConfig,
    time_of_first_passage_df: pd.DataFrame,
    out_dir: Path,
    crop_pattern: Literal["grid", "tracked"] = "grid",
) -> None:
    """Plot the time of first passage for the given trajectory data.

    Parameters
    ----------
    dataset_config
        Configuration object for the dataset.
    time_of_first_passage_df
        DataFrame containing the time of first passage data.
    columns
        List of column names to consider for the time of first passage calculation.
    threshold
        Threshold value for determining the time of first passage.
    out_dir
        Directory to save the resulting plot.
    """
    dataset_name = dataset_config.name

    # replace the NaN values (which indicate there was never a first passage) with
    # a large number because the NaNs cause incorrect histogram plotting
    time_of_first_passage_df_sub = time_of_first_passage_df.copy()
    time_of_first_passage_df_sub[
        f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}"
    ].replace({np.nan: dataset_config.duration + 1}, inplace=True)
    time_of_first_passage_df_sub[
        f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated"
    ].replace({np.nan: dataset_config.duration + 1}, inplace=True)

    num_traj_approached_fp_meas = (
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}"
        ]
        < dataset_config.duration
    ).sum()
    num_traj_approached_fp_sim = (
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated"
        ]
        < dataset_config.duration
    ).sum()
    num_crops = time_of_first_passage_df_sub[Column.CROP_INDEX].nunique()

    fig, ax = plt.subplots()
    ax.set_title(
        (
            f"{dataset_name} trajectories reaching fixed point {fixed_point_id}: "
            f"\ngrid ({num_traj_approached_fp_meas} / {num_crops}) "
            f"vs simulated ({num_traj_approached_fp_sim} / {num_crops})"
        ).title()
    )
    # plot histogram for the grid-based times of first passage (if any)
    if (
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}"
        ].nunique()
        > 1
    ):
        sns.histplot(
            data=time_of_first_passage_df_sub,
            x=f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}",
            binwidth=1,
            cumulative=True,
            element="step",
            fill=False,
            stat="percent",
            ax=ax,
        )
    # plot histogram for the simulation-based times of first passage (if any)
    if (
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated"
        ].nunique()
        > 1
    ):
        sns.histplot(
            data=time_of_first_passage_df_sub,
            x=f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated",
            binwidth=1,
            cumulative=True,
            element="step",
            fill=False,
            stat="percent",
            ax=ax,
        )
    ax.set_xlim(0, dataset_config.duration)
    ax.set_ylim(0)
    ax.axhline(100, ls="--", color="red")
    ax.set_xlabel(
        (
            f"time of first passage through fixed point {fixed_point_id} (relative track start)"
        ).title()
    )
    ax.set_ylabel(
        (f"cumulative percentage of trajectories reaching fixed point {fixed_point_id}").title()
    )
    save_plot_to_path(
        fig,
        out_dir,
        f"{crop_pattern}_trajectories_approaching_fp_{fixed_point_id}_histogram.png",
        show_and_close=False,
    )


def plot_time_of_first_passage_scatterplot(
    fixed_point_id: int,
    dataset_config: DatasetConfig,
    time_of_first_passage_df: pd.DataFrame,
    out_dir: Path,
    crop_pattern: Literal["grid", "tracked"] = "grid",
) -> None:
    dataset_name = dataset_config.name

    time_of_first_passage_df_sub = time_of_first_passage_df.dropna()

    if time_of_first_passage_df_sub.size < 2:
        logger.warning(
            f"Fewer than 2 trajectories reached fixed point {fixed_point_id} "
            f"for both {crop_pattern} and simulation in dataset {dataset_name}."
        )
        return

    line_fit = linregress(
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}"
        ],
        time_of_first_passage_df_sub[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated"
        ],
    )

    trajectories_approached_fp_measured = (
        time_of_first_passage_df[
            f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}"
        ]
        .notna()
        .sum()
    )
    trajectories_approached_fp_simulated = (
        time_of_first_passage_df[f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated"]
        .notna()
        .sum()
    )
    trajectories_total = time_of_first_passage_df[Column.CROP_INDEX].nunique()
    traj_details = (
        f"reached fixed point {fixed_point_id}:"
        f"\ntracked {trajectories_approached_fp_measured / trajectories_total:.1%}, "
        f"simulated {trajectories_approached_fp_simulated / trajectories_total:.1%}"
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(
        f"{dataset_name} ({crop_pattern}), R = {line_fit.rvalue:.2f}\n{traj_details}".title()
    )
    sns.scatterplot(
        data=time_of_first_passage_df,
        x=f"time_of_first_passage_dist_from_fp_{fixed_point_id}_{crop_pattern}",
        y=f"time_of_first_passage_dist_from_fp_{fixed_point_id}_simulated",
        markers="o",
        c="black",
        s=10,
        ax=ax,
    )
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.axline(
        (0, 0), (dataset_config.duration, dataset_config.duration), ls="--", c="grey", zorder=0
    )
    ax.axline(
        (0, line_fit.intercept),
        (dataset_config.duration, line_fit.slope * dataset_config.duration + line_fit.intercept),
        ls="--",
        c="tab:orange",
        zorder=0,
    )
    ax.set_xlabel(f"{crop_pattern} trajectory first passage time".title())
    ax.set_ylabel("simulated trajectory first passage time".title())
    save_plot_to_path(
        fig,
        out_dir,
        f"{crop_pattern}_trajectories_approaching_fp_{fixed_point_id}_scatter.png",
        show_and_close=False,
    )
