import logging
from collections import namedtuple
from collections.abc import Sequence
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
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_normalized_time,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.feature_viz import (
    get_dataset_color,
    get_label_for_column,
)
from endo_pipeline.library.visualize.diffae_features.flow_field_3d import (
    get_slice_indexes,
    plot_flow_field_slices,
    plot_one_slice_quiver,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_LARGE, FONTSIZE_SMALL
from endo_pipeline.settings.flow_field_3d import QUIVER_COLORMAP
from endo_pipeline.settings.unicode import UnicodeCharacters

logger = logging.getLogger(__name__)

plt.style.use("endo_pipeline.figure")


def _get_shear_stress_for_dataset(dataset_name: str, binned: bool) -> float | int:
    dataset_config = load_dataset_config(dataset_name)
    shear_stresses = dataset_config.flow_conditions
    if len(shear_stresses) > 1:
        raise ValueError(
            f"Dataset [{dataset_name}] has multiple flow conditions with shear stresses: {shear_stresses}. "
            "This function expects only one flow condition per dataset."
        )
    return shear_stresses[0].shear_stress_bin if binned else shear_stresses[0].shear_stress


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
        set_global_pc_lims(axs, lim=3)  # type: ignore[arg-type]

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


def plot_first_passage_time_3d_scatter(
    dataset_name: str,
    fixed_point_id: int,
    fixed_point_stability: str,
    first_passage_time_df: pd.DataFrame,
    metric_to_plot: Literal["mean", "median"],
    out_dir: Path,
) -> None:
    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_to_plot == "median" else metric_to_plot

    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
    metric = f"{metric}{suffix}"

    fig = plt.figure(figsize=(3, 3.5))
    ax: Axes3D = fig.add_subplot(projection="3d")
    ax.set_title(f"{dataset_name}".title())
    thetas, rs, rhos = zip(*first_passage_time_df[Column.VectorField.BIN_CENTER], strict=True)
    # we're using a base-2 log of the FPT-tracked to FPT-grid ratio so that the
    # fold change is symmetric and the colors end up evenly spaced regardless of
    # whether the tracked or grid-based FPT is higher
    colors = np.log2(
        first_passage_time_df[f"{metric}_tracked"] / first_passage_time_df[f"{metric}_grid"]
    )
    cmap_lim = max(abs(colors))
    cmap = "coolwarm_r"
    norm = TwoSlopeNorm(vcenter=0, vmin=-cmap_lim, vmax=cmap_lim)
    scatter3d = ax.scatter(
        xs=thetas,
        ys=rs,
        zs=rhos,
        c=colors,
        cmap=cmap,
        norm=norm,
    )
    fp_dynamics_cols = [
        f"{Column.VectorField.FIXED_POINT_PREFIX}{col}" for col in DYNAMICS_COLUMN_NAMES
    ]
    fixed_point = [first_passage_time_df[col].unique().item() for col in fp_dynamics_cols]
    ax.scatter(
        *fixed_point,
        color="black",
        s=10,
        marker="*",
    )
    ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE))
    ax.set_ylabel(get_label_for_column(Column.DiffAEData.POLAR_RADIUS))
    ax.set_zlabel(get_label_for_column(Column.DiffAEData.PC3_FLIPPED))
    # adjust the focal length of the 3D plot so that depth is easier to perceive
    ax.set_proj_type("persp", focal_length=0.5)

    # add colorbar
    cax = fig.add_axes((1.15, 0.2, 0.05, 0.6))
    fig.colorbar(scatter3d, cax=cax)

    filename = (
        f"{dataset_name}_FPT_fp_{fixed_point_id}_{fixed_point_stability}"
        f"_{metric_to_plot}_3d_scatter"
    )
    save_plot_to_path(
        fig,
        out_dir,
        filename,
        tight_layout=False,
        pad_inches=0.2,
        show_and_close=False,
        file_format=".svg",
        bbox_inches="tight",
    )


def plot_first_passage_time_parameter_sweep(
    dataset_name: str,
    fixed_point_index: int,
    fixed_point_stability: str,
    first_passage_time_param_sweep_df: pd.DataFrame,
    fixed_point_radius_threshold: float | None,
    out_dir: Path,
    metric_to_plot: Literal["mean"],
    figsize=(3, 3),
) -> tuple[Path, Path]:
    """Plot the results of the parameter sweep over the number of bins in the
    initial conditions histogram and the choice of mean vs. median FPT to plot.
    """
    shear_stress_rounded = _get_shear_stress_for_dataset(dataset_name, binned=True)

    fig_title = f"{shear_stress_rounded} dyn/cm{UnicodeCharacters.SQUARED}"
    metric = "50%" if metric_to_plot == "median" else metric_to_plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(fig_title, fontsize=FONTSIZE_LARGE)
    ax.errorbar(
        x=first_passage_time_param_sweep_df[Column.VectorField.FPT_DISTANCE_THRESHOLD],
        y=first_passage_time_param_sweep_df[f"{metric}_grid"],
        yerr=first_passage_time_param_sweep_df["std_grid"],
        label=f"MFPT {UnicodeCharacters.PLUS_MINUS} STD (patch-based)",
        fmt="o-",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1,
        capsize=3,
    )
    ax.errorbar(
        x=first_passage_time_param_sweep_df[Column.VectorField.FPT_DISTANCE_THRESHOLD],
        y=first_passage_time_param_sweep_df[f"{metric}_tracked"],
        yerr=first_passage_time_param_sweep_df["std_tracked"],
        label=f"MFPT {UnicodeCharacters.PLUS_MINUS} STD (cell-centered)",
        fmt="o-",
        color="tab:red",
        ecolor="tab:red",
        elinewidth=1,
        capsize=3,
        alpha=0.7,
    )
    if fixed_point_radius_threshold is not None:
        ax.axvline(
            fixed_point_radius_threshold,
            ls="--",
            color="black",
            label="radius used in analysis",
        )
    ax.legend(frameon=True, facecolor="white", loc="upper center")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("radius around\nfixed point".title(), fontsize=FONTSIZE_LARGE, labelpad=2)
    ax.set_ylabel(
        f"{metric_to_plot} first passage\ntime (hrs)".title(), fontsize=FONTSIZE_LARGE, labelpad=2
    )
    filename_param_sweep_fpt = f"{dataset_name}_FPT_{metric_to_plot}_vs_threshold_fp_{fixed_point_index}_{fixed_point_stability}"
    save_plot_to_path(
        fig, out_dir, filename_param_sweep_fpt, file_format=".svg", show_and_close=False
    )

    # also plot compute the fraction of trajectories that approached the fixed point
    # for each parameter combination to see how the fixed point distance threshold
    # affects the number of trajectories that are considered to have reached the
    # fixed point
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(fig_title, fontsize=FONTSIZE_LARGE)
    ax.plot(
        first_passage_time_param_sweep_df[Column.VectorField.FPT_DISTANCE_THRESHOLD],
        first_passage_time_param_sweep_df[f"{Column.VectorField.PERCENT_TRAJ_APPROACHED_FP}_grid"],
        marker="o",
        color="tab:blue",
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        ls="-",
        label="patch-based",
    )
    ax.plot(
        first_passage_time_param_sweep_df[Column.VectorField.FPT_DISTANCE_THRESHOLD],
        first_passage_time_param_sweep_df[
            f"{Column.VectorField.PERCENT_TRAJ_APPROACHED_FP}_tracked"
        ],
        marker="o",
        color="tab:red",
        markerfacecolor="tab:red",
        markeredgecolor="tab:red",
        ls="-",
        label="cell-centered",
    )
    if fixed_point_radius_threshold is not None:
        ax.axvline(
            fixed_point_radius_threshold,
            ls="--",
            color="black",
            label="radius used in analysis",
        )
    ax.legend(frameon=True, facecolor="white", loc="upper center")
    ax.set_xlim(0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("radius around\nfixed point".title(), fontsize=FONTSIZE_LARGE, labelpad=2)
    ax.set_ylabel(
        "Trajectories reaching\nfixed point (%)".title(), fontsize=FONTSIZE_LARGE, labelpad=2
    )
    filename_param_sweep_num_traj = f"{dataset_name}_FPT_percent_trajectories_vs_threshold_fp_{fixed_point_index}_{fixed_point_stability}"
    save_plot_to_path(
        fig, out_dir, filename_param_sweep_num_traj, file_format=".svg", show_and_close=False
    )

    return (
        out_dir / f"{filename_param_sweep_fpt}.svg",
        out_dir / f"{filename_param_sweep_num_traj}.svg",
    )


def plot_first_passage_time_correlations(
    dataset_name: str,
    first_passage_time_stats_df: pd.DataFrame,
    line_fit_df: pd.DataFrame,
    fixed_point_id: int,
    fixed_point_stability: str,
    out_dir: Path,
    metric_to_plot: Literal["mean"],
) -> Path:
    """Plot the correlation between the grid-based and track-based first passage
        times for a given fixed point as a scatter plot with error bars, along with
        a linear fit and the Pearson correlation coefficient.

    Parameters
    ----------
        dataset_name
            The name of the dataset being plotted.
        first_passage_time_stats_df
            A DataFrame containing the statistics of the first passage times for both
            the grid-based and track-based methods, including the mean/median and standard
            deviation for each bin of initial conditions.
        line_fit_df
            A DataFrame containing the results of the linear fit between the
            grid-based and track-based first passage times, including the slope,
            intercept, Pearson correlation coefficient, and reduced chi
            squared value for the fitted line.
        fixed_point_id
            The ID of the fixed point for which the correlations are being plotted.
        fixed_point_stability
            The stability of the fixed point.
        out_dir
            The directory where the resulting plot should be saved.
        metric_to_plot
            The metric to plot on the axes, only "mean" first passage time is supported.
            Originally "median" was also available.

    Returns
    -------
        Path
            The path to the saved plot.
    """
    shear_stress_rounded = _get_shear_stress_for_dataset(dataset_name, binned=True)
    pearson_r = line_fit_df[Column.VectorField.PEARSON_R].unique().item()

    metric = "50%" if metric_to_plot == "median" else str(metric_to_plot)
    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
    metric = f"{metric}{suffix}"

    slope = line_fit_df[Column.VectorField.LINEFIT_SLOPE].unique().item()
    intercept = line_fit_df[Column.VectorField.LINEFIT_INTERCEPT_ODR].unique().item()
    corr_metric_val = (
        line_fit_df[Column.VectorField.LINEFIT_REDUCED_CHI_SQUARED_ODR].unique().item()
    )
    corr_metric_label = (
        f"Linear Fit "
        f"({UnicodeCharacters.CHI}{UnicodeCharacters.SQUARED}{UnicodeCharacters.R_SUBSCRIPT}"
        f"={corr_metric_val:.2f})"
    )

    num_bins = (
        first_passage_time_stats_df.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.VectorField.STABILITY,
            ]
        )[Column.VectorField.BIN_INDEX]
        .nunique()
        .item()
    )

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_title(
        f"{shear_stress_rounded} dyn/cm{UnicodeCharacters.SQUARED} (R = {pearson_r:.2f})",
        fontsize=FONTSIZE_SMALL,
    )
    ax.errorbar(
        x=first_passage_time_stats_df[f"{metric}_grid"],
        y=first_passage_time_stats_df[f"{metric}_tracked"],
        xerr=first_passage_time_stats_df[f"sem{suffix}_grid"],
        yerr=first_passage_time_stats_df[f"sem{suffix}_tracked"],
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        zorder=0,
    )
    ax.scatter(
        x=first_passage_time_stats_df[f"{metric}_grid"],
        y=first_passage_time_stats_df[f"{metric}_tracked"],
        color="black",
        edgecolor="white",
        lw=0.2,
        label=f"MFPT {UnicodeCharacters.PLUS_MINUS} SEM (n={num_bins})",
    )
    ax.axline(
        xy1=(0, intercept),
        slope=slope,
        color="tab:green",
        linestyle="--",
        zorder=0,
        label=corr_metric_label,
    )
    ax.axline(xy1=(0, 0), slope=1, color="black", linestyle="--", zorder=0, label="Unity")
    ax_min = min((*ax.get_xlim(), *ax.get_ylim()))
    ax_max = max((*ax.get_xlim(), *ax.get_ylim()))
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.xaxis.set_major_locator(MaxNLocator(7, min_n_ticks=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(7, min_n_ticks=4, integer=True))
    ax.tick_params(labelsize=FONTSIZE_SMALL)
    ax.set_xlabel("Patch-based MFPT (hrs)", fontsize=FONTSIZE_SMALL, labelpad=1.0, color="tab:blue")
    ax.set_ylabel(
        "Cell-centered MFPT (hrs)", fontsize=FONTSIZE_SMALL, labelpad=1.0, color="tab:red"
    )
    ax.legend(loc="upper center")

    filename = f"{dataset_name}_FPT_fp_{fixed_point_id}_{fixed_point_stability}_{metric_to_plot}_correlation"
    save_plot_to_path(
        fig,
        out_dir,
        filename,
        file_format=".svg",
        show_and_close=False,
    )
    return out_dir / f"{filename}.svg"


def plot_first_passage_time_histogram(
    dataset_name: str,
    fixed_point_id: int,
    fixed_point_stability: str,
    first_passage_time_df: pd.DataFrame,
    metric_to_plot: Literal["mean", "median", "count"],
    bin_width_for_hist: float | None,
    out_dir: Path,
) -> None:

    dataset_color = get_dataset_color(dataset_name)

    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_to_plot == "median" else metric_to_plot

    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
    metric = f"{metric}{suffix}"

    if metric_to_plot == "count":
        xaxis_title = "Number of Trajectories in Bin"
        stat_for_hist = "count"
        yaxis_title = "number of bins".title()
    else:
        xaxis_title = f"{metric_to_plot} First Passage Time (hrs)".title()
        stat_for_hist = "probability"
        yaxis_title = "probability".title()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_title(dataset_name.title())
    sns.histplot(
        data=first_passage_time_df,
        x=f"{metric}_grid",
        stat=stat_for_hist,
        binwidth=bin_width_for_hist,
        kde=True,
        facecolor="lightgrey",
        color="black",
        alpha=0.33,
        label="grid",
        ax=ax,
    )
    sns.histplot(
        data=first_passage_time_df,
        x=f"{metric}_tracked",
        stat=stat_for_hist,
        binwidth=bin_width_for_hist,
        kde=True,
        color=dataset_color,
        hatch="..",
        fill=False,
        alpha=1.0,
        label="tracked",
        ax=ax,
    )
    ax.legend()
    ax.set_xlim(0)
    ax.set_ylabel(yaxis_title)
    ax.set_xlabel(xaxis_title)

    filename = (
        f"{dataset_name}_FPT_fp_{fixed_point_id}_{fixed_point_stability}_{metric_to_plot}_histogram"
    )
    save_plot_to_path(fig, out_dir, filename, file_format=".svg", show_and_close=False)


def plot_first_passage_time_correlation_summary(
    first_passage_time_correlation_summary_df: pd.DataFrame,
    out_dir: Path,
    filename: str,
    corr_metric_column: Column.VectorField = Column.VectorField.PEARSON_R,
    slope_column: Column.VectorField = Column.VectorField.LINEFIT_SLOPE,
    summary_fig_kwargs: dict | None = {"figsize": (6, 2.5)},
) -> None:
    """Plot a summary of the correlation results from the first passage time
    analysis across all datasets and fixed points as it will appear in the figure.
    """

    corr_metric_label = COLUMN_METADATA[corr_metric_column].label or corr_metric_column
    slope_label = COLUMN_METADATA[slope_column].label or slope_column

    # get the shear stress for the dataset and add that to the labels
    # Snap to ±1 bins; values outside any bin keep their rounded value
    first_passage_time_correlation_summary_df[Column.SHEAR_STRESS] = (
        first_passage_time_correlation_summary_df[Column.DATASET].transform(
            lambda ds: _get_shear_stress_for_dataset(ds, binned=False)
        )
    )
    first_passage_time_correlation_summary_df.sort_values(Column.SHEAR_STRESS, inplace=True)
    first_passage_time_correlation_summary_df[f"{Column.SHEAR_STRESS}_rounded"] = (
        first_passage_time_correlation_summary_df[Column.DATASET].transform(
            lambda ds: _get_shear_stress_for_dataset(ds, binned=True)
        )
    )

    xs = first_passage_time_correlation_summary_df[
        [Column.DATASET, f"{Column.SHEAR_STRESS}_rounded"]
    ].values.tolist()
    xs = [f"{load_dataset_config(dataset_name).date} ({flow})" for dataset_name, flow in xs]
    ys = first_passage_time_correlation_summary_df[corr_metric_column]
    ys2 = first_passage_time_correlation_summary_df[slope_column]

    fig, ax = plt.subplots(**(summary_fig_kwargs or {}))
    sns.stripplot(
        x=xs,
        y=ys,
        color="black",
        alpha=0.7,
        ax=ax,
    )
    ax2 = ax.twinx()
    sns.stripplot(
        x=xs,
        y=ys2,
        marker="D",
        facecolor="none",
        edgecolor="tab:orange",
        linewidth=1,
        alpha=0.7,
        ax=ax2,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel(corr_metric_label, fontsize=FONTSIZE_SMALL)
    ax2.set_ylim(0)
    ax2.set_ylabel(
        slope_label, fontsize=FONTSIZE_SMALL, rotation=270, labelpad=15, color="tab:orange"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONTSIZE_SMALL)
    ax.set_xlabel("")
    save_plot_to_path(
        fig,
        out_dir,
        filename,
        file_format=".svg",
        show_and_close=False,
    )

    fig, ax = plt.subplots(figsize=(2, 2))
    sns.histplot(
        data=first_passage_time_correlation_summary_df,
        x=corr_metric_column,
        binwidth=0.05,
        binrange=(0, 1),
        color="black",
        fill=False,
        ax=ax,
    )
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim(0, 1)
    ax.set_xlabel(corr_metric_label, fontsize=FONTSIZE_SMALL)
    ax.set_ylabel("Number of Datasets", fontsize=FONTSIZE_SMALL)
    save_plot_to_path(
        fig,
        out_dir,
        f"{filename}_histogram",
        file_format=".svg",
        show_and_close=False,
    )
